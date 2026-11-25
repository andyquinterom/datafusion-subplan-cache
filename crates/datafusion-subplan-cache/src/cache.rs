//! The in-memory index of cached results, backed by the optional
//! persistent store.
//!
//! Two kinds of cached results:
//!
//! * **Whole subtree results**, keyed by the subtree's canonical plan text
//!   plus the identity of every snapshot it reads.
//! * **Per-segment partial aggregates**: for an aggregate over one
//!   immutable-file table (optionally joined with static tables), the
//!   partial aggregate states of each group of data files. A new snapshot
//!   reuses the segments whose files are all still present.
//!
//! See [`crate::rewrite`] for how plans use them.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::Result;

use tokio::task::JoinHandle;

use crate::UnregisteredTables;
use crate::rewrite::RewriteReport;
use crate::store::{Manifest, PersistentStore, Put};
use crate::table::Registry;

/// Default memory budget for cached batches.
pub const DEFAULT_MAX_MEMORY: usize = 2 << 30; // 2 GiB

/// The batches of a cached result.
pub struct EntryData {
    pub schema: SchemaRef,
    pub partitions: Vec<Vec<RecordBatch>>,
}

/// One materialized result. The entry itself (key, deps, …) is always in
/// memory; its batches are in memory, in the persistent store, or both.
pub struct CacheEntry {
    pub key: String,
    pub id: String,
    /// One-line label of the subtree root, e.g. `Aggregate: groupBy=[...]`.
    pub label: String,
    pub rows: usize,
    pub bytes: usize,
    /// Time from first execution of the producing operator to its completion.
    pub produced_in: Duration,
    pub hits: AtomicUsize,
    /// Identities this result was computed from (file ids, snapshot ids,
    /// static table tokens). If any of them stops being live, the entry is garbage.
    pub deps: Vec<String>,
    /// For a per-segment partial aggregate: the key of the aggregate it
    /// belongs to. `deps` then lists exactly the files of the segment.
    pub segment_of: Option<String>,
    /// The batches, once in memory (set at creation, or on first load).
    pub data: OnceLock<Arc<EntryData>>,
    /// Batches are (also) in the persistent store.
    pub persisted: AtomicBool,
    /// Where the batches are persisted. Changes if another process
    /// committed the same entry first (we then use its copy).
    pub stored: Mutex<Stored>,
}

#[derive(Debug, Clone, Default)]
pub struct Stored {
    /// Data file in the store (empty until a store write is started).
    pub data_file: String,
    /// Number of batches in each output partition (to restore the layout).
    pub partition_batches: Vec<usize>,
}

impl CacheEntry {
    fn manifest(&self) -> Manifest {
        let stored = self.stored.lock().unwrap().clone();
        Manifest {
            key: self.key.clone(),
            id: self.id.clone(),
            label: self.label.clone(),
            rows: self.rows,
            bytes: self.bytes,
            produced_ms: self.produced_in.as_secs_f64() * 1e3,
            deps: self.deps.clone(),
            segment_of: self.segment_of.clone(),
            partition_batches: stored.partition_batches,
            data_file: stored.data_file,
        }
    }

    fn from_manifest(m: Manifest) -> Self {
        Self {
            key: m.key,
            id: m.id,
            label: m.label,
            rows: m.rows,
            bytes: m.bytes,
            produced_in: Duration::from_secs_f64(m.produced_ms / 1e3),
            hits: AtomicUsize::new(0),
            deps: m.deps,
            segment_of: m.segment_of,
            data: OnceLock::new(),
            persisted: AtomicBool::new(true),
            stored: Mutex::new(Stored { data_file: m.data_file, partition_batches: m.partition_batches }),
        }
    }
}

/// How one query used per-segment caching for one aggregate.
#[derive(Debug, Clone)]
pub struct SegmentUse {
    pub base_key: String,
    /// File ids of each reused cached segment.
    pub reused: Vec<Vec<String>>,
    /// File ids of each newly computed segment.
    pub computed: Vec<Vec<String>>,
    /// Filter conjuncts lifted out of the key (decided per file from stats).
    pub lifted: Vec<String>,
    /// File ids read live with the full predicate (not cached): files the
    /// lifted conjuncts keep only partly, or without statistics.
    pub edge: Vec<String>,
    /// Number of files the lifted conjuncts exclude entirely (not read).
    pub skipped: usize,
}

/// The shared state behind [`crate::SubplanCache`].
pub struct Cache {
    entries: RwLock<HashMap<String, Arc<CacheEntry>>>,
    events: Mutex<Vec<String>>,
    segment_uses: Mutex<Vec<SegmentUse>>,
    /// How many data files go into one newly computed segment.
    pub segment_files: AtomicUsize,
    /// See [`crate::Builder::lift_max_edge_fraction`] (f64 bits).
    pub lift_max_edge_fraction: AtomicU64,
    /// Optional persistent tier (local directory, S3, GCS).
    pub store: Option<Arc<PersistentStore>>,
    /// Manifests already read from the store (or written by us).
    seen: Mutex<HashSet<String>>,
    /// Uploads still in flight.
    pending: Mutex<Vec<JoinHandle<Result<String>>>>,
    /// Number of entries whose batches were read back from the store.
    pub loads: AtomicUsize,
    /// When the store was last listed. Listing is one LIST request (tens of
    /// ms on S3/GCS), so it is done at most every [`SYNC_INTERVAL`].
    last_sync: Mutex<Option<Instant>>,
    /// Tables and providers known to the cache.
    pub(crate) registry: Registry,
    pub(crate) policy: UnregisteredTables,
    pub(crate) enabled: AtomicBool,
    pub(crate) max_bytes: usize,
    reports: Mutex<Vec<RewriteReport>>,
}

/// How stale this process's view of entries written by *other* processes
/// may get. A missed entry only means recomputing (and re-writing) it.
pub const SYNC_INTERVAL: Duration = Duration::from_secs(5);

impl fmt::Debug for Cache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SubplanCache({} entries)", self.entries.read().unwrap().len())
    }
}

impl Cache {
    pub(crate) fn new(
        store: Option<Arc<PersistentStore>>,
        segment_files: usize,
        lift_max_edge_fraction: f64,
        policy: UnregisteredTables,
        max_bytes: usize,
    ) -> Self {
        Self {
            entries: Default::default(),
            events: Default::default(),
            segment_uses: Default::default(),
            segment_files: AtomicUsize::new(segment_files),
            lift_max_edge_fraction: AtomicU64::new(lift_max_edge_fraction.to_bits()),
            store,
            seen: Default::default(),
            pending: Default::default(),
            loads: AtomicUsize::new(0),
            last_sync: Mutex::new(None),
            registry: Registry::default(),
            policy,
            enabled: AtomicBool::new(true),
            max_bytes,
            reports: Default::default(),
        }
    }

    pub fn get(&self, key: &str) -> Option<Arc<CacheEntry>> {
        self.entries.read().unwrap().get(key).cloned()
    }

    fn resident_bytes(entries: &HashMap<String, Arc<CacheEntry>>) -> usize {
        entries.values().filter(|e| e.data.get().is_some()).map(|e| e.bytes).sum()
    }

    /// Stores a freshly computed entry (which carries its batches). With a
    /// persistent store, the upload starts in the background (see
    /// [`Self::flush`]) and the batches stay in memory only while they fit
    /// the memory budget. Without one, an entry over budget is not kept.
    pub fn insert(&self, entry: CacheEntry) -> bool {
        let mut entries = self.entries.write().unwrap();
        if entries.contains_key(&entry.key) {
            return false;
        }
        let fits = Self::resident_bytes(&entries) + entry.bytes <= self.max_bytes;
        let Some(store) = self.store.clone() else {
            if fits {
                entries.insert(entry.key.clone(), Arc::new(entry));
            }
            return fits;
        };
        let mut entry = entry;
        entry.stored.get_mut().unwrap().data_file = PersistentStore::new_data_file(&entry.key);
        let manifest = entry.manifest();
        let data = entry.data.get().cloned().expect("new entries carry their batches");
        if !fits {
            entry.data = OnceLock::new(); // keep it on disk only
        }
        let entry = Arc::new(entry);
        entries.insert(entry.key.clone(), entry.clone());
        drop(entries);
        let handle = tokio::spawn(async move {
            let path = match store.put(&manifest, &data).await? {
                Put::Created(path) => path,
                // Another process committed this entry first. Its result is
                // equivalent; point at its data file (ours is deleted).
                Put::Lost(path, winner) => {
                    *entry.stored.lock().unwrap() =
                        Stored { data_file: winner.data_file, partition_batches: winner.partition_batches };
                    path
                }
            };
            entry.persisted.store(true, Ordering::Relaxed);
            Ok(path)
        });
        self.pending.lock().unwrap().push(handle);
        true
    }

    /// Wait for all uploads started by [`Self::insert`].
    pub async fn flush(&self) -> Result<()> {
        let handles: Vec<_> = std::mem::take(&mut *self.pending.lock().unwrap());
        for h in handles {
            let path = h.await.map_err(|e| datafusion::common::DataFusionError::External(Box::new(e)))??;
            // Our own manifests don't need to be read back by sync().
            self.seen.lock().unwrap().insert(path);
        }
        Ok(())
    }

    /// Pick up entries in the persistent store that this process does not
    /// know yet: written by an earlier run, or by another process. Only the
    /// manifests are read; batches are loaded when a query needs them.
    pub async fn sync(&self) -> Result<usize> {
        let Some(store) = &self.store else { return Ok(0) };
        {
            let mut last = self.last_sync.lock().unwrap();
            if last.is_some_and(|t| t.elapsed() < SYNC_INTERVAL) {
                return Ok(0);
            }
            *last = Some(Instant::now());
        }
        let skip = self.seen.lock().unwrap().clone();
        let found = store.list(&skip).await?;
        let mut seen = self.seen.lock().unwrap();
        let mut entries = self.entries.write().unwrap();
        let mut added = 0;
        for l in found {
            let m = l.manifest;
            seen.insert(l.path);
            if !entries.contains_key(&m.key) {
                entries.insert(m.key.clone(), Arc::new(CacheEntry::from_manifest(m)));
                added += 1;
            }
        }
        Ok(added)
    }

    /// The batches of an entry, reading them from the store if needed.
    /// `None` if the entry was purged from the store meanwhile: it is then
    /// forgotten, and the caller computes the result instead.
    pub async fn try_data(&self, e: &CacheEntry) -> Result<Option<Arc<EntryData>>> {
        if let Some(d) = e.data.get() {
            return Ok(Some(d.clone()));
        }
        let store = self.store.as_ref().expect("an entry without batches comes from the store");
        let manifest = e.manifest();
        let Some(d) = store.load(&manifest).await? else {
            self.forget(e);
            self.event(format!("  ⚠ cached {} vanished from the store (purged?); recomputing it", e.id));
            return Ok(None);
        };
        let d = Arc::new(d);
        self.loads.fetch_add(1, Ordering::Relaxed);
        let fits = Self::resident_bytes(&self.entries.read().unwrap()) + e.bytes <= self.max_bytes;
        if fits {
            let _ = e.data.set(d.clone());
        }
        Ok(Some(d))
    }

    /// Like [`Self::try_data`], for callers that cannot fall back.
    pub async fn data(&self, e: &CacheEntry) -> Result<Arc<EntryData>> {
        self.try_data(e).await?.ok_or_else(|| {
            datafusion::common::DataFusionError::Execution(format!("cached {} vanished from the store", e.id))
        })
    }

    /// Drop an entry from memory and allow the next sync to read its
    /// manifest again (another process may write it anew).
    fn forget(&self, e: &CacheEntry) {
        let mut entries = self.entries.write().unwrap();
        if entries.get(&e.key).is_some_and(|x| std::ptr::eq(x.as_ref(), e)) {
            entries.remove(&e.key);
        }
        if let Some(store) = &self.store {
            self.seen.lock().unwrap().remove(store.manifest_path(&e.key, e.segment_of.as_deref()).as_ref());
        }
    }

    /// Cached segments of the aggregate with the given base key.
    pub fn segments(&self, base_key: &str) -> Vec<Arc<CacheEntry>> {
        self.entries
            .read()
            .unwrap()
            .values()
            .filter(|e| e.segment_of.as_deref() == Some(base_key))
            .cloned()
            .collect()
    }

    /// Forget (in this process) every entry that depends on a file or
    /// snapshot that is gone. Persisted copies are left alone: another
    /// reader may still be on an older snapshot. See [`Self::purge`].
    pub fn gc(&self, live: &HashSet<String>) -> usize {
        let mut entries = self.entries.write().unwrap();
        let before = entries.len();
        entries.retain(|_, e| e.deps.iter().all(|d| live.contains(d)));
        before - entries.len()
    }

    /// Delete persisted entries that no reader can still use.
    ///
    /// `live` must hold the identities of every snapshot a reader may still
    /// be on: the current one and those committed within the grace period
    /// (see [`crate::ImmutableFileTable::snapshots_since`]). An entry is deleted only if it
    /// depends on something outside `live` *and* its manifest is older than
    /// `grace`, so entries just written by a process on a snapshot this one
    /// has not seen yet are kept too. A reader that loses an entry anyway
    /// recomputes it (see [`Self::try_data`]).
    /// Entries that depend on a table `owns` does not recognize (registered
    /// only by other processes) are never deleted.
    pub async fn purge(&self, live: &HashSet<String>, owns: impl Fn(&str) -> bool, grace: Duration) -> Result<usize> {
        let Some(store) = &self.store else { return Ok(0) };
        self.flush().await?;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let all = store.list(&HashSet::new()).await?;
        let mut n = 0;
        for l in all {
            let old = now_ms - l.modified_ms >= grace.as_millis() as i64;
            let deps = &l.manifest.deps;
            if old && deps.iter().all(|d| owns(d)) && !deps.iter().all(|d| live.contains(d)) {
                store.delete(&l.manifest).await?;
                self.seen.lock().unwrap().remove(&l.path);
                n += 1;
            }
        }
        Ok(n)
    }

    /// Forget everything in memory. Persisted entries are re-discovered by
    /// the next [`Self::sync`].
    pub fn clear(&self) {
        self.entries.write().unwrap().clear();
        self.seen.lock().unwrap().clear();
        *self.last_sync.lock().unwrap() = None;
    }

    /// Read every manifest again on the next [`Self::sync`]. Called when a
    /// table is registered: entries dropped by [`Self::gc`] because they
    /// depended on a table not registered yet become usable again.
    pub fn rescan(&self) {
        self.seen.lock().unwrap().clear();
        *self.last_sync.lock().unwrap() = None;
    }

    pub fn push_report(&self, r: RewriteReport) {
        self.reports.lock().unwrap().push(r);
    }

    pub fn take_reports(&self) -> Vec<RewriteReport> {
        std::mem::take(&mut *self.reports.lock().unwrap())
    }

    pub fn event(&self, msg: String) {
        self.events.lock().unwrap().push(msg);
    }

    pub fn lift_max_edge_fraction(&self) -> f64 {
        f64::from_bits(self.lift_max_edge_fraction.load(Ordering::Relaxed))
    }

    pub fn record_segment_use(&self, u: SegmentUse) {
        self.segment_uses.lock().unwrap().push(u);
    }

    pub fn take_segment_uses(&self) -> Vec<SegmentUse> {
        std::mem::take(&mut *self.segment_uses.lock().unwrap())
    }

    pub fn take_events(&self) -> Vec<String> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }

    pub fn entries(&self) -> Vec<Arc<CacheEntry>> {
        let mut v: Vec<_> = self.entries.read().unwrap().values().cloned().collect();
        v.sort_by(|a, b| (&a.segment_of, &a.label, &a.id).cmp(&(&b.segment_of, &b.label, &b.id)));
        v
    }

    pub fn total_bytes(&self) -> usize {
        self.entries.read().unwrap().values().map(|e| e.bytes).sum()
    }
}
