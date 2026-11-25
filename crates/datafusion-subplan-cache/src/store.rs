//! Persistent tier of the subplan cache: local disk, S3 or GCS.
//!
//! Layout under the configured location:
//!
//! ```text
//! <location>/v<format>-df<datafusion version>/
//!     results/<sha256(key)>.json                     whole subtree results
//!     results/<sha256(key)>-<writer id>.arrow
//!     segments/<sha256(base key)>/<sha256(key)>.json   per-file partial aggregates
//!     segments/<sha256(base key)>/<sha256(key)>-<writer id>.arrow
//! ```
//!
//! The `.arrow` file is an Arrow IPC file (zstd) holding the batches. The
//! `.json` manifest is written *after* it and is the commit marker: an entry
//! exists only once its manifest exists, so a crash mid-write leaves an
//! orphaned data file, never a half-visible entry. Object store PUTs are
//! atomic, so readers never see a partially written object either.
//!
//! Several processes may write the same entry at once. Two rules keep that
//! safe without any locking:
//!
//! * Every write puts its batches in a data file of its own (a random writer
//!   id in the name) and the manifest names that file. Data files are never
//!   overwritten, so a manifest always describes exactly the file it points
//!   to (batch layout included).
//! * Manifests are written create-only (`If-None-Match: *` on S3,
//!   `ifGenerationMatch=0` on GCS, an exclusive create locally). The first
//!   writer wins; a loser deletes its own data file and adopts the winner's
//!   manifest. If the winner's data file is gone, the loser replaces that
//!   manifest with a conditional update (`If-Match` / generation match), so
//!   a broken entry repairs itself without clobbering a fresh one.
//!
//! A data file can still disappear under a reader (a purge between reading
//! a manifest and fetching its data). [`PersistentStore::load`] reports that
//! as `None`, and the cache treats it as a miss.
//!
//! The DataFusion version is part of the path because cache keys are plan
//! text and partial-aggregate state layouts are internal to DataFusion: both
//! can change between releases. Entries written by another version are
//! simply never looked at.

use std::collections::HashSet;
use std::io::Cursor;
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::ipc::CompressionType;
use datafusion::arrow::ipc::reader::FileReader;
use datafusion::arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use datafusion::common::{DataFusionError, Result};
use futures::{StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutPayload, UpdateVersion};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use url::Url;

use crate::cache::EntryData;

/// Bump when the on-disk format or the meaning of keys changes.
const FORMAT_VERSION: u32 = 3;

/// SHA-256 in hex: stable across processes, machines and Rust versions
/// (unlike `std::hash::DefaultHasher`).
pub fn stable_hash(s: &str) -> String {
    Sha256::digest(s.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

/// With `CACHE_TRACE` set, print how long each store call took.
fn trace(what: &str, start: std::time::Instant) {
    if std::env::var_os("CACHE_TRACE").is_some() {
        eprintln!("    [store] {what}: {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    }
}

fn oe(e: object_store::Error) -> DataFusionError {
    DataFusionError::ObjectStore(Box::new(e))
}

/// Everything about an entry except its batches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub key: String,
    pub id: String,
    pub label: String,
    pub rows: usize,
    pub bytes: usize,
    pub produced_ms: f64,
    pub deps: Vec<String>,
    pub segment_of: Option<String>,
    /// Number of batches of each output partition, in file order.
    pub partition_batches: Vec<usize>,
    /// Name of the data file (in the manifest's directory) written together
    /// with this manifest.
    pub data_file: String,
}

/// Outcome of [`PersistentStore::put`].
pub enum Put {
    /// Our manifest was created (its path).
    Created(String),
    /// Another writer created the entry first: its manifest path and
    /// manifest. Our data file was deleted.
    Lost(String, Manifest),
}

/// One manifest as listed by [`PersistentStore::list`].
pub struct Listed {
    pub path: String,
    pub manifest: Manifest,
    /// Last-modified time of the manifest (milliseconds since the epoch).
    pub modified_ms: i64,
}

pub struct PersistentStore {
    store: Arc<dyn ObjectStore>,
    root: Path,
    /// What the user passed, for display.
    pub location: String,
}

impl std::fmt::Debug for PersistentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PersistentStore({})", self.location)
    }
}

impl PersistentStore {
    /// `location` is a local directory, a `file://` URL, or (with the `s3` /
    /// `gcs` cargo features) `s3://bucket/prefix` / `gs://bucket/prefix`.
    /// Cloud credentials come from the usual environment variables
    /// (`AWS_ACCESS_KEY_ID`, `AWS_REGION`, `GOOGLE_SERVICE_ACCOUNT`, …).
    pub fn open(location: &str) -> Result<Self> {
        let (store, prefix): (Arc<dyn ObjectStore>, Path) = if location.contains("://") {
            let url = Url::parse(location).map_err(|e| DataFusionError::Plan(format!("bad cache location {location}: {e}")))?;
            let env = std::env::vars().map(|(k, v)| (k.to_ascii_lowercase(), v));
            let (store, path) = object_store::parse_url_opts(&url, env).map_err(oe)?;
            (Arc::from(store), path)
        } else {
            std::fs::create_dir_all(location)?;
            let store = object_store::local::LocalFileSystem::new_with_prefix(location).map_err(oe)?;
            (Arc::new(store), Path::default())
        };
        let version = format!("v{FORMAT_VERSION}-df{}", datafusion::DATAFUSION_VERSION);
        Ok(Self { store, root: prefix.clone().join(version.as_str()), location: location.to_string() })
    }

    fn dir(&self, segment_of: Option<&str>) -> Path {
        match segment_of {
            Some(base) => self.root.clone().join("segments").join(stable_hash(base).as_str()),
            None => self.root.clone().join("results"),
        }
    }

    /// Path of the manifest of the entry with this key.
    pub fn manifest_path(&self, key: &str, segment_of: Option<&str>) -> Path {
        self.dir(segment_of).join(format!("{}.json", stable_hash(key)).as_str())
    }

    fn data_path(&self, m: &Manifest) -> Path {
        self.dir(m.segment_of.as_deref()).join(m.data_file.as_str())
    }

    /// A data file name no other writer will pick: the key hash plus a
    /// random writer id.
    pub fn new_data_file(key: &str) -> String {
        use std::hash::BuildHasher;
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // RandomState is seeded from the OS's random source.
        let random = std::collections::hash_map::RandomState::new().hash_one((nanos, n));
        let writer = stable_hash(&format!("{}-{nanos}-{n}-{random}", std::process::id()));
        format!("{}-{}.arrow", stable_hash(key), &writer[..16])
    }

    /// Write an entry: its data file (named by `m.data_file`), then its
    /// manifest, create-only.
    pub async fn put(&self, m: &Manifest, data: &EntryData) -> Result<Put> {
        let options = IpcWriteOptions::default().try_with_compression(Some(CompressionType::ZSTD))?;
        let mut writer = FileWriter::try_new_with_options(Vec::new(), &data.schema, options)?;
        for batch in data.partitions.iter().flatten() {
            writer.write(batch)?;
        }
        writer.finish()?;
        let bytes = writer.into_inner()?;

        let data_path = self.data_path(m);
        let manifest_path = self.manifest_path(&m.key, m.segment_of.as_deref());
        let t = std::time::Instant::now();
        self.store.put(&data_path, PutPayload::from(bytes)).await.map_err(oe)?;
        trace("PUT data", t);
        // The manifest is the commit marker: write it last, and only if no
        // other writer has committed this entry already.
        let json = PutPayload::from(serde_json::to_vec(m).map_err(|e| DataFusionError::External(Box::new(e)))?);
        let result = self.commit(&manifest_path, json).await;
        if !matches!(result, Ok(Put::Created(_))) {
            // No manifest names our data file: remove it.
            self.delete_quietly(&data_path).await?;
        }
        result
    }

    /// Create the manifest unless a usable one exists. A manifest whose data
    /// file is gone (deleted by something other than `purge`, e.g. a bucket
    /// lifecycle rule) is replaced, but only if it is still the version we
    /// looked at, so a manifest another writer just committed is never
    /// clobbered.
    async fn commit(&self, path: &Path, json: PutPayload) -> Result<Put> {
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let created = self.store.put_opts(path, json.clone(), PutMode::Create.into()).await;
            trace("PUT manifest (create-only)", t);
            match created {
                Ok(_) => return Ok(Put::Created(path.to_string())),
                Err(object_store::Error::AlreadyExists { .. }) => {}
                Err(e) => return Err(oe(e)),
            }
            // Someone committed this entry. Adopt theirs if its data is there.
            let Some((winner, version)) = self.get_manifest(path).await? else { continue };
            if self.exists(&self.data_path(&winner)).await? {
                return Ok(Put::Lost(path.to_string(), winner));
            }
            match self.store.put_opts(path, json.clone(), PutMode::Update(version).into()).await {
                Ok(_) => return Ok(Put::Created(path.to_string())),
                // Changed since we read it: look again.
                Err(object_store::Error::Precondition { .. }) => continue,
                // The local file system has no conditional update: delete and
                // create again (only processes on this machine share it).
                Err(object_store::Error::NotImplemented { .. }) => self.delete_quietly(path).await?,
                Err(e) => return Err(oe(e)),
            }
        }
        Err(DataFusionError::Execution(format!("cache entry {path} keeps changing; not cached")))
    }

    async fn get_manifest(&self, path: &Path) -> Result<Option<(Manifest, UpdateVersion)>> {
        let r = match self.store.get(path).await {
            Ok(r) => r,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(e) => return Err(oe(e)),
        };
        let version = UpdateVersion { e_tag: r.meta.e_tag.clone(), version: r.meta.version.clone() };
        let bytes = r.bytes().await.map_err(oe)?;
        let m = serde_json::from_slice(&bytes).map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(Some((m, version)))
    }

    async fn exists(&self, path: &Path) -> Result<bool> {
        match self.store.head(path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(oe(e)),
        }
    }

    async fn delete_quietly(&self, path: &Path) -> Result<()> {
        match self.store.delete(path).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(oe(e)),
        }
    }

    /// The batches of an entry, or `None` if its data file is gone (the
    /// entry was purged after its manifest was read).
    pub async fn load(&self, m: &Manifest) -> Result<Option<EntryData>> {
        let data_path = self.data_path(m);
        let t = std::time::Instant::now();
        let bytes = match self.store.get(&data_path).await {
            Ok(r) => r.bytes().await.map_err(oe)?,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(e) => return Err(oe(e)),
        };
        trace("GET data", t);
        let reader = FileReader::try_new(Cursor::new(bytes), None)?;
        let schema = reader.schema();
        let batches: Vec<RecordBatch> = reader.collect::<Result<_, _>>()?;
        let mut it = batches.into_iter();
        let partitions = m.partition_batches.iter().map(|n| it.by_ref().take(*n).collect()).collect();
        Ok(Some(EntryData { schema, partitions }))
    }

    /// Manifests whose path is not in `skip`. One LIST plus one GET per new
    /// manifest.
    pub async fn list(&self, skip: &HashSet<String>) -> Result<Vec<Listed>> {
        let t = std::time::Instant::now();
        let metas: Vec<_> = self.store.list(Some(&self.root)).try_collect().await.map_err(oe)?;
        trace(&format!("LIST ({} objects)", metas.len()), t);
        let new: Vec<_> = metas
            .into_iter()
            .filter(|m| m.location.as_ref().ends_with(".json") && !skip.contains(m.location.as_ref()))
            .collect();
        // Fetch the new manifests concurrently: on a cloud store each GET is
        // a round trip, so one at a time would add up quickly.
        let t = std::time::Instant::now();
        let n_new = new.len();
        let store = &self.store;
        let fetched: Vec<_> = futures::stream::iter(new.into_iter().map(|meta| async move {
            let store = store;
            // Deleted between LIST and GET (purged): skip it.
            let bytes = match store.get(&meta.location).await {
                Ok(r) => Some(r.bytes().await.map_err(oe)?),
                Err(object_store::Error::NotFound { .. }) => None,
                Err(e) => return Err(oe(e)),
            };
            Ok::<_, DataFusionError>((meta.location.to_string(), meta.last_modified.timestamp_millis(), bytes))
        }))
        .buffer_unordered(32)
        .try_collect()
        .await?;
        if n_new > 0 {
            trace(&format!("GET {n_new} manifests"), t);
        }
        let mut out = vec![];
        for (path, modified_ms, bytes) in fetched {
            let Some(bytes) = bytes else { continue };
            match serde_json::from_slice::<Manifest>(&bytes) {
                Ok(manifest) => out.push(Listed { path, manifest, modified_ms }),
                Err(e) => eprintln!("  ! skipping unreadable cache manifest {path}: {e}"),
            }
        }
        Ok(out)
    }

    /// Delete an entry: manifest first (it disappears atomically for
    /// readers), then its data file.
    pub async fn delete(&self, m: &Manifest) -> Result<()> {
        self.delete_quietly(&self.manifest_path(&m.key, m.segment_of.as_deref())).await?;
        self.delete_quietly(&self.data_path(m)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

    fn manifest(key: &str, partition_batches: Vec<usize>) -> Manifest {
        Manifest {
            key: key.into(),
            id: "t".into(),
            label: "t".into(),
            rows: 0,
            bytes: 0,
            produced_ms: 0.0,
            deps: vec![],
            segment_of: None,
            partition_batches,
            data_file: PersistentStore::new_data_file(key),
        }
    }

    /// `parts` partitions of one batch each, together holding 0..12.
    fn data(parts: usize) -> EntryData {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let per = 12 / parts as i64;
        let partitions = (0..parts as i64)
            .map(|p| {
                let col = Int64Array::from_iter_values(p * per..(p + 1) * per);
                vec![RecordBatch::try_new(schema.clone(), vec![Arc::new(col)]).unwrap()]
            })
            .collect();
        EntryData { schema, partitions }
    }

    /// A fresh store: a temp directory, or a new prefix under
    /// `CACHE_TEST_STORE` (e.g. `gs://bucket/tests`) to run against a bucket.
    fn temp_store(name: &str) -> PersistentStore {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        if let Ok(base) = std::env::var("CACHE_TEST_STORE") {
            return PersistentStore::open(&format!("{}/{name}-{nanos}", base.trim_end_matches('/'))).unwrap();
        }
        let dir = std::env::temp_dir().join(format!("cache-race-{name}-{nanos}"));
        PersistentStore::open(dir.to_str().unwrap()).unwrap()
    }

    async fn files(store: &PersistentStore) -> Vec<String> {
        let metas: Vec<_> = store.store.list(Some(&store.root)).try_collect().await.unwrap();
        metas.into_iter().map(|m| m.location.to_string()).collect()
    }

    /// Many writers commit the same key at once, each with a different
    /// batch layout. Exactly one wins; every loser adopts the winner's
    /// manifest, and that manifest describes the one data file left.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_writers_of_one_entry() {
        let store = Arc::new(temp_store("writers"));
        let layouts = [1usize, 2, 3, 4, 6, 12];
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let store = store.clone();
                let parts = layouts[i % layouts.len()];
                tokio::spawn(async move { store.put(&manifest("same key", vec![1; parts]), &data(parts)).await.unwrap() })
            })
            .collect();
        let mut created = vec![];
        let mut adopted = vec![];
        for h in handles {
            match h.await.unwrap() {
                Put::Created(path) => created.push(path),
                Put::Lost(_, m) => adopted.push(m),
            }
        }
        assert_eq!(created.len(), 1, "exactly one writer commits");
        let files = files(&store).await;
        assert_eq!(files.iter().filter(|f| f.ends_with(".json")).count(), 1);
        assert_eq!(files.iter().filter(|f| f.ends_with(".arrow")).count(), 1, "losers delete their data: {files:?}");
        let listed = store.list(&HashSet::new()).await.unwrap();
        let winner = &listed[0].manifest;
        for m in &adopted {
            assert_eq!((&m.data_file, &m.partition_batches), (&winner.data_file, &winner.partition_batches));
        }
        // The manifest's layout matches its own data file.
        let d = store.load(winner).await.unwrap().unwrap();
        assert_eq!(d.partitions.len(), winner.partition_batches.len());
        let values: Vec<i64> = d
            .partitions
            .iter()
            .flatten()
            .flat_map(|b| b.column(0).as_any().downcast_ref::<Int64Array>().unwrap().values().to_vec())
            .collect();
        assert_eq!(values, (0..12).collect::<Vec<_>>());
    }

    /// A manifest whose data file was deleted behind the cache's back is
    /// replaced by the next writer instead of blocking the entry forever.
    #[tokio::test]
    async fn dangling_manifest_is_repaired() {
        let store = temp_store("dangling");
        let first = manifest("k", vec![1]);
        assert!(matches!(store.put(&first, &data(1)).await.unwrap(), Put::Created(_)));
        store.store.delete(&store.data_path(&first)).await.unwrap();
        let second = manifest("k", vec![1, 1, 1]);
        assert!(matches!(store.put(&second, &data(3)).await.unwrap(), Put::Created(_)));
        let listed = store.list(&HashSet::new()).await.unwrap();
        assert_eq!(listed[0].manifest.data_file, second.data_file);
        assert_eq!(store.load(&listed[0].manifest).await.unwrap().unwrap().partitions.len(), 3);
    }

    /// A data file deleted after its manifest was read is a miss, not an error.
    #[tokio::test]
    async fn purged_data_is_a_miss() {
        let store = temp_store("purged");
        let m = manifest("k", vec![1, 1]);
        assert!(matches!(store.put(&m, &data(2)).await.unwrap(), Put::Created(_)));
        let listed = store.list(&HashSet::new()).await.unwrap();
        store.delete(&m).await.unwrap();
        assert!(store.load(&listed[0].manifest).await.unwrap().is_none());
        // And the entry can be written again afterwards.
        assert!(matches!(store.put(&manifest("k", vec![2]), &data(2)).await.unwrap(), Put::Created(_)));
    }
}
