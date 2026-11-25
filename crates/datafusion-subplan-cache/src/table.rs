//! Tables whose data is a set of immutable files, and the registry that
//! connects them to the plans DataFusion builds.
//!
//! The cache never trusts a table *version*. It trusts the set of data files
//! a snapshot consists of, each with an identity that changes whenever the
//! rows the file contributes can change. So:
//!   * an append adds identities and leaves the existing ones intact;
//!   * a rewrite (UPDATE/DELETE/MERGE/compaction) removes the identities of
//!     the files it replaces, because the new files have new names;
//!   * a delete file / deletion vector applied to an existing file must
//!     change that file's identity (the adapter puts it into the id).
//!
//! Any table format that never rewrites a data file in place fits: Delta,
//! Iceberg, DuckLake, Hudi copy-on-write, or a folder that only ever gets
//! new uniquely named Parquet files.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::common::{Result, ScalarValue};
use datafusion::datasource::source_as_provider;
use datafusion::logical_expr::TableSource;
use datafusion::prelude::SessionContext;

use crate::store::stable_hash;

/// Exact statistics of one column over all rows of one data file.
///
/// Only fill these when they are exact: `min`/`max` must be the true
/// smallest/largest non-null values (or bounds no tighter than them: a
/// smaller `min` or larger `max` is still correct, just less useful), and
/// `null_count`/`row_count` must be exact. `min`/`max` are null when every
/// row is null. Their type must be the column's type in the table schema.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnStats {
    pub min: ScalarValue,
    pub max: ScalarValue,
    pub null_count: u64,
    pub row_count: u64,
}

/// One data file of a snapshot.
#[derive(Debug, Clone)]
pub struct DataFile {
    /// The path the adapter understands (passed back to
    /// [`FileSnapshot::provider_for`]).
    pub path: String,
    /// Content identity. It must change whenever the rows this file
    /// contributes can change: path + size (+ etag/mtime) + any delete
    /// files / deletion vectors that apply to it.
    pub id: String,
    /// Per-column statistics, by column name (optional: columns without
    /// them simply never let the cache skip or reuse the file per
    /// predicate). Derived from the content, so not part of `id`.
    pub stats: Arc<HashMap<String, ColumnStats>>,
}

impl PartialEq for DataFile {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.id == other.id
    }
}
impl Eq for DataFile {}
impl std::hash::Hash for DataFile {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.path.hash(state);
        self.id.hash(state);
    }
}

impl DataFile {
    /// A file whose identity is `path` plus `identity` (everything else that
    /// pins its content, e.g. `"{size}|{etag}|{deletion vector}"`).
    pub fn new(path: impl Into<String>, identity: &str) -> Self {
        let path = path.into();
        let name = path.rsplit('/').next().unwrap_or(&path);
        let short = &name[..name.floor_char_boundary(12)];
        let id = format!("{short}:{}", &stable_hash(&format!("{path}|{identity}"))[..16]);
        Self { path, id, stats: Default::default() }
    }

    /// Attach exact statistics of `column` (see [`ColumnStats`]).
    pub fn with_stats(mut self, column: impl Into<String>, stats: ColumnStats) -> Self {
        Arc::make_mut(&mut self.stats).insert(column.into(), stats);
        self
    }

    /// Attach statistics of many columns at once.
    pub fn with_all_stats(mut self, stats: Arc<HashMap<String, ColumnStats>>) -> Self {
        self.stats = stats;
        self
    }
}

/// One snapshot of a table: a fixed set of immutable files, and how to read
/// all of them or a subset.
#[async_trait]
pub trait FileSnapshot: Send + Sync + fmt::Debug {
    /// For display only (a Delta version, an Iceberg snapshot id, …).
    fn version(&self) -> String;

    /// Every data file of the snapshot.
    fn files(&self) -> &[DataFile];

    /// A provider reading the whole snapshot. It is registered in the
    /// session under the table's name.
    async fn provider(&self) -> Result<Arc<dyn TableProvider>>;

    /// A provider reading only `files` of this snapshot, with any row-level
    /// deletes applied. Its schema must equal [`Self::provider`]'s.
    /// `None` if the format cannot restrict a scan: the cache then reuses
    /// whole results only, never per-file partial aggregates.
    async fn provider_for(&self, files: &[&DataFile]) -> Result<Option<Arc<dyn TableProvider>>>;
}

/// A table whose data files are immutable and uniquely named.
#[async_trait]
pub trait ImmutableFileTable: Send + Sync {
    /// The latest snapshot (re-read the log, catalog or listing). Called on
    /// [`crate::SubplanCache::register_table`] and on every
    /// [`crate::SubplanCache::refresh`].
    async fn latest(&self) -> Result<Arc<dyn FileSnapshot>>;

    /// Every snapshot a reader may still be on: those committed at or after
    /// `since`, plus the one that was current at `since`. Used by
    /// [`crate::SubplanCache::purge`] so other processes lagging behind keep
    /// their entries. The default, the latest snapshot only, is right for
    /// formats without history.
    async fn snapshots_since(&self, _since: SystemTime) -> Result<Vec<Arc<dyn FileSnapshot>>> {
        Ok(vec![self.latest().await?])
    }

    /// Called once with the session the table is registered in (e.g. to
    /// register object stores the provider needs).
    fn attach(&self, _ctx: &SessionContext) -> Result<()> {
        Ok(())
    }
}

/// A snapshot as the cache sees it: the adapter's snapshot plus the name it
/// is registered under and the hash of its file set.
pub struct Snapshot {
    pub name: String,
    /// Hash of all file ids: the identity of the snapshot's contents.
    pub fileset: String,
    pub inner: Arc<dyn FileSnapshot>,
}

impl fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{} (v{}, {} files)", self.name, self.fileset, self.inner.version(), self.files().len())
    }
}

impl Snapshot {
    pub fn new(name: &str, inner: Arc<dyn FileSnapshot>) -> Self {
        let mut ids: Vec<&str> = inner.files().iter().map(|f| f.id.as_str()).collect();
        ids.sort_unstable();
        let fileset = stable_hash(&ids.join(","))[..16].to_string();
        Self { name: name.to_string(), fileset, inner }
    }

    pub fn files(&self) -> &[DataFile] {
        self.inner.files()
    }

    pub fn version(&self) -> String {
        self.inner.version()
    }

    /// Dependency token for results computed from this exact snapshot.
    pub fn fileset_dep(&self) -> String {
        format!("{}@{}", self.name, self.fileset)
    }

    /// Dependency token for results computed from one file of this table.
    pub fn file_dep(&self, f: &DataFile) -> String {
        format!("{}/{}", self.name, f.id)
    }

    /// Every identity a result computed from this snapshot may depend on.
    pub fn deps(&self) -> impl Iterator<Item = String> + '_ {
        self.files().iter().map(|f| self.file_dep(f)).chain([self.fileset_dep()])
    }
}

/// What a scan's provider is, as far as the cache knows.
#[derive(Clone)]
pub enum Binding {
    /// A registered immutable-file table at one snapshot.
    Files(Arc<Snapshot>),
    /// A provider the user promised never changes (its dependency token,
    /// `<name>@static`).
    Static(String),
}

/// How many superseded providers per table stay recognizable, so a query
/// planned just before a refresh is still matched to its own snapshot.
const KEEP_PROVIDERS: usize = 4;

pub(crate) struct Registered {
    pub table: Arc<dyn ImmutableFileTable>,
    pub current: Arc<Snapshot>,
}

/// Maps the providers registered in sessions to what they read.
#[derive(Default)]
pub(crate) struct Registry {
    pub tables: RwLock<HashMap<String, Registered>>,
    /// Provider address → (provider, kept alive so the address stays unique; binding).
    providers: RwLock<HashMap<usize, (Arc<dyn TableProvider>, Binding)>>,
    /// Provider addresses per table name, newest last.
    recent: RwLock<HashMap<String, VecDeque<usize>>>,
}

pub(crate) fn addr(p: &Arc<dyn TableProvider>) -> usize {
    Arc::as_ptr(p) as *const () as usize
}

impl Registry {
    /// Remember that `provider` (registered as `name`) reads `binding`.
    pub fn bind(&self, name: &str, provider: Arc<dyn TableProvider>, binding: Binding) {
        let a = addr(&provider);
        let mut providers = self.providers.write().unwrap();
        let mut recent = self.recent.write().unwrap();
        let list = recent.entry(name.to_string()).or_default();
        providers.insert(a, (provider, binding));
        list.retain(|x| *x != a);
        list.push_back(a);
        while list.len() > KEEP_PROVIDERS {
            if let Some(old) = list.pop_front() {
                providers.remove(&old);
            }
        }
    }

    /// Forget every provider registered under `name`.
    pub fn unbind(&self, name: &str) {
        let mut providers = self.providers.write().unwrap();
        if let Some(list) = self.recent.write().unwrap().remove(name) {
            for a in list {
                providers.remove(&a);
            }
        }
        self.tables.write().unwrap().remove(name);
    }

    /// What a scan reads, or `None` if its provider is not known.
    pub fn resolve(&self, source: &Arc<dyn TableSource>) -> Option<Binding> {
        let provider = source_as_provider(source).ok()?;
        self.providers.read().unwrap().get(&addr(&provider)).map(|(_, b)| b.clone())
    }

    pub fn current(&self, name: &str) -> Option<Arc<Snapshot>> {
        self.tables.read().unwrap().get(name).map(|r| r.current.clone())
    }

    /// Does this dependency token belong to a table registered here?
    pub fn owns(&self, dep: &str) -> bool {
        let tables = self.tables.read().unwrap();
        let recent = self.recent.read().unwrap();
        let name = dep.split(['@', '/']).next().unwrap_or(dep);
        tables.contains_key(name) || recent.contains_key(name)
    }

    /// Identities of the current snapshot of every registered table, and the
    /// tokens of static tables. (Superseded providers still resolve, so a
    /// query planned just before a refresh is keyed correctly, but entries
    /// built only from superseded snapshots are dropped.)
    pub fn live_deps(&self) -> HashSet<String> {
        let mut live: HashSet<String> =
            self.tables.read().unwrap().values().flat_map(|r| r.current.deps()).collect();
        for (_, b) in self.providers.read().unwrap().values() {
            if let Binding::Static(dep) = b {
                live.insert(dep.clone());
            }
        }
        live
    }
}
