//! Transparent subplan caching for DataFusion over immutable files.
//!
//! Install the cache into a session, register the tables whose data is a
//! set of immutable, uniquely named Parquet files (Delta, Iceberg, DuckLake,
//! an append-only folder, …), and query as usual:
//!
//! ```no_run
//! # async fn demo() -> datafusion::common::Result<()> {
//! use datafusion::execution::SessionStateBuilder;
//! use datafusion::prelude::SessionContext;
//! use datafusion_subplan_cache::{ParquetFiles, SubplanCache};
//!
//! let cache = SubplanCache::builder().store("/tmp/plan-cache")?.build();
//! let state = cache.install(SessionStateBuilder::new().with_default_features()).build();
//! let ctx = SessionContext::new_with_state(state);
//!
//! let events = ParquetFiles::open(&ctx, "file:///data/events/").await?;
//! cache.register_table(&ctx, "events", events).await?;
//!
//! // Cached: repeats are answered from the cache, and after new files show
//! // up (and `refresh`) only the new files are aggregated.
//! ctx.sql("SELECT kind, count(*) FROM events GROUP BY kind").await?.collect().await?;
//! cache.refresh(&ctx).await?;
//! # Ok(()) }
//! ```
//!
//! What gets cached:
//! * **whole results** of aggregates, joins, windows and DISTINCTs, keyed by
//!   their optimized plan text plus the identity of every snapshot they read;
//! * **partial aggregate states per group of files** for an aggregate over one
//!   file table (optionally joined with static tables). A later snapshot
//!   reuses the states of files it still contains and computes only the
//!   new ones.
//!
//! Entries live in memory and, with [`Builder::store`], in a directory or
//! bucket shared by any number of processes.

pub mod adapters;
mod cache;
mod exec;
pub mod rewrite;
pub mod store;
pub mod table;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime};

pub use datafusion;
use datafusion::catalog::TableProvider;
use datafusion::common::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::{SessionConfig, SessionContext};

pub use adapters::parquet::ParquetFiles;
#[cfg(feature = "delta")]
pub use adapters::delta::DeltaSource;
pub use cache::{CacheEntry, EntryData, SegmentUse};
pub use exec::{CacheExtensionPlanner, CachingQueryPlanner};
pub use rewrite::RewriteReport;
pub use store::PersistentStore;
pub use table::{DataFile, FileSnapshot, ImmutableFileTable, Snapshot};

use cache::Cache;
use table::{Binding, Registered};

/// What to do with a query that reads a table the cache does not know
/// (registered on the context directly, through a catalog, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnregisteredTables {
    /// Run the parts of the plan that read it live, uncached; cache the
    /// rest. An event says which table kept a subtree from being cached.
    #[default]
    Skip,
    /// Fail planning when an aggregate, join, window or DISTINCT reads it.
    /// For services that must not silently lose caching.
    Error,
    /// Treat it as static: cache by plan text alone. Serves stale results
    /// if its data changes. Opt in only for tables that truly never change.
    AssumeStatic,
}

/// Configures a [`SubplanCache`].
pub struct Builder {
    store: Option<Arc<PersistentStore>>,
    segment_files: usize,
    policy: UnregisteredTables,
    max_memory: usize,
}

impl Builder {
    /// Also keep entries in a persistent store: a local directory, or with
    /// the `gcs` / `s3` features `gs://bucket/prefix` / `s3://bucket/prefix`.
    /// Any number of processes may share one store.
    pub fn store(mut self, location: &str) -> Result<Self> {
        self.store = Some(Arc::new(PersistentStore::open(location)?));
        Ok(self)
    }

    /// Use an already opened store.
    pub fn with_store(mut self, store: Arc<PersistentStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// How many data files go into one newly computed segment (default 4).
    /// Smaller segments mean less recomputation when one file changes, and
    /// more (smaller) cache entries.
    pub fn segment_files(mut self, n: usize) -> Self {
        self.segment_files = n.max(1);
        self
    }

    /// See [`UnregisteredTables`] (default `Skip`).
    pub fn unregistered_tables(mut self, policy: UnregisteredTables) -> Self {
        self.policy = policy;
        self
    }

    /// Memory budget for cached batches (default 2 GiB). With a store,
    /// entries beyond it stay in the store only.
    pub fn max_memory(mut self, bytes: usize) -> Self {
        self.max_memory = bytes;
        self
    }

    pub fn build(self) -> SubplanCache {
        SubplanCache {
            inner: Arc::new(Cache::new(self.store, self.segment_files, self.policy, self.max_memory)),
        }
    }
}

/// A subplan cache. Cheap to clone; clones share everything.
#[derive(Clone, Debug)]
pub struct SubplanCache {
    pub(crate) inner: Arc<Cache>,
}

impl Default for SubplanCache {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl SubplanCache {
    pub fn builder() -> Builder {
        Builder {
            store: None,
            segment_files: 4,
            policy: UnregisteredTables::default(),
            max_memory: cache::DEFAULT_MAX_MEMORY,
        }
    }

    /// Make sessions built by `builder` use this cache: installs the query
    /// planner and keeps partial aggregates from skipping aggregation
    /// (which would make cached partial states as large as their input).
    pub fn install(&self, mut builder: SessionStateBuilder) -> SessionStateBuilder {
        let mut config = builder.config().take().unwrap_or_else(SessionConfig::new);
        config.options_mut().execution.skip_partial_aggregation_probe_rows_threshold = usize::MAX;
        builder
            .with_config(config)
            .with_query_planner(Arc::new(CachingQueryPlanner { cache: self.inner.clone() }))
    }

    /// A new `SessionContext` with default features and this cache.
    pub fn session_context(&self) -> SessionContext {
        SessionContext::new_with_state(self.install(SessionStateBuilder::new().with_default_features()).build())
    }

    /// Register a table whose data files are immutable, under `name`, in
    /// `ctx` (replacing any table of that name). Returns its snapshot.
    pub async fn register_table(
        &self,
        ctx: &SessionContext,
        name: &str,
        table: impl ImmutableFileTable + 'static,
    ) -> Result<Arc<Snapshot>> {
        let table: Arc<dyn ImmutableFileTable> = Arc::new(table);
        table.attach(ctx)?;
        let snap = Arc::new(Snapshot::new(name, table.latest().await?));
        self.inner.registry.unbind(name);
        self.install_snapshot(ctx, name, snap.clone()).await?;
        self.inner.registry.tables.write().unwrap().insert(name.to_string(), Registered { table, current: snap.clone() });
        self.inner.rescan();
        Ok(snap)
    }

    async fn install_snapshot(&self, ctx: &SessionContext, name: &str, snap: Arc<Snapshot>) -> Result<()> {
        let provider = snap.inner.provider().await?;
        ctx.deregister_table(name)?;
        ctx.register_table(name, provider.clone())?;
        self.inner.registry.bind(name, provider, Binding::Files(snap));
        Ok(())
    }

    /// Register a table whose contents never change (a dimension table, a
    /// lookup file). Queries joining it stay cacheable. If its contents do
    /// change, register it under a new name or call [`Self::clear`].
    pub fn register_static(&self, ctx: &SessionContext, name: &str, provider: Arc<dyn TableProvider>) -> Result<()> {
        self.inner.registry.unbind(name);
        ctx.deregister_table(name)?;
        ctx.register_table(name, provider.clone())?;
        self.inner.registry.bind(name, provider, Binding::Static(format!("{name}@static")));
        self.inner.rescan();
        Ok(())
    }

    /// Register a static table from the provider already registered in
    /// `ctx` under `name` (e.g. after `ctx.register_parquet`).
    pub async fn mark_static(&self, ctx: &SessionContext, name: &str) -> Result<()> {
        let provider = ctx.table_provider(name).await?;
        self.register_static(ctx, name, provider)
    }

    /// Forget a table and remove it from `ctx`.
    pub fn deregister(&self, ctx: &SessionContext, name: &str) -> Result<()> {
        self.inner.registry.unbind(name);
        ctx.deregister_table(name)?;
        Ok(())
    }

    /// Move every registered table to its latest snapshot (new commits by
    /// this or any other writer). Returns the names of the tables that
    /// changed. Queries see a table's new snapshot only after this.
    pub async fn refresh(&self, ctx: &SessionContext) -> Result<Vec<String>> {
        let tables: Vec<(String, Arc<dyn ImmutableFileTable>, String)> = {
            let t = self.inner.registry.tables.read().unwrap();
            t.iter().map(|(n, r)| (n.clone(), r.table.clone(), r.current.fileset.clone())).collect()
        };
        let mut changed = vec![];
        for (name, table, fileset) in tables {
            let snap = Arc::new(Snapshot::new(&name, table.latest().await?));
            if snap.fileset == fileset {
                continue;
            }
            self.install_snapshot(ctx, &name, snap.clone()).await?;
            if let Some(r) = self.inner.registry.tables.write().unwrap().get_mut(&name) {
                r.current = snap.clone();
            }
            self.inner.event(format!("  ↻ {name} changed: now v{}", snap.version()));
            changed.push(name);
        }
        Ok(changed)
    }

    /// The current snapshot of a registered table.
    pub fn snapshot(&self, name: &str) -> Option<Arc<Snapshot>> {
        self.inner.registry.current(name)
    }

    /// Wait until every entry computed so far is written to the store.
    /// Writes happen in the background; call this before exiting.
    pub async fn flush(&self) -> Result<()> {
        self.inner.flush().await
    }

    /// Delete persisted entries no reader can still use: those depending on
    /// files that left the registered tables more than `grace` ago. Readers
    /// in other processes may lag up to `grace` behind; one that loses an
    /// entry anyway just recomputes it. Entries depending on tables not
    /// registered here are left alone. Returns the number deleted.
    pub async fn purge(&self, grace: Duration) -> Result<usize> {
        let since = SystemTime::now().checked_sub(grace).unwrap_or(SystemTime::UNIX_EPOCH);
        let mut live: HashSet<String> = self.inner.registry.live_deps();
        let tables: Vec<(String, Arc<dyn ImmutableFileTable>)> = {
            let t = self.inner.registry.tables.read().unwrap();
            t.iter().map(|(n, r)| (n.clone(), r.table.clone())).collect()
        };
        for (name, table) in tables {
            for s in table.snapshots_since(since).await? {
                live.extend(Snapshot::new(&name, s).deps());
            }
        }
        let registry = &self.inner.registry;
        self.inner.purge(&live, |d| registry.owns(d), grace).await
    }

    /// Rewrite an optimized logical plan to use the cache, for callers with
    /// their own [`datafusion::execution::context::QueryPlanner`]: plan the
    /// result with a `DefaultPhysicalPlanner` that has a
    /// [`CacheExtensionPlanner`].
    pub async fn rewrite(&self, optimized: &LogicalPlan) -> Result<LogicalPlan> {
        exec::rewrite_with_cache(&self.inner, optimized).await
    }

    // --- inspection and control -------------------------------------------

    /// Turn caching on or off for sessions using this cache.
    pub fn set_enabled(&self, on: bool) {
        self.inner.enabled.store(on, Ordering::Relaxed);
    }

    pub fn enabled(&self) -> bool {
        self.inner.enabled.load(Ordering::Relaxed)
    }

    pub fn set_segment_files(&self, n: usize) {
        self.inner.segment_files.store(n.max(1), Ordering::Relaxed);
    }

    pub fn segment_files(&self) -> usize {
        self.inner.segment_files.load(Ordering::Relaxed)
    }

    /// All entries known in memory (with or without their batches loaded).
    pub fn entries(&self) -> Vec<Arc<CacheEntry>> {
        self.inner.entries()
    }

    pub fn get(&self, key: &str) -> Option<Arc<CacheEntry>> {
        self.inner.get(key)
    }

    pub fn total_bytes(&self) -> usize {
        self.inner.total_bytes()
    }

    pub fn store(&self) -> Option<&Arc<PersistentStore>> {
        self.inner.store.as_ref()
    }

    /// Number of entries whose batches were read from the store so far.
    pub fn loads(&self) -> usize {
        self.inner.loads.load(Ordering::Relaxed)
    }

    /// Human-readable events since the last call (refreshes, store reads,
    /// segment reuse, subtrees not cached and why).
    pub fn take_events(&self) -> Vec<String> {
        self.inner.take_events()
    }

    /// How each planned query used the cache, since the last call.
    pub fn take_reports(&self) -> Vec<RewriteReport> {
        self.inner.take_reports()
    }

    /// Per-segment reuse of each planned aggregate, since the last call.
    pub fn take_segment_uses(&self) -> Vec<SegmentUse> {
        self.inner.take_segment_uses()
    }

    /// Forget everything in memory. Persisted entries are found again on
    /// the next query.
    pub fn clear(&self) {
        self.inner.clear()
    }
}
