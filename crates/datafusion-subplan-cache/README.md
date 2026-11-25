# datafusion-subplan-cache

Transparent result caching for [DataFusion](https://datafusion.apache.org)
over tables made of **immutable, uniquely named Parquet files**: Delta Lake,
Iceberg, DuckLake, Hudi copy-on-write, or a folder that only ever gets new
files.

- **Whole results** of aggregates, joins, windows and DISTINCTs are reused
  when the same subtree runs again on the same data, even inside a different
  query (other ORDER BY / HAVING / LIMIT on top).
- **Per-file partial aggregates**: after an append, delete or compaction,
  an aggregate reads only the files that changed and merges them with the
  cached partial states of the files that did not.
- **Filters decided per file**: simple filter conjuncts (`ts >= now() -
  interval '1 hour'`, `region = 'EU'`, `day IN (…)`, `x IS NULL`) are taken
  out of the cache key and decided per file from its column statistics.
  Files a filter keeps whole reuse their cached partial states, files it
  excludes are not read, and only the files its bounds cut through are read
  live. A sliding time window recomputes just the newest files.
- **Shared persistent store** (optional): a local directory, `gs://` or `s3://`
  prefix. Any number of processes read and write it concurrently, with no
  locks: manifests are created with conditional writes, so the first writer
  wins and the others adopt its result.

## Quick start

```rust
use datafusion_subplan_cache::{ParquetFiles, SubplanCache};

let cache = SubplanCache::builder()
    .store("gs://my-bucket/plan-cache")?   // optional; needs the `gcs` feature
    .build();
let ctx = cache.session_context();          // or: cache.install(SessionStateBuilder) → your own state

// Tables the cache can follow file by file:
cache.register_table(&ctx, "events", ParquetFiles::open(&ctx, "file:///data/events/").await?).await?;
cache.register_table(&ctx, "sales", DeltaSource::open("gs://bucket/sales/").await?).await?;   // `delta` feature
cache.register_table(&ctx, "orders", DuckLakeSource::open_sqlite("sqlite:/data/lake.db", "main", "orders").await?).await?;   // `ducklake` feature

// Tables that never change (dimension tables, lookups):
ctx.register_parquet("customers", "/data/customers.parquet", Default::default()).await?;
cache.mark_static(&ctx, "customers").await?;

ctx.sql("SELECT …").await?.collect().await?;  // cached transparently

cache.refresh(&ctx).await?;   // follow new commits / files (before queries that must see them)
cache.flush().await?;         // wait for background store writes (before exiting)
cache.purge(Duration::from_secs(3600)).await?;   // maintenance: delete entries no recent snapshot uses
```

Runnable: `cargo run -p datafusion-subplan-cache --example quickstart_parquet`
, `--features delta --example quickstart_delta` and
`--features ducklake --example quickstart_ducklake`.

Features: `delta` (Delta adapter via delta-rs), `ducklake` (DuckLake adapter
via datafusion-ducklake, SQLite catalogs), `ducklake-postgres` (adds
PostgreSQL catalogs), `gcs`, `s3` (cloud stores for the cache store and
`ParquetFiles`).

`DuckLakeSource::new` takes any `datafusion_ducklake::MetadataProvider`, so
other catalog backends (MySQL, DuckDB) work by enabling that backend's feature
on your own `datafusion-ducklake` dependency. As with any DuckLake catalog,
register the object store of the lake's data path on the session
(`ctx.register_object_store`) unless it is local.

## The contract: immutable files

A cached result is keyed by its optimized plan text plus the identity of the
data it read. For a registered table that identity is the set of its data
files, each with an id that **must change whenever the rows that file
contributes can change**. Formats that never rewrite a file in place give
this for free: a new file gets a new name.

What breaks the contract (and so must be reflected in the id):

- row-level deletes stored separately (Delta deletion vectors, Iceberg
  position/equality delete files, DuckLake delete files): include the delete
  files that apply to a data file in its id. The Delta adapter does this for
  deletion vectors, the DuckLake adapter for delete files and inlined
  deletions (DuckLake rows inlined in the catalog count as one more file);
- overwriting a file under the same name: include its etag/mtime in the id
  (`ParquetFiles` does).

## Which tables are cached

Scans are matched by **provider**, not by name. A subtree is cached only if
every table it reads was registered with the cache (`register_table`,
`register_static`/`mark_static`). For anything else, the
`unregistered_tables` policy decides:

| Policy | Behavior |
|---|---|
| `Skip` (default) | subtrees reading it run live; everything else is still cached; an event says why |
| `Error` | planning an aggregate/join/window/DISTINCT over it fails |
| `AssumeStatic` | cached by plan text alone; stale if its data changes |

A table re-registered behind the cache's back (a plain
`ctx.register_table`) is unknown again, never matched to an old snapshot.

## Filters decided per file (`now()` windows and friends)

DataFusion folds `now()` into a literal, so `WHERE ts >= now() - interval
'1 hour'` has a new plan text on every run. For an aggregate over one file
table, the cache lifts such conjuncts out of the key and classifies each
file by its per-column min / max / null count:

| | the file's rows | the file is |
|---|---|---|
| inside | all pass every lifted conjunct | answered from its cached *unfiltered* partial state |
| outside | none pass some lifted conjunct | not read |
| edge | some pass, or no statistics | read live with the full predicate, not cached |

Liftable conjuncts are `col op literal` (`= != < <= > >=`, either side,
literal casts allowed), `BETWEEN`, `IN (…)`, `c = a OR c = b …`, and
`IS [NOT] NULL`, on a plain column of the table that is not a float. Other
conjuncts stay in the key as before.

No configuration is needed: `ParquetFiles` reads each new file's footer
once (`with_column_stats(false)` turns that off), and `DeltaSource` uses the
log's per-file stats and partition values. `DuckLakeSource` uses the
catalog's `ducklake_file_column_stats`, so it reads no files either. For your own adapter, attach
exact stats with `DataFile::with_stats`. A conjunct is lifted only if it
leaves at most `Builder::lift_max_edge_fraction` (default 0.25, at least two
files) of the files as edge files. So predicates on columns the data is not
laid out by, like `amount > 100` with every file spanning all amounts, stay
in the key, and repeating such a query is still a whole-result hit.
Lifted aggregates store no whole result. A repeat merges the cached
per-file states and rereads the edge files.

New segments are ordered by the first lifted column, so a sliding window
invalidates few of them. Use a small `segment_files` for time series.

## Writing an adapter (Iceberg, DuckLake, …)

Implement two traits. Sketch for an Iceberg table with merge-on-read deletes:

```rust
struct IcebergTable { catalog: MyCatalog, ident: TableIdent }
struct IcebergSnap { id: i64, files: Vec<DataFile>, scan: MyScanBuilder }

#[async_trait]
impl ImmutableFileTable for IcebergTable {
    async fn latest(&self) -> Result<Arc<dyn FileSnapshot>> {
        let snap = self.catalog.load_table(&self.ident).await?.current_snapshot();
        let files = snap.data_files().map(|f| {
            // Deletes that apply to this file are part of its identity.
            let deletes: Vec<_> = snap.delete_files_for(f).map(|d| d.path()).collect();
            DataFile::new(f.path(), &format!("{}|{}", f.size(), deletes.join(",")))
        }).collect();
        Ok(Arc::new(IcebergSnap { id: snap.id(), files, scan: snap.scan_builder() }))
    }
    // Optional, for purge with lagging readers: snapshots committed since `t`.
    async fn snapshots_since(&self, t: SystemTime) -> Result<Vec<Arc<dyn FileSnapshot>>> { … }
}

#[async_trait]
impl FileSnapshot for IcebergSnap {
    fn version(&self) -> String { self.id.to_string() }
    fn files(&self) -> &[DataFile] { &self.files }
    async fn provider(&self) -> Result<Arc<dyn TableProvider>> { self.scan.all_files().provider() }
    // A scan of just these files, deletes applied. Return Ok(None) if you
    // cannot restrict a scan: then only whole results are cached.
    async fn provider_for(&self, files: &[&DataFile]) -> Result<Option<Arc<dyn TableProvider>>> {
        Ok(Some(self.scan.only_files(files.iter().map(|f| &f.path)).provider()))
    }
}
```

Without delete files, no adapter is needed: pass the snapshot's file list to
`ParquetFiles::from_files(&ctx, "gs://bucket/", files)`, and call
`set_files` + `cache.refresh` when the snapshot changes.

## Custom query planners

`install` sets the session's `QueryPlanner`. If you already have one, call
`cache.rewrite(&optimized_plan)` in it and plan the result with a
`DefaultPhysicalPlanner` that includes `CacheExtensionPlanner::new(&cache)`.

## Limits

- Cache keys include DataFusion's plan text and partial-state layout; the
  store path includes the DataFusion version, so upgrading starts a fresh
  cache.
- Volatile expressions (`random()`) and correlated subqueries are never
  cached. `now()` is folded to a literal and is handled by per-file filters
  (above) where its column has statistics; otherwise it makes a new key on
  every run.
- Per-file reuse applies to aggregates over one registered file table,
  optionally inner/semi-joined with static tables.
