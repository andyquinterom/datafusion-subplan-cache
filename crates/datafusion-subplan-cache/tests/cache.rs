//! End-to-end behavior through a real SessionContext. Local temp dirs only.

use std::path::PathBuf;
use std::sync::Arc;

use datafusion_subplan_cache::datafusion::arrow::array::RecordBatch;
use datafusion_subplan_cache::datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion_subplan_cache::datafusion::common::Result;
use datafusion_subplan_cache::datafusion::prelude::{ParquetReadOptions, SessionContext};
use datafusion_subplan_cache::{ParquetFiles, PersistentStore, RewriteReport, SubplanCache, UnregisteredTables};

fn temp_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("subplan-cache-test-{name}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write `facts` file `n`: rows k in 0..10, v = n*1000 + i.
async fn write_fact_file(dir: &PathBuf, n: u64) -> Result<()> {
    let sql = format!(
        "COPY (SELECT value % 10 AS k, value AS v FROM range({}, {}))
         TO '{}/part-{n:04}.parquet' STORED AS PARQUET",
        n * 1000,
        (n + 1) * 1000,
        dir.display()
    );
    SessionContext::new().sql(&sql).await?.collect().await?;
    Ok(())
}

async fn write_dim(dir: &PathBuf) -> Result<String> {
    let path = format!("{}/dim.parquet", dir.display());
    let sql = format!("COPY (SELECT value AS k, value % 3 AS grp FROM range(0, 10)) TO '{path}' STORED AS PARQUET");
    SessionContext::new().sql(&sql).await?.collect().await?;
    Ok(path)
}

struct Fixture {
    cache: SubplanCache,
    ctx: SessionContext,
    facts: PathBuf,
}

async fn fixture(name: &str, files: u64, builder: datafusion_subplan_cache::Builder) -> Result<Fixture> {
    let facts = temp_dir(name);
    for n in 0..files {
        write_fact_file(&facts, n).await?;
    }
    let cache = builder.segment_files(1).build();
    let ctx = cache.session_context();
    let table = ParquetFiles::open(&ctx, &format!("file://{}/", facts.display())).await?;
    cache.register_table(&ctx, "facts", table).await?;
    Ok(Fixture { cache, ctx, facts })
}

fn text(b: &[RecordBatch]) -> String {
    pretty_format_batches(b).unwrap().to_string()
}

/// Run through the cache, check against an uncached run, return the report.
async fn run(f: &Fixture, sql: &str) -> Result<RewriteReport> {
    f.cache.take_reports();
    let cached = f.ctx.sql(sql).await?.collect().await?;
    let mut report = RewriteReport::default();
    for r in f.cache.take_reports() {
        report.hits.extend(r.hits);
        report.writes.extend(r.writes);
        report.unregistered.extend(r.unregistered);
    }
    f.cache.set_enabled(false);
    let reference = f.ctx.sql(sql).await?.collect().await?;
    f.cache.set_enabled(true);
    assert_eq!(text(&cached), text(&reference), "cached result differs for {sql}");
    f.cache.flush().await?;
    Ok(report)
}

const AGG: &str = "SELECT k, sum(v) AS s, count(*) AS c FROM facts GROUP BY k ORDER BY k";

#[tokio::test]
async fn repeat_is_a_hit() -> Result<()> {
    let f = fixture("repeat", 3, SubplanCache::builder()).await?;
    let r = run(&f, AGG).await?;
    assert!(r.hits.is_empty());
    assert!(!r.writes.is_empty());
    let r = run(&f, AGG).await?;
    assert_eq!(r.hits.len(), 1, "{r:?}");
    Ok(())
}

#[tokio::test]
async fn append_reuses_segments() -> Result<()> {
    let f = fixture("append", 3, SubplanCache::builder()).await?;
    run(&f, AGG).await?;
    f.cache.take_segment_uses();

    write_fact_file(&f.facts, 3).await?;
    // Not visible before a refresh: the registered snapshot is fixed.
    let r = run(&f, AGG).await?;
    assert_eq!(r.hits.len(), 1);

    assert_eq!(f.cache.refresh(&f.ctx).await?, vec!["facts".to_string()]);
    let r = run(&f, AGG).await?;
    assert!(r.hits.is_empty(), "new snapshot: the whole result must miss");
    let uses = f.cache.take_segment_uses();
    assert_eq!(uses.len(), 1);
    assert_eq!(uses[0].reused.len(), 3);
    assert_eq!(uses[0].computed.len(), 1);
    Ok(())
}

#[tokio::test]
async fn removed_file_is_not_reused() -> Result<()> {
    let f = fixture("remove", 3, SubplanCache::builder()).await?;
    run(&f, AGG).await?;
    f.cache.take_segment_uses();
    std::fs::remove_file(f.facts.join("part-0001.parquet"))?;
    f.cache.refresh(&f.ctx).await?;
    run(&f, AGG).await?;
    let uses = f.cache.take_segment_uses();
    assert_eq!(uses[0].reused.len(), 2);
    assert_eq!(uses[0].computed.len(), 0);
    Ok(())
}

const JOIN: &str = "SELECT a.k, a.s, b.c FROM
      (SELECT k, sum(v) AS s FROM facts GROUP BY k) a
      JOIN (SELECT k, count(*) AS c FROM dim GROUP BY k) b ON a.k = b.k
    ORDER BY a.k";

/// Register `dim` directly on the context, behind the cache's back.
async fn with_unregistered_dim(f: &Fixture) -> Result<()> {
    let dim = write_dim(&temp_dir("dim")).await?;
    f.ctx.register_parquet("dim", &dim, ParquetReadOptions::default()).await
}

#[tokio::test]
async fn unregistered_table_skips_only_its_subtrees() -> Result<()> {
    let f = fixture("skip", 3, SubplanCache::builder()).await?;
    with_unregistered_dim(&f).await?;
    let r = run(&f, JOIN).await?;
    assert_eq!(r.unregistered, vec!["dim".to_string()]);
    // The facts aggregate is cached even though the join is not.
    assert_eq!(r.writes.len(), 1, "{r:?}");
    let r = run(&f, JOIN).await?;
    assert_eq!(r.hits.len(), 1);
    assert!(r.hits[0].label.contains("facts"), "{}", r.hits[0].label);
    Ok(())
}

#[tokio::test]
async fn unregistered_table_error_policy() -> Result<()> {
    let f = fixture("error", 2, SubplanCache::builder().unregistered_tables(UnregisteredTables::Error)).await?;
    with_unregistered_dim(&f).await?;
    let err = f.ctx.sql(JOIN).await?.collect().await.unwrap_err();
    assert!(err.to_string().contains("not registered"), "{err}");
    // Plain scans of an unregistered table are never cached, so they pass.
    f.ctx.sql("SELECT * FROM dim").await?.collect().await?;
    Ok(())
}

#[tokio::test]
async fn unregistered_table_assume_static() -> Result<()> {
    let f = fixture("static", 2, SubplanCache::builder().unregistered_tables(UnregisteredTables::AssumeStatic)).await?;
    with_unregistered_dim(&f).await?;
    run(&f, JOIN).await?;
    let r = run(&f, JOIN).await?;
    assert_eq!(r.hits.len(), 1);
    assert!(r.hits[0].label.starts_with("Inner Join") || r.hits[0].label.contains("Join"), "{}", r.hits[0].label);
    Ok(())
}

#[tokio::test]
async fn static_table_join_is_cached_and_segmented() -> Result<()> {
    let f = fixture("registered-static", 3, SubplanCache::builder()).await?;
    with_unregistered_dim(&f).await?;
    f.cache.mark_static(&f.ctx, "dim").await?;
    let q = "SELECT d.grp, sum(f.v) AS s FROM facts f JOIN dim d ON f.k = d.k GROUP BY d.grp ORDER BY d.grp";
    run(&f, q).await?;
    let uses = f.cache.take_segment_uses();
    assert_eq!(uses.len(), 1, "aggregate over facts ⋈ static dim is segmented");
    assert_eq!(uses[0].computed.len(), 3);
    write_fact_file(&f.facts, 3).await?;
    f.cache.refresh(&f.ctx).await?;
    run(&f, q).await?;
    let uses = f.cache.take_segment_uses();
    assert_eq!((uses[0].reused.len(), uses[0].computed.len()), (3, 1));
    Ok(())
}

#[tokio::test]
async fn two_caches_share_a_store() -> Result<()> {
    let store_dir = temp_dir("store");
    let store = Arc::new(PersistentStore::open(store_dir.to_str().unwrap())?);
    let a = fixture("shared-a", 3, SubplanCache::builder().with_store(store.clone())).await?;
    run(&a, AGG).await?;

    // A second "process": new cache, new context, same files and store.
    let cache = SubplanCache::builder().segment_files(1).with_store(store).build();
    let ctx = cache.session_context();
    cache
        .register_table(&ctx, "facts", ParquetFiles::open(&ctx, &format!("file://{}/", a.facts.display())).await?)
        .await?;
    let b = Fixture { cache, ctx, facts: a.facts.clone() };
    let r = run(&b, AGG).await?;
    assert_eq!(r.hits.len(), 1, "{r:?}");
    assert_eq!(b.cache.loads(), 1);
    Ok(())
}

#[tokio::test]
async fn purge_deletes_only_superseded_entries() -> Result<()> {
    let store_dir = temp_dir("purge");
    let f = fixture("purge", 2, SubplanCache::builder().store(store_dir.to_str().unwrap())?).await?;
    run(&f, AGG).await?; // 2 segments + 1 whole result
    std::fs::remove_file(f.facts.join("part-0000.parquet"))?;
    f.cache.refresh(&f.ctx).await?;
    // Within the grace period nothing goes.
    assert_eq!(f.cache.purge(std::time::Duration::from_secs(3600)).await?, 0);
    // The whole result and the segment of the removed file are superseded.
    assert_eq!(f.cache.purge(std::time::Duration::ZERO).await?, 2);
    run(&f, AGG).await?;
    Ok(())
}

#[cfg(feature = "delta")]
#[tokio::test]
async fn delta_delete_recomputes_only_the_rewritten_file() -> Result<()> {
    use datafusion_subplan_cache::DeltaSource;
    use deltalake::DeltaTable;
    use deltalake::protocol::SaveMode;
    let de = |e: deltalake::DeltaTableError| datafusion_subplan_cache::datafusion::common::DataFusionError::External(Box::new(e));

    let dir = temp_dir("delta");
    let url = url::Url::from_directory_path(&dir).unwrap();
    let plain = SessionContext::new();
    let mut table = DeltaTable::try_from_url(url.clone()).await.map_err(de)?;
    for n in 0..3 {
        let b = plain
            .sql(&format!("SELECT value % 10 AS k, value AS v FROM range({}, {})", n * 1000, (n + 1) * 1000))
            .await?
            .collect()
            .await?;
        table = table.write(b).with_save_mode(SaveMode::Append).await.map_err(de)?;
    }
    let cache = SubplanCache::builder().segment_files(1).build();
    let ctx = cache.session_context();
    cache.register_table(&ctx, "facts", DeltaSource::open(url.as_str()).await?).await?;
    let f = Fixture { cache, ctx, facts: dir };
    run(&f, AGG).await?;
    f.cache.take_segment_uses();

    table.delete().with_predicate("v = 1500").await.map_err(de)?;
    f.cache.refresh(&f.ctx).await?;
    run(&f, AGG).await?;
    let uses = f.cache.take_segment_uses();
    assert_eq!((uses[0].reused.len(), uses[0].computed.len()), (2, 1));
    Ok(())
}

/// Delta partition columns come back dictionary-encoded, one dictionary per
/// file: a cached result over several files must still persist and load.
#[cfg(feature = "delta")]
#[tokio::test]
async fn delta_partitioned_entries_persist() -> Result<()> {
    use datafusion_subplan_cache::DeltaSource;
    use deltalake::DeltaTable;
    use deltalake::protocol::SaveMode;
    let de = |e: deltalake::DeltaTableError| datafusion_subplan_cache::datafusion::common::DataFusionError::External(Box::new(e));

    let dir = temp_dir("delta-part");
    let url = url::Url::from_directory_path(&dir).unwrap();
    let plain = SessionContext::new();
    let mut table = DeltaTable::try_from_url(url.clone()).await.map_err(de)?;
    for n in 0..3 {
        let sql = format!("SELECT 'p{n}' AS part, value % 10 AS k, value AS v FROM range({}, {})", n * 1000, (n + 1) * 1000);
        let b = plain.sql(&sql).await?.collect().await?;
        table = table.write(b).with_save_mode(SaveMode::Append).with_partition_columns(["part"]).await.map_err(de)?;
    }
    const BY_PART: &str = "SELECT part, k, sum(v) AS s FROM facts GROUP BY part, k ORDER BY part, k";
    let store = Arc::new(PersistentStore::open(temp_dir("delta-part-store").to_str().unwrap())?);
    let mut hits = vec![];
    for _ in 0..2 {
        // Two "processes" sharing one store.
        let cache = SubplanCache::builder().segment_files(1).with_store(store.clone()).build();
        let ctx = cache.session_context();
        cache.register_table(&ctx, "facts", DeltaSource::open(url.as_str()).await?).await?;
        let f = Fixture { cache, ctx, facts: dir.clone() };
        hits.push(run(&f, BY_PART).await?.hits.len());
        assert_eq!(f.cache.loads(), hits.len() - 1);
    }
    assert_eq!(hits, [0, 1]);
    Ok(())
}

/// A SQLite-backed DuckLake in a temp dir with table `main.facts` written in
/// `files` INSERTs of 1000 rows each (one Parquet file each). Returns the
/// writer session (catalog `lake`) and the catalog connection string.
#[cfg(feature = "ducklake")]
async fn ducklake_lake(name: &str, files: u64, inline_limit: usize) -> Result<(SessionContext, String)> {
    use datafusion_ducklake::{
        DuckLakeCatalog, DuckLakeWriteOptions, MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter,
    };
    let dl = |e: datafusion_ducklake::DuckLakeError| datafusion_subplan_cache::datafusion::common::DataFusionError::External(Box::new(e));
    let dir = temp_dir(name);
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let conn = format!("sqlite:{}?mode=rwc", dir.join("catalog.db").display());
    let writer = SqliteMetadataWriter::new_with_init(&conn).await.map_err(dl)?;
    writer.set_data_path(data.to_str().unwrap()).map_err(dl)?;
    let snapshot = writer.create_snapshot().map_err(dl)?;
    writer.get_or_create_schema("main", None, snapshot).map_err(dl)?;
    let provider = SqliteMetadataProvider::new(&conn).await.map_err(dl)?;
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer))
        .map_err(dl)?
        .with_write_options(DuckLakeWriteOptions::default().with_data_inlining_row_limit(inline_limit));
    let ctx = SessionContext::new();
    DuckLakeCatalog::register(&ctx, "lake", catalog);
    let rows = |n: u64| format!("SELECT value % 10 AS k, value AS v FROM range({}, {})", n * 1000, (n + 1) * 1000);
    ctx.sql("CREATE TABLE lake.main.facts (k BIGINT NOT NULL, v BIGINT NOT NULL)").await?.collect().await?;
    for n in 0..files {
        ctx.sql(&format!("INSERT INTO lake.main.facts {}", rows(n))).await?.collect().await?;
    }
    Ok((ctx, conn))
}

#[cfg(feature = "ducklake")]
async fn ducklake_fixture(name: &str, files: u64, inline_limit: usize) -> Result<(Fixture, SessionContext)> {
    use datafusion_subplan_cache::DuckLakeSource;
    let (lake, conn) = ducklake_lake(name, files, inline_limit).await?;
    let cache = SubplanCache::builder().segment_files(1).build();
    let ctx = cache.session_context();
    cache.register_table(&ctx, "facts", DuckLakeSource::open_sqlite(&conn, "main", "facts").await?).await?;
    Ok((Fixture { cache, ctx, facts: temp_dir(name) }, lake))
}

#[cfg(feature = "ducklake")]
async fn ducklake_expected(lake: &SessionContext) -> Result<String> {
    Ok(text(&lake.sql(&AGG.replace("FROM facts", "FROM lake.main.facts")).await?.collect().await?))
}

#[cfg(feature = "ducklake")]
#[tokio::test]
async fn ducklake_append_reuses_existing_files() -> Result<()> {
    let (f, lake) = ducklake_fixture("ducklake-append", 3, 0).await?;
    run(&f, AGG).await?;
    let uses = f.cache.take_segment_uses();
    assert_eq!((uses[0].reused.len(), uses[0].computed.len()), (0, 3));

    lake.sql("INSERT INTO lake.main.facts SELECT value % 10 AS k, value AS v FROM range(3000, 4000)").await?.collect().await?;
    f.cache.refresh(&f.ctx).await?;
    let got = text(&f.ctx.sql(AGG).await?.collect().await?);
    let uses = f.cache.take_segment_uses();
    assert_eq!((uses[0].reused.len(), uses[0].computed.len()), (3, 1));
    assert_eq!(got, ducklake_expected(&lake).await?);
    // Walks the catalog's snapshot history (timestamps parsed from the catalog).
    f.cache.purge(std::time::Duration::from_secs(3600)).await?;
    Ok(())
}

#[cfg(feature = "ducklake")]
#[tokio::test]
async fn ducklake_delete_recomputes_only_that_file() -> Result<()> {
    let (f, lake) = ducklake_fixture("ducklake-delete", 3, 0).await?;
    run(&f, AGG).await?;
    f.cache.take_segment_uses();

    lake.sql("DELETE FROM lake.main.facts WHERE v = 1500").await?.collect().await?;
    f.cache.refresh(&f.ctx).await?;
    let got = text(&f.ctx.sql(AGG).await?.collect().await?);
    let uses = f.cache.take_segment_uses();
    assert_eq!((uses[0].reused.len(), uses[0].computed.len()), (2, 1));
    assert_eq!(got, ducklake_expected(&lake).await?);
    Ok(())
}

#[cfg(feature = "ducklake")]
#[tokio::test]
async fn ducklake_inlined_rows_are_counted_once() -> Result<()> {
    // Inserts of up to 100 rows stay in the catalog; the 1000-row ones are files.
    let (f, lake) = ducklake_fixture("ducklake-inlined", 2, 100).await?;
    run(&f, AGG).await?;
    f.cache.take_segment_uses();

    lake.sql("INSERT INTO lake.main.facts SELECT value % 10 AS k, value AS v FROM range(5000, 5010)").await?.collect().await?;
    f.cache.refresh(&f.ctx).await?;
    let got = text(&f.ctx.sql(AGG).await?.collect().await?);
    let uses = f.cache.take_segment_uses();
    assert_eq!((uses[0].reused.len(), uses[0].computed.len()), (2, 1));
    assert!(uses[0].computed[0][0].contains("/__inlined__:"), "{:?}", uses[0].computed);
    assert_eq!(got, ducklake_expected(&lake).await?);
    Ok(())
}
