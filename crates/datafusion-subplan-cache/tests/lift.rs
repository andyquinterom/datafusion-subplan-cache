//! Predicate lifting: filters decided per file from column statistics.
//!
//! The table `events` has one Parquet file per (hour, region): 12 hours from
//! `BASE`, regions EU and US. Every query is checked against an uncached run.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion_subplan_cache::datafusion::arrow::array::RecordBatch;
use datafusion_subplan_cache::datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion_subplan_cache::datafusion::common::Result;
use datafusion_subplan_cache::datafusion::prelude::SessionContext;
use datafusion_subplan_cache::{Builder, ParquetFiles, RewriteReport, SegmentUse, SubplanCache};

/// 2026-01-01T00:00:00Z.
const BASE: i64 = 1_767_225_600;
const HOURS: i64 = 12;
const REGIONS: [&str; 2] = ["EU", "US"];

fn temp_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("subplan-lift-{name}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One file of 600 rows, timestamps spread over the hour starting at `start`
/// (seconds since the epoch). `null_every`: every n-th row gets a NULL ts.
async fn write_hour(dir: &Path, start: i64, region: &str, null_every: Option<u64>) -> Result<()> {
    let ts = match null_every {
        Some(n) => format!("CASE WHEN value % {n} = 0 THEN NULL ELSE to_timestamp({start} + value * 6) END"),
        None => format!("to_timestamp({start} + value * 6)"),
    };
    let sql = format!(
        "COPY (SELECT {ts} AS ts, '{region}' AS region, value % 10 AS k, value + {start} AS v
               FROM range(0, 600))
         TO '{}/h{start}-{region}.parquet' STORED AS PARQUET",
        dir.display()
    );
    SessionContext::new().sql(&sql).await?.collect().await?;
    Ok(())
}

struct Fixture {
    cache: SubplanCache,
    ctx: SessionContext,
    dir: PathBuf,
}

async fn fixture_at(name: &str, base: i64, builder: Builder) -> Result<Fixture> {
    let dir = temp_dir(name);
    for h in 0..HOURS {
        for r in REGIONS {
            write_hour(&dir, base + h * 3600, r, None).await?;
        }
    }
    let cache = builder.segment_files(1).build();
    let ctx = cache.session_context();
    let table = ParquetFiles::open(&ctx, &format!("file://{}/", dir.display())).await?;
    cache.register_table(&ctx, "events", table).await?;
    Ok(Fixture { cache, ctx, dir })
}

async fn fixture(name: &str) -> Result<Fixture> {
    fixture_at(name, BASE, SubplanCache::builder()).await
}

fn text(b: &[RecordBatch]) -> String {
    pretty_format_batches(b).unwrap().to_string()
}

/// Run through the cache, compare with an uncached run, return what the
/// cache did: the rewrite report and the single segmented aggregate's use.
async fn run(f: &Fixture, sql: &str) -> Result<(RewriteReport, SegmentUse)> {
    f.cache.take_reports();
    f.cache.take_segment_uses();
    let cached = f.ctx.sql(sql).await?.collect().await?;
    let mut report = RewriteReport::default();
    for r in f.cache.take_reports() {
        report.hits.extend(r.hits);
        report.writes.extend(r.writes);
    }
    let mut uses = f.cache.take_segment_uses();
    f.cache.set_enabled(false);
    let reference = f.ctx.sql(sql).await?.collect().await?;
    f.cache.set_enabled(true);
    assert_eq!(text(&cached), text(&reference), "cached result differs for {sql}");
    assert!(cached.iter().map(|b| b.num_rows()).sum::<usize>() > 0, "test query returned nothing: {sql}");
    let u = match uses.len() {
        1 => uses.remove(0),
        0 => SegmentUse { base_key: String::new(), reused: vec![], computed: vec![], lifted: vec![], edge: vec![], skipped: 0 },
        n => panic!("expected one segmented aggregate, got {n}"),
    };
    Ok((report, u))
}

/// `TIMESTAMP '…'` of hour `h` after `BASE`, plus `minutes`.
fn at(h: i64, minutes: i64) -> String {
    let t = BASE + h * 3600 + minutes * 60;
    format!("to_timestamp({t})")
}

fn by_region_window(lo: &str, hi: &str) -> String {
    format!(
        "SELECT region, k, count(*) AS c, sum(v) AS s FROM events
         WHERE ts >= {lo} AND ts < {hi} GROUP BY region, k ORDER BY region, k"
    )
}

#[tokio::test]
async fn sliding_window_reuses_files_inside_it() -> Result<()> {
    let f = fixture("sliding").await?;
    // Hours 0-2: 6 files computed, the other 18 not read.
    let (r, u) = run(&f, &by_region_window(&at(0, 0), &at(3, 0))).await?;
    assert_eq!(u.lifted.len(), 2, "{:?}", u.lifted);
    assert_eq!((u.reused.len(), u.computed.len(), u.edge.len(), u.skipped), (0, 6, 0, 18));
    assert!(r.writes.is_empty(), "no whole-result entry for a lifted window: {:?}", r.writes);

    // Hours 1-3: hours 1-2 come from the cache, only hour 3 is read.
    let (_, u) = run(&f, &by_region_window(&at(1, 0), &at(4, 0))).await?;
    assert_eq!((u.reused.len(), u.computed.len(), u.edge.len(), u.skipped), (4, 2, 0, 18));

    // The same window again: everything from the cache.
    let (_, u) = run(&f, &by_region_window(&at(1, 0), &at(4, 0))).await?;
    assert_eq!((u.reused.len(), u.computed.len()), (6, 0));
    Ok(())
}

#[tokio::test]
async fn bound_inside_a_file_reads_that_file_live() -> Result<()> {
    let f = fixture("edge").await?;
    run(&f, &by_region_window(&at(0, 0), &at(4, 0))).await?;
    // From 1:30 on: both hour-1 files are only partly inside.
    let (_, u) = run(&f, &by_region_window(&at(1, 30), &at(4, 0))).await?;
    assert_eq!(u.edge.len(), 2, "{u:?}");
    assert_eq!((u.reused.len(), u.computed.len(), u.skipped), (4, 0, 18));
    // Edge files are never cached: asking again reads them again.
    let (_, u) = run(&f, &by_region_window(&at(1, 30), &at(4, 0))).await?;
    assert_eq!((u.reused.len(), u.edge.len()), (4, 2));
    Ok(())
}

#[tokio::test]
async fn literal_forms_are_recognized() -> Result<()> {
    let f = fixture("forms").await?;
    // String literals coerced to timestamps, BETWEEN, a literal on the left.
    for q in [
        "SELECT k, count(*) FROM events WHERE ts >= '2026-01-01T02:00:00Z' AND ts < '2026-01-01T05:00:00Z' GROUP BY k ORDER BY k",
        "SELECT k, count(*) FROM events WHERE ts BETWEEN '2026-01-01T02:00:00Z' AND '2026-01-01T04:59:59Z' GROUP BY k ORDER BY k",
        "SELECT k, count(*) FROM events WHERE to_timestamp(1767232800) <= ts AND ts < to_timestamp(1767243600) GROUP BY k ORDER BY k",
    ] {
        let (_, u) = run(&f, q).await?;
        assert!(!u.lifted.is_empty(), "nothing lifted for {q}");
        assert_eq!(u.reused.len() + u.computed.len() + u.edge.len(), 6, "{q}: {u:?}");
        assert_eq!(u.skipped, 18, "{q}");
    }
    Ok(())
}

#[tokio::test]
async fn equality_in_and_not_equal() -> Result<()> {
    let f = fixture("eq").await?;
    let q = |pred: &str| format!("SELECT k, count(*) AS c, sum(v) AS s FROM events WHERE {pred} GROUP BY k ORDER BY k");
    let (_, u) = run(&f, &q("region = 'EU'")).await?;
    assert_eq!((u.computed.len(), u.skipped), (12, 12));
    // Same shape without the predicate: the EU files are reused.
    let (_, u) = run(&f, &q("region IN ('EU', 'US')")).await?;
    assert_eq!((u.reused.len(), u.computed.len(), u.skipped), (12, 12, 0));
    let (_, u) = run(&f, &q("region <> 'US'")).await?;
    assert_eq!((u.reused.len(), u.computed.len(), u.skipped), (12, 0, 12));
    // Long IN lists stay IN lists.
    let (_, u) = run(&f, &q("region IN ('EU', 'AP', 'SA', 'AF', 'OC')")).await?;
    assert_eq!((u.reused.len(), u.skipped), (12, 12));
    Ok(())
}

#[tokio::test]
async fn several_columns_combine() -> Result<()> {
    let f = fixture("combined").await?;
    let q = format!(
        "SELECT k, sum(v) AS s FROM events WHERE region = 'US' AND ts >= {} GROUP BY k ORDER BY k",
        at(9, 0)
    );
    let (_, u) = run(&f, &q).await?;
    assert_eq!(u.lifted.len(), 2, "{:?}", u.lifted);
    assert_eq!((u.computed.len(), u.skipped), (3, 21));
    Ok(())
}

#[tokio::test]
async fn predicate_on_a_scattered_column_stays_in_the_key() -> Result<()> {
    let f = fixture("scattered").await?;
    // k is 0..9 in every file: no file is decided, so nothing is lifted and
    // the query is cached as a whole, as without lifting.
    let q = "SELECT region, count(*) AS c FROM events WHERE k = 3 GROUP BY region ORDER BY region";
    let (r, u) = run(&f, q).await?;
    assert!(u.lifted.is_empty(), "{:?}", u.lifted);
    assert_eq!(r.writes.len(), 1);
    let (r, _) = run(&f, q).await?;
    assert_eq!(r.hits.len(), 1);
    Ok(())
}

#[tokio::test]
async fn other_predicates_stay_in_the_key() -> Result<()> {
    let f = fixture("mixed").await?;
    // `k = 3` stays (scattered), the window is lifted around it.
    let q = |lo: &str| {
        format!("SELECT region, count(*) AS c FROM events WHERE k = 3 AND ts >= {lo} GROUP BY region ORDER BY region")
    };
    let (_, u) = run(&f, &q(&at(6, 0))).await?;
    assert_eq!(u.lifted.len(), 1, "{:?}", u.lifted);
    assert_eq!((u.computed.len(), u.skipped), (12, 12));
    let (_, u) = run(&f, &q(&at(8, 0))).await?;
    assert_eq!((u.reused.len(), u.computed.len(), u.skipped), (8, 0, 16));
    // A different k is a different aggregate: nothing shared.
    let (_, u) = run(&f, &q(&at(8, 0)).replace("k = 3", "k = 4")).await?;
    assert_eq!(u.reused.len(), 0);
    Ok(())
}

#[tokio::test]
async fn null_timestamps_make_a_file_edge() -> Result<()> {
    let f = fixture("nulls").await?;
    // An extra EU file in hour 2 where every 7th ts is NULL.
    write_hour(&f.dir, BASE + 2 * 3600 + 1, "EU", Some(7)).await?;
    f.cache.refresh(&f.ctx).await?;
    let (_, u) = run(&f, &by_region_window(&at(0, 0), &at(12, 0))).await?;
    assert_eq!(u.edge.len(), 1, "{u:?}");
    assert_eq!(u.computed.len(), 24);
    let (_, u) = run(&f, "SELECT region, count(*) AS c FROM events WHERE ts IS NULL GROUP BY region").await?;
    assert_eq!((u.edge.len(), u.skipped), (1, 24));
    Ok(())
}

#[tokio::test]
async fn appended_files_join_the_window() -> Result<()> {
    let f = fixture("append").await?;
    let q = by_region_window(&at(9, 0), &at(24, 0));
    let (_, u) = run(&f, &q).await?;
    assert_eq!((u.computed.len(), u.skipped), (6, 18));
    for r in REGIONS {
        write_hour(&f.dir, BASE + 12 * 3600, r, None).await?;
    }
    f.cache.refresh(&f.ctx).await?;
    let (_, u) = run(&f, &q).await?;
    assert_eq!((u.reused.len(), u.computed.len(), u.skipped), (6, 2, 18));
    Ok(())
}

#[tokio::test]
async fn now_relative_window_reuses_work() -> Result<()> {
    // Files for the 12 hours up to the current one.
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let first = now - now % 3600 - (HOURS - 1) * 3600;
    let f = fixture_at("now", first, SubplanCache::builder()).await?;
    let q = "SELECT region, k, count(*) AS c, sum(v) AS s FROM events
             WHERE ts >= now() - interval '3 hours' GROUP BY region, k ORDER BY region, k";
    let (_, u) = run(&f, q).await?;
    assert_eq!(u.lifted.len(), 1, "now() is folded to a literal and lifted: {:?}", u.lifted);
    assert!(u.computed.len() >= 4, "{u:?}");
    // A later run (the literal differs) reuses every file still inside.
    let (_, u) = run(&f, q).await?;
    assert!(u.reused.len() >= 4, "{u:?}");
    assert!(u.computed.len() <= 2, "{u:?}");
    Ok(())
}

#[tokio::test]
async fn lifting_can_be_turned_off() -> Result<()> {
    let f = fixture_at("off", BASE, SubplanCache::builder().lift_max_edge_fraction(0.0)).await?;
    run(&f, &by_region_window(&at(0, 0), &at(3, 0))).await?;
    let (_, u) = run(&f, &by_region_window(&at(1, 0), &at(4, 0))).await?;
    assert!(u.lifted.is_empty());
    assert_eq!(u.reused.len(), 0, "without lifting a different window shares nothing");
    Ok(())
}

#[tokio::test]
async fn global_aggregate_with_everything_excluded() -> Result<()> {
    let f = fixture("empty").await?;
    // No file overlaps the window: the result is the aggregate of no rows.
    let q = format!("SELECT count(*) AS c, sum(v) AS s FROM events WHERE ts >= {}", at(100, 0));
    let cached = f.ctx.sql(&q).await?.collect().await?;
    f.cache.set_enabled(false);
    let reference = f.ctx.sql(&q).await?.collect().await?;
    assert_eq!(text(&cached), text(&reference));
    Ok(())
}

#[cfg(feature = "delta")]
#[tokio::test]
async fn delta_stats_and_partitions() -> Result<()> {
    use datafusion_subplan_cache::DeltaSource;
    use deltalake::DeltaTable;
    use deltalake::protocol::SaveMode;
    let de = |e: deltalake::DeltaTableError| datafusion_subplan_cache::datafusion::common::DataFusionError::External(Box::new(e));

    let dir = temp_dir("delta");
    let url = url::Url::from_directory_path(&dir).unwrap();
    let plain = SessionContext::new();
    let mut table = DeltaTable::try_from_url(url.clone()).await.map_err(de)?;
    for h in 0..6 {
        for r in REGIONS {
            // Sub-millisecond timestamps: Delta's stats keep milliseconds only.
            let start = (BASE + h * 3600) * 1_000_000;
            let b = plain
                .sql(&format!(
                    "SELECT to_timestamp_micros({start} + value * 6000123) AS ts, '{r}' AS region,
                            value % 10 AS k, value AS v FROM range(0, 590)"
                ))
                .await?
                .collect()
                .await?;
            table = table
                .write(b)
                .with_save_mode(SaveMode::Append)
                .with_partition_columns(["region"])
                .await
                .map_err(de)?;
        }
    }
    let cache = SubplanCache::builder().segment_files(1).build();
    let ctx = cache.session_context();
    cache.register_table(&ctx, "events", DeltaSource::open(url.as_str()).await?).await?;
    let f = Fixture { cache, ctx, dir };

    let (_, u) = run(&f, &by_region_window(&at(1, 0), &at(3, 0))).await?;
    assert_eq!((u.computed.len(), u.skipped), (4, 8), "{u:?}");
    let (_, u) = run(&f, &by_region_window(&at(2, 0), &at(4, 0))).await?;
    assert_eq!((u.reused.len(), u.computed.len()), (2, 2), "{u:?}");
    // Partition values are exact stats.
    let (_, u) = run(&f, "SELECT k, count(*) FROM events WHERE region = 'US' GROUP BY k ORDER BY k").await?;
    assert_eq!(u.skipped, 6, "{u:?}");
    Ok(())
}

#[cfg(feature = "ducklake")]
#[tokio::test]
async fn ducklake_catalog_stats() -> Result<()> {
    use datafusion_ducklake::{DuckLakeCatalog, MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter};
    use datafusion_subplan_cache::DuckLakeSource;
    let dl = |e: datafusion_ducklake::DuckLakeError| datafusion_subplan_cache::datafusion::common::DataFusionError::External(Box::new(e));

    let dir = temp_dir("ducklake");
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let conn = format!("sqlite:{}?mode=rwc", dir.join("catalog.db").display());
    let writer = SqliteMetadataWriter::new_with_init(&conn).await.map_err(dl)?;
    writer.set_data_path(data.to_str().unwrap()).map_err(dl)?;
    let snapshot = writer.create_snapshot().map_err(dl)?;
    writer.get_or_create_schema("main", None, snapshot).map_err(dl)?;
    let provider = SqliteMetadataProvider::new(&conn).await.map_err(dl)?;
    let lake = SessionContext::new();
    DuckLakeCatalog::register(&lake, "lake", DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).map_err(dl)?);
    lake.sql("CREATE TABLE lake.main.events (ts TIMESTAMP, region VARCHAR NOT NULL, k BIGINT NOT NULL, v BIGINT NOT NULL)")
        .await?
        .collect()
        .await?;
    // One INSERT (one file) per hour and region.
    for h in 0..6 {
        for r in REGIONS {
            let start = BASE + h * 3600;
            lake.sql(&format!(
                "INSERT INTO lake.main.events SELECT to_timestamp_micros(({start} + value * 6) * 1000000), '{r}',
                        value % 10, value FROM range(0, 600)"
            ))
            .await?
            .collect()
            .await?;
        }
    }

    let cache = SubplanCache::builder().segment_files(1).build();
    let ctx = cache.session_context();
    cache.register_table(&ctx, "events", DuckLakeSource::open_sqlite(&conn, "main", "events").await?).await?;
    let f = Fixture { cache, ctx, dir };

    let (_, u) = run(&f, &by_region_window(&at(1, 0), &at(3, 0))).await?;
    assert_eq!((u.computed.len(), u.skipped), (4, 8), "{u:?}");
    let (_, u) = run(&f, &by_region_window(&at(2, 0), &at(4, 0))).await?;
    assert_eq!((u.reused.len(), u.computed.len()), (2, 2), "{u:?}");
    // A boundary inside an hour: those two files are read live.
    let (_, u) = run(&f, &by_region_window(&at(2, 30), &at(4, 0))).await?;
    assert_eq!((u.reused.len(), u.edge.len()), (2, 2), "{u:?}");
    let (_, u) = run(&f, "SELECT k, count(*) FROM events WHERE region = 'US' GROUP BY k ORDER BY k").await?;
    assert_eq!(u.skipped, 6, "{u:?}");

    // A delete keeps the file's (now looser) catalog bounds: still exact results.
    lake.sql(&format!("DELETE FROM lake.main.events WHERE v = 10 AND ts < {}", at(1, 0))).await?.collect().await?;
    f.cache.refresh(&f.ctx).await?;
    let (_, u) = run(&f, &by_region_window(&at(0, 0), &at(2, 0))).await?;
    assert_eq!(u.skipped, 8, "{u:?}");
    Ok(())
}
