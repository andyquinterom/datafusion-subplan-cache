//! Benchmarks: vanilla DataFusion vs. DataFusion + subplan cache, on Delta
//! and DuckLake tables on local disk.
//!
//!     cargo run --release -p datafusion-subplan-cache --features delta,ducklake --example bench
//!
//! Environment: `BENCH_DIR` (data, default `./data/bench`), `BENCH_OUT`
//! (results JSON, default `crates/datafusion-subplan-cache/docs/bench/results.json`),
//! `BENCH_HOURS` (history, default 72), `BENCH_ROWS` (rows per region and
//! hour, default 200000), `BENCH_FORMATS` (default `delta,ducklake`).
//!
//! The table is request telemetry: `ts, region, device_id, latency_ms,
//! status`, one commit per hour holding one file per region (3 regions).
//! Every scenario starts from a fresh, in-memory cache. "Vanilla" runs the
//! same SQL on a plain `SessionContext` over the same snapshot (median of 3
//! warm runs); "cached" is the single first run after each change, i.e.
//! what a user actually sees. Both see a warm OS page cache. Every cached
//! result is checked against the vanilla one.

use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, DuckLakeTable, MergeOptions, MetadataWriter, SqliteMetadataProvider,
    SqliteMetadataWriter,
};
use datafusion_subplan_cache::datafusion::arrow::array::RecordBatch;
use datafusion_subplan_cache::datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion_subplan_cache::datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion_subplan_cache::datafusion::logical_expr::{col, lit};
use datafusion_subplan_cache::datafusion::prelude::SessionContext;
use datafusion_subplan_cache::{DeltaSource, DuckLakeSource, SubplanCache};
use deltalake::protocol::SaveMode;
use serde_json::{Value, json};

const REGIONS: [&str; 3] = ["AP", "EU", "US"];

fn de(e: deltalake::DeltaTableError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}
fn dl(e: DuckLakeError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Rows of one region for the hour starting at `start` (Unix seconds).
/// `salt` varies the values so micro-batches differ from hourly files.
fn rows_sql(start: i64, region_idx: usize, rows: i64, span_secs: i64, salt: i64) -> String {
    let region = REGIONS[region_idx];
    format!(
        "SELECT arrow_cast(to_timestamp_micros({start} * 1000000 + value * {step}), 'Timestamp(Microsecond, Some(\"UTC\"))') AS ts,
                '{region}' AS region,
                (value * 7919 + {salt}) % 500 AS device_id,
                CAST((value * 7919 + {region_idx} * 13 + {salt}) % 10000 AS DOUBLE) / 10.0 + {offset} AS latency_ms,
                CAST(CASE WHEN (value * 31 + {salt}) % 100 < 2 THEN 500 ELSE 200 END AS INT) AS status
         FROM range(0, {rows})",
        step = span_secs * 1_000_000 / rows,
        // A daily pattern plus per-hour jitter, so hours differ.
        offset = 40.0 * ((start % 86_400) as f64 / 86_400.0 * std::f64::consts::TAU).sin()
            + ((start / 3600 * 37 + region_idx as i64 * 11) % 97) as f64 / 7.0,
    )
}

async fn gen_batches(sql: &str) -> Result<Vec<RecordBatch>> {
    SessionContext::new().sql(sql).await?.collect().await
}

fn ts_lit(secs: i64) -> datafusion_subplan_cache::datafusion::logical_expr::Expr {
    lit(ScalarValue::TimestampMicrosecond(Some(secs * 1_000_000), Some("UTC".into())))
}

// ---------------------------------------------------------------------------
// The two lake formats behind one interface
// ---------------------------------------------------------------------------

enum Lake {
    Delta { url: String },
    DuckLake { conn: String, ctx: SessionContext },
}

impl Lake {
    fn name(&self) -> &'static str {
        match self {
            Lake::Delta { .. } => "delta",
            Lake::DuckLake { .. } => "ducklake",
        }
    }

    async fn create(format: &str, dir: &std::path::Path) -> Result<Self> {
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir)?;
        match format {
            "delta" => {
                let url = url::Url::from_directory_path(dir).unwrap().to_string();
                Ok(Lake::Delta { url })
            }
            _ => {
                let data = dir.join("data");
                std::fs::create_dir_all(&data)?;
                let conn = format!("sqlite:{}?mode=rwc", dir.join("catalog.db").display());
                let writer = SqliteMetadataWriter::new_with_init(&conn).await.map_err(dl)?;
                writer.set_data_path(data.to_str().unwrap()).map_err(dl)?;
                let snapshot = writer.create_snapshot().map_err(dl)?;
                writer.get_or_create_schema("main", None, snapshot).map_err(dl)?;
                let ctx = Self::ducklake_ctx(&conn).await?;
                ctx.sql(
                    "CREATE TABLE lake.main.events (ts TIMESTAMP, region VARCHAR NOT NULL, device_id BIGINT NOT NULL,
                                                    latency_ms DOUBLE NOT NULL, status INT NOT NULL)",
                )
                .await?
                .collect()
                .await?;
                Ok(Lake::DuckLake { conn, ctx })
            }
        }
    }

    async fn ducklake_ctx(conn: &str) -> Result<SessionContext> {
        let writer = SqliteMetadataWriter::new(conn).await.map_err(dl)?;
        let provider = SqliteMetadataProvider::new(conn).await.map_err(dl)?;
        let ctx = SessionContext::new();
        DuckLakeCatalog::register(
            &ctx,
            "lake",
            DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).map_err(dl)?,
        );
        Ok(ctx)
    }

    async fn delta(&self) -> Result<deltalake::DeltaTable> {
        let Lake::Delta { url } = self else { unreachable!() };
        let u = url::Url::parse(url).unwrap();
        match deltalake::open_table(u.clone()).await {
            Ok(t) => Ok(t),
            Err(_) => deltalake::DeltaTable::try_from_url(u).await.map_err(de),
        }
    }

    /// One commit with `rows` rows per region spread over `span` seconds
    /// from `start`: one file per region.
    async fn append(&self, start: i64, rows: i64, span: i64, salt: i64) -> Result<()> {
        match self {
            Lake::Delta { .. } => {
                let mut batches = vec![];
                for r in 0..REGIONS.len() {
                    batches.extend(gen_batches(&rows_sql(start, r, rows, span, salt)).await?);
                }
                self.delta()
                    .await?
                    .write(batches)
                    .with_save_mode(SaveMode::Append)
                    .with_partition_columns(["region"])
                    .await
                    .map_err(de)?;
            }
            Lake::DuckLake { ctx, .. } => {
                // One INSERT per region: one file each. (Each is its own
                // DuckLake snapshot; the cache refreshes after all three.)
                for r in 0..REGIONS.len() {
                    let sql = format!(
                        "INSERT INTO lake.main.events SELECT arrow_cast(ts, 'Timestamp(Microsecond, None)') AS ts, region, device_id, latency_ms, status FROM ({})",
                        rows_sql(start, r, rows, span, salt)
                    );
                    ctx.sql(&sql).await?.collect().await?;
                }
            }
        }
        Ok(())
    }

    /// DELETE rows of `region` in [from, to).
    async fn delete_range(&self, region: &str, from: i64, to: i64) -> Result<String> {
        match self {
            Lake::Delta { .. } => {
                let pred = col("region").eq(lit(region)).and(col("ts").gt_eq(ts_lit(from))).and(col("ts").lt(ts_lit(to)));
                let (_, m) = self.delta().await?.delete().with_predicate(pred).await.map_err(de)?;
                Ok(format!("{} file(s) rewritten", m.num_removed_files))
            }
            Lake::DuckLake { ctx, .. } => {
                let sql = format!(
                    "DELETE FROM lake.main.events WHERE region = '{region}' AND ts >= to_timestamp_micros({}) AND ts < to_timestamp_micros({})",
                    from * 1_000_000,
                    to * 1_000_000
                );
                ctx.sql(&sql).await?.collect().await?;
                Ok("delete file(s) added".into())
            }
        }
    }

    async fn delete_device(&self, device: i64) -> Result<String> {
        match self {
            Lake::Delta { .. } => {
                let (_, m) = self.delta().await?.delete().with_predicate(col("device_id").eq(lit(device))).await.map_err(de)?;
                Ok(format!("{} file(s) rewritten", m.num_removed_files))
            }
            Lake::DuckLake { ctx, .. } => {
                ctx.sql(&format!("DELETE FROM lake.main.events WHERE device_id = {device}")).await?.collect().await?;
                Ok("delete files added".into())
            }
        }
    }

    /// UPDATE latency of `region` in [from, to).
    async fn update_range(&self, region: &str, from: i64, to: i64) -> Result<String> {
        match self {
            Lake::Delta { .. } => {
                let pred = col("region").eq(lit(region)).and(col("ts").gt_eq(ts_lit(from))).and(col("ts").lt(ts_lit(to)));
                let (_, m) = self
                    .delta()
                    .await?
                    .update()
                    .with_predicate(pred)
                    .with_update("latency_ms", col("latency_ms") * lit(2.0))
                    .await
                    .map_err(de)?;
                Ok(format!("{} rows updated, {} file(s) rewritten", m.num_updated_rows, m.num_removed_files))
            }
            Lake::DuckLake { ctx, .. } => {
                let sql = format!(
                    "UPDATE lake.main.events SET latency_ms = latency_ms * 2 WHERE region = '{region}' AND ts >= to_timestamp_micros({}) AND ts < to_timestamp_micros({})",
                    from * 1_000_000,
                    to * 1_000_000
                );
                ctx.sql(&sql).await?.collect().await?;
                Ok("rows updated".into())
            }
        }
    }

    /// Compact files smaller than `target` bytes.
    async fn compact(&self, target: u64) -> Result<String> {
        match self {
            Lake::Delta { .. } => {
                let (_, m) = self
                    .delta()
                    .await?
                    .optimize()
                    .with_target_size(std::num::NonZeroU64::new(target).unwrap())
                    .await
                    .map_err(de)?;
                Ok(format!("{} file(s) → {}", m.num_files_removed, m.num_files_added))
            }
            Lake::DuckLake { conn, .. } => {
                let ctx = Self::ducklake_ctx(conn).await?;
                let provider = ctx.catalog("lake").unwrap().schema("main").unwrap().table("events").await?.unwrap();
                let table = (provider.as_ref() as &dyn std::any::Any)
                    .downcast_ref::<DuckLakeTable>()
                    .expect("a DuckLakeTable")
                    .clone();
                let r = table
                    .merge_adjacent_files(&ctx.state(), MergeOptions { target_file_size: target, max_merged_files: 1024, min_file_size: 0 })
                    .await
                    .map_err(dl)?;
                Ok(format!("{} file(s) → {}", r.files_processed, r.files_created))
            }
        }
    }

    async fn register(&self, cache: &SubplanCache, ctx: &SessionContext) -> Result<()> {
        match self {
            Lake::Delta { url } => {
                cache.register_table(ctx, "events", DeltaSource::open(url).await?).await?;
            }
            Lake::DuckLake { conn, .. } => {
                cache.register_table(ctx, "events", DuckLakeSource::open_sqlite(conn, "main", "events").await?).await?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Measuring
// ---------------------------------------------------------------------------

struct Bench {
    cache: SubplanCache,
    ctx: SessionContext,
    results: Vec<Value>,
}

/// Rows as strings (sorted), for comparing cached vs vanilla output.
fn rows_of(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let opts = FormatOptions::default();
    let mut rows = vec![];
    for b in batches {
        let fmts: Vec<_> = b.columns().iter().map(|c| ArrayFormatter::try_new(c.as_ref(), &opts).unwrap()).collect();
        for i in 0..b.num_rows() {
            rows.push(fmts.iter().map(|f| f.value(i).to_string()).collect());
        }
    }
    rows.sort();
    rows
}

fn same(a: &[Vec<String>], b: &[Vec<String>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.iter().zip(y).all(|(u, v)| {
                u == v
                    || match (u.parse::<f64>(), v.parse::<f64>()) {
                        (Ok(p), Ok(q)) => (p - q).abs() <= 1e-9 * p.abs().max(q.abs()).max(1.0),
                        _ => false,
                    }
            })
        })
}

fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e5).round() / 100.0
}

impl Bench {
    fn new() -> Self {
        let cache = SubplanCache::builder().segment_files(1).build();
        let ctx = cache.session_context();
        Self { cache, ctx, results: vec![] }
    }

    fn total_files(&self) -> usize {
        self.cache.snapshot("events").map(|s| s.files().len()).unwrap_or(0)
    }

    /// Vanilla DataFusion over the same snapshot: median of 3 runs.
    async fn vanilla(&self, sql: &str) -> Result<(f64, Vec<Vec<String>>)> {
        let snap = self.cache.snapshot("events").unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("events", snap.inner.provider().await?)?;
        let mut times = vec![];
        let mut out = vec![];
        for _ in 0..3 {
            let t = Instant::now();
            out = ctx.sql(sql).await?.collect().await?;
            times.push(t.elapsed());
        }
        times.sort();
        Ok((ms(times[1]), rows_of(&out)))
    }

    /// Run `sql` once through the cache and once (x3) vanilla; record it.
    async fn measure(&mut self, extra: Value, sql: &str) -> Result<Value> {
        let (vanilla_ms, expected) = self.vanilla(sql).await?;
        self.cache.take_segment_uses();
        self.cache.take_reports();
        let t = Instant::now();
        let out = self.ctx.sql(sql).await?.collect().await?;
        let cached_ms = ms(t.elapsed());
        let got = rows_of(&out);
        if !same(&got, &expected) {
            eprintln!("expected: {expected:?}\ngot:      {got:?}");
            for e in self.cache.take_events() {
                eprintln!("{e}");
            }
            panic!("cached result differs from vanilla for {sql}");
        }
        let uses = self.cache.take_segment_uses();
        let reports = self.cache.take_reports();
        let hits: usize = reports.iter().map(|r| r.hits.len()).sum();
        let sum = |f: &dyn Fn(&datafusion_subplan_cache::SegmentUse) -> usize| uses.iter().map(f).sum::<usize>();
        let reused = sum(&|u| u.reused.iter().map(Vec::len).sum());
        let computed = sum(&|u| u.computed.iter().map(Vec::len).sum());
        let edge = sum(&|u| u.edge.len());
        let skipped = sum(&|u| u.skipped);
        let lifted: Vec<String> = uses.iter().flat_map(|u| u.lifted.clone()).collect();
        let mut v = json!({
            "vanilla_ms": vanilla_ms, "cached_ms": cached_ms, "speedup": (vanilla_ms / cached_ms.max(0.01) * 10.0).round() / 10.0,
            "whole_hits": hits, "files_reused": reused, "files_computed": computed, "files_edge": edge,
            "files_skipped": skipped, "files_total": self.total_files(), "lifted": lifted, "rows": out.iter().map(|b| b.num_rows()).sum::<usize>(),
        });
        for (k, val) in extra.as_object().unwrap() {
            v[k] = val.clone();
        }
        println!(
            "  {:<40} vanilla {:>8.1} ms   cached {:>8.1} ms   ×{:<6} hits {} reused {} computed {} edge {} skipped {} / {} files",
            v["label"].as_str().unwrap_or(""),
            vanilla_ms,
            cached_ms,
            v["speedup"],
            hits,
            reused,
            computed,
            edge,
            skipped,
            v["files_total"]
        );
        self.results.push(v.clone());
        Ok(v)
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// The dashboard's core aggregate. Many different queries are built on it.
fn hourly(filter: &str) -> String {
    format!(
        "SELECT region, date_trunc('hour', ts) AS hour, count(*) AS requests,
                avg(latency_ms) AS avg_latency, max(latency_ms) AS max_latency,
                sum(CASE WHEN status >= 500 THEN 1 ELSE 0 END) AS errors
         FROM events {filter} GROUP BY region, date_trunc('hour', ts)"
    )
}

/// `now()` pinned to `t` (Unix seconds). DataFusion folds `now()` into a
/// literal anyway; pinning it lets the vanilla and cached runs of one step
/// see exactly the same window.
fn now_at(t: i64) -> String {
    format!("to_timestamp_seconds({t})")
}

fn cross_queries(now: i64) -> Vec<(&'static str, String)> {
    let now = now_at(now);
    let h = hourly("");
    vec![
        ("Hourly rollup per region (first query)", format!("{h} ORDER BY region, hour")),
        ("Top 10 slowest hours (ORDER BY … LIMIT)", format!("{h} ORDER BY avg_latency DESC LIMIT 10")),
        (
            "Error-rate alert (filter on aggregate)",
            format!("SELECT *, errors * 1.0 / requests AS err_rate FROM ({h}) WHERE errors * 1.0 / requests > 0.0201"),
        ),
        (
            "6-hour moving average (window)",
            format!(
                "SELECT *, avg(avg_latency) OVER (PARTITION BY region ORDER BY hour ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS ma6 FROM ({h})"
            ),
        ),
        (
            "Hour-over-hour delta (self-join)",
            format!(
                "WITH h AS ({h})
                 SELECT a.*, b.requests AS prev_requests, b.avg_latency AS prev_avg, b.max_latency AS prev_max, b.errors AS prev_errors
                 FROM h a JOIN h b ON a.region = b.region AND a.hour = b.hour + INTERVAL '1 hour'"
            ),
        ),
        (
            "Daily summary (re-aggregation)",
            format!(
                "SELECT region, date_trunc('day', hour) AS day, sum(requests) AS requests, avg(avg_latency) AS avg_hourly_latency,
                        max(max_latency) AS max_latency, sum(errors) AS errors
                 FROM ({h}) GROUP BY region, date_trunc('day', hour)"
            ),
        ),
        ("EU only (WHERE region = 'EU')", format!("{} ORDER BY hour", hourly("WHERE region = 'EU'"))),
        ("Last 24 h (WHERE ts >= now() - 24h)", format!("{} ORDER BY region, hour", hourly(&format!("WHERE ts >= {now} - INTERVAL '24 hours'")))),
        (
            "EU + US, last 48 h (two lifted filters)",
            format!("{} ORDER BY region, hour", hourly(&format!("WHERE region IN ('EU', 'US') AND ts >= {now} - INTERVAL '48 hours'"))),
        ),
    ]
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

struct Setup {
    hours: i64,
    rows: i64,
    /// Start of the newest hour of history (Unix seconds).
    current_hour: i64,
    /// The simulated wall clock: inside the newest hour.
    now: i64,
}

async fn build_history(lake: &Lake, s: &Setup) -> Result<Duration> {
    let t = Instant::now();
    for h in 0..s.hours {
        lake.append(s.current_hour - (s.hours - 1 - h) * 3600, s.rows, 3600, 0).await?;
    }
    Ok(t.elapsed())
}

/// 1. Different queries share work: each query runs for the first time,
/// after only the ones before it.
async fn scenario_cross(lake: &Lake, now: i64, all: &mut Vec<Value>) -> Result<()> {
    println!("━━ [{}] different queries, each seen for the first time", lake.name());
    let mut b = Bench::new();
    lake.register(&b.cache, &b.ctx).await?;
    for (i, (label, sql)) in cross_queries(now).into_iter().enumerate() {
        b.measure(json!({"scenario": "cross_query", "format": lake.name(), "step": i, "label": label}), &sql).await?;
    }
    all.extend(b.results);
    Ok(())
}

/// 2. Time series: a new hour lands every step; two dashboard panels refresh.
async fn scenario_timeseries(lake: &Lake, s: &mut Setup, steps: i64, all: &mut Vec<Value>) -> Result<()> {
    println!("━━ [{}] time series: a new hour every step", lake.name());
    let mut b = Bench::new();
    lake.register(&b.cache, &b.ctx).await?;
    for k in 0..=steps {
        if k > 0 {
            s.current_hour += 3600;
            s.now += 3600;
            lake.append(s.current_hour, s.rows, 3600, 0).await?;
            b.cache.refresh(&b.ctx).await?;
        }
                let now = now_at(s.now);
        let last6 = format!(
            "SELECT region, count(*) AS requests, avg(latency_ms) AS avg_latency, max(latency_ms) AS max_latency
             FROM events WHERE ts >= {now} - INTERVAL '6 hours' GROUP BY region ORDER BY region"
        );
        let daily = "SELECT region, date_trunc('day', ts) AS day, count(*) AS requests, avg(latency_ms) AS avg_latency,
                            sum(CASE WHEN status >= 500 THEN 1 ELSE 0 END) AS errors
                     FROM events GROUP BY region, date_trunc('day', ts) ORDER BY region, day";
        let label = if k == 0 { "cold".to_string() } else { format!("+{k}h") };
        b.measure(json!({"scenario": "timeseries", "panel": "last_6h", "format": lake.name(), "step": k, "label": format!("last 6h {label}")}), &last6)
            .await?;
        b.measure(json!({"scenario": "timeseries", "panel": "daily_all_time", "format": lake.name(), "step": k, "label": format!("daily {label}")}), daily)
            .await?;
    }
    all.extend(b.results);
    Ok(())
}

/// 3. Table maintenance: appends, streaming micro-batches, DELETE, UPDATE,
/// compaction. After each, the same all-time dashboard query.
async fn scenario_maintenance(lake: &Lake, s: &mut Setup, all: &mut Vec<Value>) -> Result<()> {
    println!("━━ [{}] table maintenance", lake.name());
    let mut b = Bench::new();
    lake.register(&b.cache, &b.ctx).await?;
    let q = format!("{} ORDER BY region, hour", hourly(""));
    let mut step = 0;
    async fn run(b: &mut Bench, q: &str, step: &mut i32, format: &str, label: &str, op: String) -> Result<Value> {
        *step += 1;
        b.measure(json!({"scenario": "maintenance", "format": format, "step": *step, "label": label, "op": op}), q).await
    }
    let f = lake.name();
    run(&mut b, &q, &mut step, f, "first query (cold cache)", String::new()).await?;
    run(&mut b, &q, &mut step, f, "same query again", String::new()).await?;

    s.current_hour += 3600;
    lake.append(s.current_hour, s.rows, 3600, 0).await?;
    b.cache.refresh(&b.ctx).await?;
    run(&mut b, &q, &mut step, f, "APPEND one hour", "3 files added".into()).await?;

    // A streaming writer: 4 small commits within the next hour.
    let micro = (s.rows / 20).max(1);
    for m in 0..4 {
        lake.append(s.current_hour + 3600 + m * 900, micro, 900, 1 + m).await?;
    }
    b.cache.refresh(&b.ctx).await?;
    run(&mut b, &q, &mut step, f, "4 streaming micro-batches", "12 small files added".into()).await?;

    let day_ago = s.current_hour - 24 * 3600;
    let op = lake.delete_range("US", day_ago, day_ago + 3600).await?;
    b.cache.refresh(&b.ctx).await?;
    run(&mut b, &q, &mut step, f, "DELETE one region-hour", op).await?;

    let two_days_ago = s.current_hour - 48 * 3600;
    let op = lake.update_range("AP", two_days_ago, two_days_ago + 3600).await?;
    b.cache.refresh(&b.ctx).await?;
    run(&mut b, &q, &mut step, f, "UPDATE one region-hour", op).await?;

    // Compact the small (micro-batch) files; hourly files are larger than
    // the target and stay.
    // Hourly files: ~1.7 MB (Delta), ~0.5 MB (DuckLake); micro-batch files
    // are ~10× smaller.
    let target = env_or("BENCH_COMPACT_TARGET", if f == "delta" { 1_000_000u64 } else { 300_000 });
    let op = lake.compact(target).await?;
    b.cache.refresh(&b.ctx).await?;
    run(&mut b, &q, &mut step, f, "OPTIMIZE / compact small files", op).await?;

    let op = lake.delete_device(7).await?;
    b.cache.refresh(&b.ctx).await?;
    run(&mut b, &q, &mut step, f, "DELETE touching every file", op).await?;

    all.extend(b.results);
    Ok(())
}

/// 4. A new process with a warm persistent store (local directory).
async fn scenario_store(lake: &Lake, dir: &std::path::Path, now: i64, all: &mut Vec<Value>) -> Result<()> {
    println!("━━ [{}] persistent store: a second process", lake.name());
    let store = dir.join("plan-cache");
    let _ = std::fs::remove_dir_all(&store);
    std::fs::create_dir_all(&store)?;
    let q = format!("{} ORDER BY region, hour", hourly(""));
    {
        let cache = SubplanCache::builder().segment_files(1).store(store.to_str().unwrap())?.build();
        let ctx = cache.session_context();
        lake.register(&cache, &ctx).await?;
        ctx.sql(&q).await?.collect().await?;
        cache.flush().await?;
    }
    let cache = SubplanCache::builder().segment_files(1).store(store.to_str().unwrap())?.build();
    let ctx = cache.session_context();
    lake.register(&cache, &ctx).await?;
    let mut b = Bench { cache, ctx, results: vec![] };
    b.measure(json!({"scenario": "store", "format": lake.name(), "step": 0, "label": "new process, entries read from disk"}), &q).await?;
    let q24 = format!("{} ORDER BY region, hour", hourly(&format!("WHERE ts >= {} - INTERVAL '24 hours'", now_at(now))));
    b.measure(json!({"scenario": "store", "format": lake.name(), "step": 1, "label": "new process, last 24 h"}), &q24).await?;
    all.extend(b.results);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("BENCH_DIR").unwrap_or_else(|_| "data/bench".into()));
    let out = std::env::var("BENCH_OUT").unwrap_or_else(|_| "crates/datafusion-subplan-cache/docs/bench/results.json".into());
    let formats = std::env::var("BENCH_FORMATS").unwrap_or_else(|_| "delta,ducklake".into());
    let hours: i64 = env_or("BENCH_HOURS", 72);
    let rows: i64 = env_or("BENCH_ROWS", 200_000);
    let steps: i64 = env_or("BENCH_STEPS", 12);
    std::fs::create_dir_all(&root)?;
    let root = root.canonicalize()?;

    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let mut all = vec![];
    let mut meta = vec![];
    for format in formats.split(',') {
        let dir = root.join(format);
        let lake = Lake::create(format, &dir).await?;
        let mut s = Setup { hours, rows, current_hour: now - now % 3600, now };
        let took = build_history(&lake, &s).await?;
        println!("[{format}] wrote {hours} h × 3 regions × {rows} rows in {took:.1?}");
        meta.push(json!({"format": format, "generate_s": took.as_secs_f64()}));
        scenario_cross(&lake, s.now, &mut all).await?;
        scenario_store(&lake, &dir, s.now, &mut all).await?;
        scenario_timeseries(&lake, &mut s, steps, &mut all).await?;
        scenario_maintenance(&lake, &mut s, &mut all).await?;
    }

    let doc = json!({
        "setup": {
            "hours": hours, "rows_per_region_hour": rows, "regions": REGIONS.len(),
            "rows_total": hours * rows * REGIONS.len() as i64, "timeseries_steps": steps,
            "cpu": std::process::Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]).output().ok()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()),
            "threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            "formats": meta,
        },
        "results": all,
    });
    if let Some(p) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap())?;
    println!("results → {out}");
    Ok(())
}
