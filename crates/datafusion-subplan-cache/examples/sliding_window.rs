//! A "last 6 hours" dashboard over time-series files, with `now()`.
//!
//!     cargo run --release -p datafusion-subplan-cache --example sliding_window
//!
//! One Parquet file per hour and region (a stream sink partitioned by
//! region, say), 24 hours of history. The
//! query filters on `ts >= now() - interval '6 hours'`. DataFusion folds
//! `now()` into a different literal on every run, so the query text never
//! repeats; the cache instead decides the filter per file from the files'
//! min/max statistics:
//!
//! * files entirely inside the window reuse their cached partial states,
//! * files entirely before it are not read,
//! * the files the window boundary cuts through are read live.
//!
//! Then a new hour arrives, the window slides, and only that hour's files and
//! the boundary files are read. Finally the same machinery handles
//! `region = 'EU'`: any comparison, IN list or IS NULL on a column the files
//! are laid out by works the same way.

use datafusion_subplan_cache::datafusion::arrow::util::pretty::print_batches;
use datafusion_subplan_cache::datafusion::common::Result;
use datafusion_subplan_cache::datafusion::prelude::SessionContext;
use datafusion_subplan_cache::{ParquetFiles, SubplanCache};

/// Per region.
const ROWS_PER_HOUR: i64 = 200_000;

const LAST_6H: &str = "SELECT region, count(*) AS events, round(avg(latency_ms), 2) AS avg_latency
     FROM requests WHERE ts >= now() - interval '6 hours'
     GROUP BY region ORDER BY region";

const EU_ALL_TIME: &str = "SELECT date_trunc('day', ts) AS day, count(*) AS events
     FROM requests WHERE region = 'EU' GROUP BY day ORDER BY day";

/// The files of the hour starting at `start` (Unix seconds), one per
/// region, with rows spread evenly over the hour.
async fn write_hour(dir: &str, start: i64) -> Result<()> {
    for (i, region) in ["AP", "EU", "US"].into_iter().enumerate() {
        let sql = format!(
            "COPY (SELECT to_timestamp({start} + value * 3600 / {ROWS_PER_HOUR}) AS ts,
                          '{region}' AS region,
                          (value * 7919 + {i}) % 1000 AS latency_ms
                   FROM range(0, {ROWS_PER_HOUR}))
             TO '{dir}/hour-{start}-{region}.parquet' STORED AS PARQUET",
        );
        SessionContext::new().sql(&sql).await?.collect().await?;
    }
    Ok(())
}

async fn run(ctx: &SessionContext, cache: &SubplanCache, title: &str, sql: &str) -> Result<()> {
    println!("━━ {title}");
    let t = std::time::Instant::now();
    let batches = ctx.sql(sql).await?.collect().await?;
    let elapsed = t.elapsed();
    print_batches(&batches)?;
    for e in cache.take_events() {
        println!("{e}");
    }
    for u in cache.take_segment_uses() {
        println!(
            "  files: {} reused from cache, {} computed and cached, {} read live (edge), {} skipped",
            u.reused.iter().map(Vec::len).sum::<usize>(),
            u.computed.iter().map(Vec::len).sum::<usize>(),
            u.edge.len(),
            u.skipped
        );
    }
    println!("  ⏱  {elapsed:.1?}\n");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("subplan-cache-sliding-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let dir = dir.to_str().unwrap().to_string();

    // 24 hours of history, the newest file being the hour that started last.
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let current_hour = now - now % 3600;
    for h in 0..24 {
        write_hour(&dir, current_hour - (23 - h) * 3600).await?;
    }

    let cache = SubplanCache::builder().segment_files(1).build();
    let ctx = cache.session_context();
    // Reads each file's footer once for its column statistics.
    let requests = ParquetFiles::open(&ctx, &format!("file://{dir}/")).await?;
    cache.register_table(&ctx, "requests", requests).await?;

    run(&ctx, &cache, "last 6 hours, cold: files in the window computed and cached", LAST_6H).await?;
    run(&ctx, &cache, "last 6 hours again (a new now()): cached files reused", LAST_6H).await?;

    // The next hour's files land. Rather than wait an hour, simulate the
    // clock moving on by shifting `now()` in the query.
    write_hour(&dir, current_hour + 3600).await?;
    cache.refresh(&ctx).await?;
    let slid = LAST_6H.replace("now()", "(now() + interval '1 hour')");
    run(&ctx, &cache, "an hour later: the window slides, only the new hour is computed", &slid).await?;

    run(&ctx, &cache, "all-time EU per day, cold: AP and US files are skipped", EU_ALL_TIME).await?;
    write_hour(&dir, current_hour + 2 * 3600).await?;
    cache.refresh(&ctx).await?;
    run(&ctx, &cache, "all-time EU after another hour: one new file computed", EU_ALL_TIME).await?;

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
