//! An append-only folder of Parquet files, cached.
//!
//!     cargo run --release -p datafusion-subplan-cache --example quickstart_parquet
//!
//! Writes three files, runs an aggregate twice (the repeat is answered from
//! the cache), adds a fourth file, and runs it again: only the new file is
//! read; the partial states of the first three come from the cache.

use datafusion_subplan_cache::datafusion::common::Result;
use datafusion_subplan_cache::datafusion::prelude::SessionContext;
use datafusion_subplan_cache::datafusion::arrow::util::pretty::print_batches;
use datafusion_subplan_cache::{ParquetFiles, SubplanCache};

const QUERY: &str = "SELECT kind, count(*) AS n, sum(amount) AS total FROM events GROUP BY kind ORDER BY kind";

async fn write_file(dir: &str, n: u64) -> Result<()> {
    let sql = format!(
        "COPY (SELECT CASE value % 3 WHEN 0 THEN 'click' WHEN 1 THEN 'view' ELSE 'buy' END AS kind, value AS amount
               FROM range({}, {}))
         TO '{dir}/part-{n:04}.parquet' STORED AS PARQUET",
        n * 1_000_000,
        (n + 1) * 1_000_000
    );
    SessionContext::new().sql(&sql).await?.collect().await?;
    Ok(())
}

async fn run(ctx: &SessionContext, cache: &SubplanCache, title: &str) -> Result<()> {
    println!("━━ {title}");
    let t = std::time::Instant::now();
    let batches = ctx.sql(QUERY).await?.collect().await?;
    print_batches(&batches)?;
    for e in cache.take_events() {
        println!("{e}");
    }
    for r in cache.take_reports() {
        for h in r.hits {
            println!("  ✔ whole result from cache: {}", h.label);
        }
    }
    println!("  ⏱  {:.1?}\n", t.elapsed());
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("subplan-cache-quickstart-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let dir = dir.to_str().unwrap().to_string();
    for n in 0..3 {
        write_file(&dir, n).await?;
    }

    // 1. A cache, installed into a session.
    let cache = SubplanCache::builder().segment_files(1).build();
    let ctx = cache.session_context();

    // 2. The table: every *.parquet in the folder, re-listed on refresh.
    let events = ParquetFiles::open(&ctx, &format!("file://{dir}/")).await?;
    cache.register_table(&ctx, "events", events).await?;

    run(&ctx, &cache, "first run: every file read, partial states cached per file").await?;
    run(&ctx, &cache, "repeat").await?;

    write_file(&dir, 3).await?;
    cache.refresh(&ctx).await?;
    run(&ctx, &cache, "after a 4th file appeared: only that file read").await?;

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
