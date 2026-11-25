//! A DuckLake table, cached.
//!
//!     cargo run --release -p datafusion-subplan-cache --features ducklake --example quickstart_ducklake
//!
//! Creates a DuckLake (SQLite catalog, local data files) with three inserts,
//! runs an aggregate, then appends and deletes: only the new file and the file
//! that got a delete file are read again.

use std::sync::Arc;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeError, MetadataWriter, SqliteMetadataProvider, SqliteMetadataWriter,
};
use datafusion_subplan_cache::datafusion::arrow::util::pretty::print_batches;
use datafusion_subplan_cache::datafusion::common::{DataFusionError, Result};
use datafusion_subplan_cache::datafusion::prelude::SessionContext;
use datafusion_subplan_cache::{DuckLakeSource, SubplanCache};

const QUERY: &str = "SELECT store, count(*) AS orders, sum(amount) AS total FROM orders GROUP BY store ORDER BY store";

fn dl(e: DuckLakeError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

fn rows(range: std::ops::Range<i64>) -> String {
    format!(
        "SELECT value AS order_id, CASE value % 3 WHEN 0 THEN 'north' WHEN 1 THEN 'south' ELSE 'east' END AS store,
                value % 100 AS amount FROM range({}, {})",
        range.start, range.end
    )
}

async fn run(ctx: &SessionContext, cache: &SubplanCache, title: &str) -> Result<()> {
    println!("━━ {title}");
    let batches = ctx.sql(QUERY).await?.collect().await?;
    print_batches(&batches)?;
    for e in cache.take_events() {
        println!("{e}");
    }
    println!();
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("subplan-cache-ducklake-{}", std::process::id()));
    let data = dir.join("data");
    std::fs::create_dir_all(&data)?;
    let conn = format!("sqlite:{}?mode=rwc", dir.join("catalog.db").display());

    // A writer session: any DuckLake client (DuckDB, another process) would do.
    let writer = SqliteMetadataWriter::new_with_init(&conn).await.map_err(dl)?;
    writer.set_data_path(data.to_str().unwrap()).map_err(dl)?;
    let snapshot = writer.create_snapshot().map_err(dl)?;
    writer.get_or_create_schema("main", None, snapshot).map_err(dl)?;
    let provider = SqliteMetadataProvider::new(&conn).await.map_err(dl)?;
    let lake = SessionContext::new();
    DuckLakeCatalog::register(&lake, "lake", DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).map_err(dl)?);
    lake.sql("CREATE TABLE lake.main.orders (order_id BIGINT NOT NULL, store VARCHAR NOT NULL, amount BIGINT NOT NULL)")
        .await?
        .collect()
        .await?;
    for c in 0..3 {
        lake.sql(&format!("INSERT INTO lake.main.orders {}", rows(c * 100_000..(c + 1) * 100_000))).await?.collect().await?;
    }

    let cache = SubplanCache::builder().segment_files(1).build();
    let ctx = cache.session_context();
    cache.register_table(&ctx, "orders", DuckLakeSource::open_sqlite(&conn, "main", "orders").await?).await?;

    run(&ctx, &cache, "first run").await?;

    lake.sql(&format!("INSERT INTO lake.main.orders {}", rows(300_000..400_000))).await?.collect().await?;
    lake.sql("DELETE FROM lake.main.orders WHERE order_id < 10").await?.collect().await?;
    // A refresh moves the cache's view to the catalog's latest snapshot.
    cache.refresh(&ctx).await?;
    run(&ctx, &cache, "after INSERT + DELETE: only the new file and the file with deletes are read").await?;

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
