//! A Delta table, cached.
//!
//!     cargo run --release -p datafusion-subplan-cache --features delta --example quickstart_delta
//!
//! Creates a Delta table with three commits, runs an aggregate, then deletes
//! a few rows: Delta rewrites one file, and only that file is read again.

use std::sync::Arc;

use datafusion_subplan_cache::datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
use datafusion_subplan_cache::datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion_subplan_cache::datafusion::arrow::util::pretty::print_batches;
use datafusion_subplan_cache::datafusion::common::{DataFusionError, Result};
use datafusion_subplan_cache::datafusion::prelude::SessionContext;
use datafusion_subplan_cache::{DeltaSource, SubplanCache};
use deltalake::DeltaTable;
use deltalake::protocol::SaveMode;

const QUERY: &str = "SELECT store, count(*) AS orders, sum(amount) AS total FROM orders GROUP BY store ORDER BY store";

fn de(e: deltalake::DeltaTableError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

fn batch(ids: std::ops::Range<i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("store", DataType::Utf8, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let stores: Vec<&str> = ids.clone().map(|i| ["north", "south", "east"][(i % 3) as usize]).collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from_iter_values(ids.clone())),
            Arc::new(StringArray::from(stores)),
            Arc::new(Int64Array::from_iter_values(ids.map(|i| i % 100))),
        ],
    )
    .unwrap()
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
    let dir = std::env::temp_dir().join(format!("subplan-cache-delta-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let url = url::Url::from_directory_path(&dir).unwrap();
    let mut table = DeltaTable::try_from_url(url.clone()).await.map_err(de)?;
    for c in 0..3 {
        table = table.write(vec![batch(c * 100_000..(c + 1) * 100_000)]).with_save_mode(SaveMode::Append).await.map_err(de)?;
    }

    let cache = SubplanCache::builder().segment_files(1).build();
    let ctx = cache.session_context();
    cache.register_table(&ctx, "orders", DeltaSource::open(url.as_str()).await?).await?;

    run(&ctx, &cache, "first run").await?;

    // Any writer (this process or another) commits a change…
    let (_table, m) = table.delete().with_predicate("order_id < 10").await.map_err(de)?;
    println!("DELETE: {} file(s) rewritten\n", m.num_removed_files);
    // …and a refresh moves the cache's view to the new version.
    cache.refresh(&ctx).await?;
    run(&ctx, &cache, "after DELETE: only the rewritten file is read").await?;

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
