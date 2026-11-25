//! Delta Lake tables of the demo: creating them and writing to them.
//! Reading (and caching) goes through the library's `DeltaSource`.

use std::collections::HashMap;
use std::path::Path;

use datafusion::arrow::array::RecordBatch;
use datafusion::common::{DataFusionError, Result};
use datafusion::prelude::SessionContext;
use datafusion_subplan_cache::{DeltaSource, SubplanCache};
use deltalake::DeltaTable;
use deltalake::protocol::SaveMode;
use url::Url;

fn de(e: deltalake::DeltaTableError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// The Delta tables of the demo: registered with the cache for reading,
/// with a handle kept here for writing. Every write is followed by a cache
/// refresh, so the next query sees the new version.
#[derive(Default)]
pub struct DeltaTables {
    tables: HashMap<String, DeltaTable>,
}

impl DeltaTables {
    pub async fn open(&mut self, cache: &SubplanCache, ctx: &SessionContext, name: &str, dir: &str) -> Result<()> {
        let table = deltalake::open_table(dir_url(dir)?).await.map_err(de)?;
        cache.register_table(ctx, name, DeltaSource::new(table.clone())).await?;
        self.tables.insert(name.to_string(), table);
        Ok(())
    }

    /// The latest version, for writing.
    async fn table(&self, name: &str) -> Result<DeltaTable> {
        let mut t = self
            .tables
            .get(name)
            .cloned()
            .ok_or_else(|| DataFusionError::Plan(format!("{name} is not a Delta table")))?;
        t.update_state().await.map_err(de)?;
        Ok(t)
    }

    async fn written(&mut self, cache: &SubplanCache, ctx: &SessionContext, name: &str, table: DeltaTable) -> Result<()> {
        self.tables.insert(name.to_string(), table);
        cache.refresh(ctx).await?;
        Ok(())
    }

    pub async fn append(&mut self, cache: &SubplanCache, ctx: &SessionContext, name: &str, batches: Vec<RecordBatch>) -> Result<()> {
        let table = self.table(name).await?.write(batches).with_save_mode(SaveMode::Append).await.map_err(de)?;
        self.written(cache, ctx, name, table).await
    }

    pub async fn delete(&mut self, cache: &SubplanCache, ctx: &SessionContext, name: &str, predicate: &str) -> Result<String> {
        let (table, m) = self.table(name).await?.delete().with_predicate(predicate).await.map_err(de)?;
        self.written(cache, ctx, name, table).await?;
        Ok(format!("{} files removed, {} files added", m.num_removed_files, m.num_added_files))
    }

    pub async fn update(&mut self, cache: &SubplanCache, ctx: &SessionContext, name: &str, predicate: &str, column: &str, value: &str) -> Result<String> {
        let (table, m) = self.table(name).await?.update().with_predicate(predicate).with_update(column, value).await.map_err(de)?;
        self.written(cache, ctx, name, table).await?;
        Ok(format!("{} files removed, {} files added", m.num_removed_files, m.num_added_files))
    }

    /// Compact small files. Files larger than `target_size` (default: the
    /// table's target, 100 MiB) are never rewritten. Returns (removed, added).
    pub async fn optimize(&mut self, cache: &SubplanCache, ctx: &SessionContext, name: &str, target_size: Option<u64>) -> Result<(u64, u64)> {
        let mut op = self.table(name).await?.optimize();
        if let Some(t) = target_size.and_then(std::num::NonZeroU64::new) {
            op = op.with_target_size(t);
        }
        let (table, m) = op.await.map_err(de)?;
        self.written(cache, ctx, name, table).await?;
        Ok((m.num_files_removed, m.num_files_added))
    }
}

pub fn dir_url(dir: &str) -> Result<Url> {
    if crate::location::is_remote(dir) {
        let dir = format!("{}/", dir.trim_end_matches('/'));
        return Url::parse(&dir).map_err(|e| DataFusionError::Plan(format!("bad location {dir}: {e}")));
    }
    std::fs::create_dir_all(dir)?;
    let abs = Path::new(dir).canonicalize()?;
    Url::from_directory_path(&abs).map_err(|_| DataFusionError::Plan(format!("bad path {dir}")))
}

/// Create a Delta table from a sequence of batches, one commit per element.
pub async fn create_with_commits(dir: &str, commits: Vec<Vec<RecordBatch>>) -> Result<()> {
    let mut table = DeltaTable::try_from_url(dir_url(dir)?).await.map_err(de)?;
    for batches in commits {
        table = table.write(batches).with_save_mode(SaveMode::Append).await.map_err(de)?;
    }
    Ok(())
}
