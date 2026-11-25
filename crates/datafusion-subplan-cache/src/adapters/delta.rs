//! Delta Lake tables (delta-rs).
//!
//! File identity = path + size + modification time + deletion vector, so a
//! DELETE that adds a deletion vector to an existing file changes that
//! file's identity, and rewritten files get new ones.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::prelude::SessionContext;
use deltalake::DeltaTable;
use futures::TryStreamExt;
use tokio::sync::Mutex;

use crate::table::{ColumnStats, DataFile, FileSnapshot, ImmutableFileTable};

fn de(e: deltalake::DeltaTableError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// A Delta table, followed to its latest version on every refresh.
pub struct DeltaSource {
    table: Mutex<DeltaTable>,
}

impl DeltaSource {
    pub fn new(table: DeltaTable) -> Self {
        Self { table: Mutex::new(table) }
    }

    /// Open the table at `url` (`file:///…/`, `gs://…`, `s3://…`).
    pub async fn open(url: &str) -> Result<Self> {
        let url = url::Url::parse(url).map_err(|e| DataFusionError::Plan(format!("bad table url {url}: {e}")))?;
        Ok(Self::new(deltalake::open_table(url).await.map_err(de)?))
    }
}

/// One version of a Delta table.
pub struct DeltaSnapshot {
    table: DeltaTable,
    files: Vec<DataFile>,
}

impl std::fmt::Debug for DeltaSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeltaSnapshot(v{}, {} files)", self.version(), self.files.len())
    }
}

impl DeltaSnapshot {
    /// The table at its currently loaded version.
    pub fn capture(table: &DeltaTable) -> Result<Self> {
        let state = table.snapshot().map_err(de)?;
        let schema = state.snapshot().arrow_schema();
        let mut files: Vec<DataFile> = state
            .log_data()
            .iter()
            .map(|f| {
                let dv = f.deletion_vector_descriptor().map(|d| format!("{d:?}")).unwrap_or_default();
                let stats = file_stats(&schema, f.stats().as_deref(), &f.partition_values_map());
                DataFile::new(f.path().to_string(), &format!("{}|{}|{dv}", f.size(), f.modification_time()))
                    .with_all_stats(Arc::new(stats))
            })
            .collect();
        files.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(Self { table: table.clone(), files })
    }

    /// The table at this version.
    pub fn table(&self) -> &DeltaTable {
        &self.table
    }
}

/// Column statistics of one add action: its `stats` JSON (`numRecords`,
/// `minValues`, `maxValues`, `nullCount`) plus its partition values, which
/// are exact. Deletion vectors only remove rows, so the bounds stay valid.
///
/// Left out: string columns (Delta truncates their min/max), floats, nested
/// columns. Timestamps are written at millisecond precision, so `max` is
/// widened to the end of its millisecond.
fn file_stats(schema: &Schema, json: Option<&str>, partitions: &HashMap<String, Option<String>>) -> HashMap<String, ColumnStats> {
    let mut out = HashMap::new();
    let Some(stats) = json.and_then(|j| serde_json::from_str::<serde_json::Value>(j).ok()) else { return out };
    let Some(rows) = stats.get("numRecords").and_then(|v| v.as_u64()) else { return out };
    for field in schema.fields() {
        let ty = field.data_type();
        let name = field.name();
        let null = || ScalarValue::try_from(ty).ok();
        if let Some(value) = partitions.get(name) {
            let s = match value {
                Some(v) => ScalarValue::Utf8(Some(v.clone())).cast_to(ty).ok().map(|v| ColumnStats {
                    min: v.clone(),
                    max: v,
                    null_count: 0,
                    row_count: rows,
                }),
                None => null().map(|n| ColumnStats { min: n.clone(), max: n, null_count: rows, row_count: rows }),
            };
            if let Some(s) = s {
                out.insert(name.clone(), s);
            }
            continue;
        }
        if ty.is_floating() || ty.is_nested() || matches!(ty, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View | DataType::Binary | DataType::LargeBinary | DataType::BinaryView) {
            continue;
        }
        let Some(nulls) = stats.pointer(&format!("/nullCount/{name}")).and_then(|v| v.as_u64()) else { continue };
        let value = |which: &str| -> Option<ScalarValue> {
            match stats.pointer(&format!("/{which}/{name}")) {
                None | Some(serde_json::Value::Null) => None,
                Some(serde_json::Value::String(s)) => ScalarValue::Utf8(Some(s.clone())).cast_to(ty).ok(),
                Some(serde_json::Value::Bool(b)) => ScalarValue::Boolean(Some(*b)).cast_to(ty).ok(),
                Some(serde_json::Value::Number(n)) => match n.as_i64() {
                    Some(i) => ScalarValue::Int64(Some(i)).cast_to(ty).ok(),
                    None => ScalarValue::Utf8(Some(n.to_string())).cast_to(ty).ok(),
                },
                _ => None,
            }
        };
        let s = if nulls == rows {
            null().map(|n| ColumnStats { min: n.clone(), max: n, null_count: rows, row_count: rows })
        } else {
            match (value("minValues"), value("maxValues").map(end_of_millisecond)) {
                (Some(min), Some(max)) => Some(ColumnStats { min, max, null_count: nulls, row_count: rows }),
                _ => None,
            }
        };
        if let Some(s) = s {
            out.insert(name.clone(), s);
        }
    }
    out
}

/// The last instant of the millisecond `v` falls in (for timestamps finer
/// than milliseconds).
fn end_of_millisecond(v: ScalarValue) -> ScalarValue {
    match v {
        ScalarValue::TimestampMicrosecond(Some(t), tz) => ScalarValue::TimestampMicrosecond(Some(t.div_euclid(1000) * 1000 + 999), tz),
        ScalarValue::TimestampNanosecond(Some(t), tz) => {
            ScalarValue::TimestampNanosecond(Some(t.div_euclid(1_000_000) * 1_000_000 + 999_999), tz)
        }
        other => other,
    }
}

#[async_trait]
impl FileSnapshot for DeltaSnapshot {
    fn version(&self) -> String {
        self.table.version().unwrap_or(0).to_string()
    }

    fn files(&self) -> &[DataFile] {
        &self.files
    }

    async fn provider(&self) -> Result<Arc<dyn TableProvider>> {
        Ok(Arc::new(self.table.table_provider().build().await?))
    }

    async fn provider_for(&self, files: &[&DataFile]) -> Result<Option<Arc<dyn TableProvider>>> {
        let scan = self.table.table_provider().with_file_paths(files.iter().map(|f| f.path.clone())).build().await?;
        Ok(Some(Arc::new(scan)))
    }
}

#[async_trait]
impl ImmutableFileTable for DeltaSource {
    async fn latest(&self) -> Result<Arc<dyn FileSnapshot>> {
        let mut table = self.table.lock().await;
        table.update_state().await.map_err(de)?;
        Ok(Arc::new(DeltaSnapshot::capture(&table)?))
    }

    /// The current version, every version committed at or after `since`,
    /// and the version that was current at `since`.
    async fn snapshots_since(&self, since: SystemTime) -> Result<Vec<Arc<dyn FileSnapshot>>> {
        let table = self.table.lock().await.clone();
        let cutoff = since.duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0);
        let current = table.version().unwrap_or(0);
        let mut out: Vec<Arc<dyn FileSnapshot>> = vec![Arc::new(DeltaSnapshot::capture(&table)?)];
        // Newest first: versions current, current-1, …
        let history: Vec<_> = table.history(None).try_collect().await.map_err(de)?;
        for (i, info) in history.iter().enumerate() {
            let Some(v) = current.checked_sub(i as u64) else { break };
            if v < current {
                let mut old = table.clone();
                old.load_version(v).await.map_err(de)?;
                out.push(Arc::new(DeltaSnapshot::capture(&old)?));
            }
            if info.timestamp.unwrap_or(0) < cutoff {
                break; // v was committed before the cutoff: current at the cutoff
            }
        }
        Ok(out)
    }

    fn attach(&self, ctx: &SessionContext) -> Result<()> {
        // Registers the table's object store with the session.
        self.table.try_lock().map_err(|e| DataFusionError::External(Box::new(e)))?.update_datafusion_session(&ctx.state()).map_err(de)
    }
}
