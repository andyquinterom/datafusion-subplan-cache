//! DuckLake tables (datafusion-ducklake).
//!
//! A DuckLake snapshot is a set of Parquet data files listed in the catalog
//! database, plus row-level changes that do not rewrite them:
//! * delete files (one per data file, replaced on each delete);
//! * inlined deletions (deleted row positions kept in the catalog);
//! * inlined rows (small inserts kept in the catalog instead of Parquet).
//!
//! File identity = catalog data file id + size + delete file + a hash of the
//! file's inlined deletions (+ the snapshot, for a compacted file whose rows
//! span snapshots the reader is not past). Inlined rows are one pseudo file,
//! `<table path>/__inlined__`, whose identity is the set of inlined row ids.
//!
//! The object store of the lake's data path must be registered on the
//! session's runtime (`ctx.register_object_store`), as for any DuckLake
//! catalog; local paths work as they are.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::common::{DataFusionError, Result};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion_ducklake::inlined_filter::{InlinedDataScan, InlinedFilter};
use datafusion_ducklake::metadata_provider::{
    ColumnWithTable, DataFileChange, DeleteFileChange, DuckLakeFileMetadata, DuckLakeInlinedData,
    DuckLakeInlinedDelete, DuckLakeNameMapping, DuckLakeStatistics, DuckLakeTableColumn, DuckLakeTableField,
    FileWithTable, SchemaMetadata, SnapshotMetadata, TableMetadata, TableWithSchema, ViewMetadata, ViewWithSchema,
};
use datafusion_ducklake::path_resolver::{parse_object_store_url, resolve_path};
use datafusion_ducklake::stats_filter::StatsFilter;
use datafusion_ducklake::{
    ColumnTag, DuckLakeError, DuckLakeTable, DuckLakeTableFile, DuckLakeTag, MetadataProvider, ObjectTag,
    PartitionSpec, SnapshotChangeMetadata, SortSpec, TagTarget,
};

use crate::store::stable_hash;
use crate::table::{DataFile, FileSnapshot, ImmutableFileTable};

fn de(e: DuckLakeError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Name of the pseudo file standing for a snapshot's inlined rows.
const INLINED: &str = "__inlined__";

/// One table of a DuckLake catalog, followed to the catalog's latest
/// snapshot on every refresh.
pub struct DuckLakeSource {
    provider: Arc<dyn MetadataProvider>,
    schema: String,
    table: String,
    object_store_url: Arc<ObjectStoreUrl>,
    catalog_path: String,
}

impl DuckLakeSource {
    /// Table `schema.table` of the catalog `provider` reads.
    pub fn new(provider: Arc<dyn MetadataProvider>, schema: &str, table: &str) -> Result<Self> {
        let (url, catalog_path) = parse_object_store_url(&provider.get_data_path().map_err(de)?).map_err(de)?;
        Ok(Self {
            provider,
            schema: schema.to_string(),
            table: table.to_string(),
            object_store_url: Arc::new(url),
            catalog_path,
        })
    }

    /// Table `schema.table` of a SQLite catalog (`sqlite:/path/catalog.db`).
    pub async fn open_sqlite(connection: &str, schema: &str, table: &str) -> Result<Self> {
        let provider = datafusion_ducklake::SqliteMetadataProvider::new(connection).await.map_err(de)?;
        Self::new(Arc::new(provider), schema, table)
    }

    /// Table `schema.table` of a PostgreSQL catalog (`postgresql://…`).
    #[cfg(feature = "ducklake-postgres")]
    pub async fn open_postgres(connection: &str, schema: &str, table: &str) -> Result<Self> {
        let provider = datafusion_ducklake::PostgresMetadataProvider::new(connection).await.map_err(de)?;
        Self::new(Arc::new(provider), schema, table)
    }

    pub fn provider(&self) -> &Arc<dyn MetadataProvider> {
        &self.provider
    }

    /// The table at `snapshot_id`, or `None` if it did not exist then.
    fn capture(&self, snapshot_id: i64) -> Result<Option<DuckLakeSnapshot>> {
        let p = self.provider.as_ref();
        let Some(schema) = p.get_schema_by_name(&self.schema, snapshot_id).map_err(de)? else {
            return Ok(None);
        };
        let Some(meta) = p.get_table_by_name(schema.schema_id, &self.table, snapshot_id).map_err(de)? else {
            return Ok(None);
        };
        let schema_path = resolve_path(&self.catalog_path, &schema.path, schema.path_is_relative).map_err(de)?;
        let table_path = resolve_path(&schema_path, &meta.path, meta.path_is_relative).map_err(de)?;
        let table_id = meta.table_id;

        let mut inlined_deletes: HashMap<i64, Vec<i64>> = HashMap::new();
        for d in p.get_inlined_deletes(table_id, snapshot_id).map_err(de)? {
            inlined_deletes.entry(d.data_file_id).or_default().push(d.row_id);
        }
        let mut files = Vec::new();
        let mut file_ids = HashMap::new();
        for f in p.get_table_files_for_select(table_id, snapshot_id).map_err(de)? {
            let path = resolve_path(&table_path, &f.file.path, f.file.path_is_relative).map_err(de)?;
            let identity = file_identity(&f, snapshot_id, inlined_deletes.get_mut(&f.data_file_id));
            file_ids.insert(path.clone(), f.data_file_id);
            files.push(DataFile::new(path, &identity));
        }

        let columns = p.get_table_structure(table_id, snapshot_id).map_err(de)?;
        let inlined = p.scan_inlined_data(table_id, snapshot_id, &columns, None).map_err(de)?;
        let inlined_rows: usize = inlined.batches.iter().map(|b| b.num_rows()).sum();
        let inlined_path = format!("{}/{INLINED}", table_path.trim_end_matches('/'));
        if inlined_rows > 0 {
            let mut ids: Vec<(i64, i64)> = p
                .get_inlined_data_with_row_ids(table_id, snapshot_id, &columns)
                .map_err(de)?
                .iter()
                .flat_map(|d| d.row_ids.iter().copied().zip(d.begin_snapshots.iter().copied()))
                .collect();
            ids.sort_unstable();
            // Without row ids from the backend, the snapshot is the only safe identity.
            let identity = if ids.len() == inlined_rows {
                stable_hash(&format!("{ids:?}"))
            } else {
                format!("snapshot {snapshot_id}")
            };
            files.push(DataFile::new(inlined_path.clone(), &identity));
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(Some(DuckLakeSnapshot {
            snapshot_id,
            table_id,
            table_name: meta.table_name,
            table_path,
            provider: self.provider.clone(),
            object_store_url: self.object_store_url.clone(),
            files,
            file_ids,
            inlined_path,
        }))
    }
}

/// Everything that pins the rows `f` contributes at `snapshot_id`.
fn file_identity(f: &DuckLakeTableFile, snapshot_id: i64, inlined_deletes: Option<&mut Vec<i64>>) -> String {
    let delete = f.delete_file.as_ref().map(|d| format!("{}:{}", d.path, d.file_size_bytes)).unwrap_or_default();
    let inlined = inlined_deletes
        .map(|rows| {
            rows.sort_unstable();
            rows.dedup();
            stable_hash(&format!("{rows:?}"))
        })
        .unwrap_or_default();
    // A compacted file holds rows of several snapshots; a reader on a
    // snapshot older than its newest rows sees only some of them.
    let visible = match f.partial_max {
        Some(max) if snapshot_id < max => snapshot_id.to_string(),
        _ => String::new(),
    };
    format!(
        "{}|{}|{:?}|{delete}|{inlined}|{visible}",
        f.data_file_id, f.file.file_size_bytes, f.delete_file_id
    )
}

/// One snapshot of a DuckLake table.
pub struct DuckLakeSnapshot {
    snapshot_id: i64,
    table_id: i64,
    table_name: String,
    table_path: String,
    provider: Arc<dyn MetadataProvider>,
    object_store_url: Arc<ObjectStoreUrl>,
    files: Vec<DataFile>,
    /// DataFile path → catalog data file id.
    file_ids: HashMap<String, i64>,
    inlined_path: String,
}

impl std::fmt::Debug for DuckLakeSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DuckLakeSnapshot({}, s{}, {} files)", self.table_name, self.snapshot_id, self.files.len())
    }
}

impl DuckLakeSnapshot {
    pub fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }

    fn table(&self, provider: Arc<dyn MetadataProvider>) -> Result<Arc<dyn TableProvider>> {
        let table = DuckLakeTable::new(
            self.table_id,
            self.table_name.clone(),
            provider,
            self.snapshot_id,
            self.object_store_url.clone(),
            self.table_path.clone(),
        )
        .map_err(de)?;
        Ok(Arc::new(table))
    }
}

#[async_trait]
impl FileSnapshot for DuckLakeSnapshot {
    fn version(&self) -> String {
        self.snapshot_id.to_string()
    }

    fn files(&self) -> &[DataFile] {
        &self.files
    }

    async fn provider(&self) -> Result<Arc<dyn TableProvider>> {
        self.table(self.provider.clone())
    }

    async fn provider_for(&self, files: &[&DataFile]) -> Result<Option<Arc<dyn TableProvider>>> {
        let mut keep = HashSet::new();
        let mut inlined = false;
        for f in files {
            if f.path == self.inlined_path {
                inlined = true;
            } else {
                let id = self.file_ids.get(&f.path).ok_or_else(|| {
                    DataFusionError::Internal(format!("{} is not a file of {self:?}", f.path))
                })?;
                keep.insert(*id);
            }
        }
        let subset = FileSubset { inner: self.provider.clone(), table_id: self.table_id, keep, inlined };
        self.table(Arc::new(subset)).map(Some)
    }
}

/// Parse a catalog snapshot timestamp (UTC, `YYYY-MM-DD HH:MM:SS[.f][+00]`).
fn snapshot_time(raw: &str) -> Option<i64> {
    let mut s = raw.trim();
    for suffix in ["Z", " UTC", "+00:00", "+00"] {
        if let Some(stripped) = s.strip_suffix(suffix) {
            s = stripped.trim();
            break;
        }
    }
    datafusion::arrow::compute::kernels::cast_utils::string_to_timestamp_nanos(s).ok().map(|n| n / 1_000_000)
}

#[async_trait]
impl ImmutableFileTable for DuckLakeSource {
    async fn latest(&self) -> Result<Arc<dyn FileSnapshot>> {
        let snapshot_id = self.provider.get_current_snapshot().map_err(de)?;
        let snapshot = self.capture(snapshot_id)?.ok_or_else(|| {
            DataFusionError::Plan(format!("DuckLake table {}.{} not found at snapshot {snapshot_id}", self.schema, self.table))
        })?;
        Ok(Arc::new(snapshot))
    }

    /// The current snapshot, every snapshot taken at or after `since`, and
    /// the one that was current at `since`.
    async fn snapshots_since(&self, since: SystemTime) -> Result<Vec<Arc<dyn FileSnapshot>>> {
        let cutoff = since.duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0);
        let current = self.provider.get_current_snapshot().map_err(de)?;
        let mut snapshots: Vec<SnapshotMetadata> =
            self.provider.list_snapshots().map_err(de)?.into_iter().filter(|s| s.snapshot_id <= current).collect();
        snapshots.sort_by_key(|s| std::cmp::Reverse(s.snapshot_id));
        let mut out: Vec<Arc<dyn FileSnapshot>> = vec![self.latest().await?];
        for s in snapshots {
            if s.snapshot_id < current
                && let Some(snapshot) = self.capture(s.snapshot_id)?
            {
                out.push(Arc::new(snapshot));
            }
            // Unparsable times count as recent: keeping a snapshot is always safe.
            if s.timestamp.as_deref().and_then(snapshot_time).is_some_and(|t| t < cutoff) {
                break; // taken before the cutoff: current at the cutoff
            }
        }
        Ok(out)
    }
}

/// The catalog as a scan of some of one table's files sees it: the other
/// data files of that table, and its inlined rows unless kept, are not
/// listed. Everything else is the inner catalog's answer.
#[derive(Debug)]
struct FileSubset {
    inner: Arc<dyn MetadataProvider>,
    table_id: i64,
    keep: HashSet<i64>,
    inlined: bool,
}

impl FileSubset {
    fn ours(&self, table_id: i64) -> bool {
        table_id == self.table_id
    }

    fn no_inlined(&self, table_id: i64) -> bool {
        self.ours(table_id) && !self.inlined
    }

    /// Keyset-paged listing filtered to `keep`: keep fetching inner pages so
    /// a page is only short (or empty) when the inner listing is exhausted.
    fn page(
        &self,
        after: Option<i64>,
        limit: usize,
        mut fetch: impl FnMut(Option<i64>) -> datafusion_ducklake::Result<Vec<DuckLakeFileMetadata>>,
    ) -> datafusion_ducklake::Result<Vec<DuckLakeFileMetadata>> {
        let mut out = Vec::new();
        let mut cursor = after;
        while out.len() < limit {
            let page = fetch(cursor)?;
            let Some(last) = page.last() else { break };
            cursor = Some(last.file.data_file_id);
            let full = page.len() >= limit;
            out.extend(page.into_iter().filter(|m| self.keep.contains(&m.file.data_file_id)));
            if !full {
                break;
            }
        }
        out.truncate(limit);
        Ok(out)
    }
}

type DlResult<T> = datafusion_ducklake::Result<T>;

impl MetadataProvider for FileSubset {
    fn get_current_snapshot(&self) -> DlResult<i64> {
        self.inner.get_current_snapshot()
    }

    fn get_data_path(&self) -> DlResult<String> {
        self.inner.get_data_path()
    }

    fn get_metadata_settings(&self, schema_id: Option<i64>, table_id: Option<i64>) -> DlResult<HashMap<String, String>> {
        self.inner.get_metadata_settings(schema_id, table_id)
    }

    fn list_snapshots(&self) -> DlResult<Vec<SnapshotMetadata>> {
        self.inner.list_snapshots()
    }

    fn list_snapshot_changes(&self) -> DlResult<Vec<SnapshotChangeMetadata>> {
        self.inner.list_snapshot_changes()
    }

    fn find_snapshot_by_commit_extra_info(&self, needle: &str) -> DlResult<Option<i64>> {
        self.inner.find_snapshot_by_commit_extra_info(needle)
    }

    fn list_schemas(&self, snapshot_id: i64) -> DlResult<Vec<SchemaMetadata>> {
        self.inner.list_schemas(snapshot_id)
    }

    fn list_tables(&self, schema_id: i64, snapshot_id: i64) -> DlResult<Vec<TableMetadata>> {
        self.inner.list_tables(schema_id, snapshot_id)
    }

    fn list_views(&self, schema_id: i64, snapshot_id: i64) -> DlResult<Vec<ViewMetadata>> {
        self.inner.list_views(schema_id, snapshot_id)
    }

    fn get_table_structure(&self, table_id: i64, snapshot_id: i64) -> DlResult<Vec<DuckLakeTableColumn>> {
        self.inner.get_table_structure(table_id, snapshot_id)
    }

    fn get_table_fields(&self, table_id: i64, snapshot_id: i64) -> DlResult<Vec<DuckLakeTableField>> {
        self.inner.get_table_fields(table_id, snapshot_id)
    }

    fn get_name_mapping(&self, mapping_id: i64) -> DlResult<DuckLakeNameMapping> {
        self.inner.get_name_mapping(mapping_id)
    }

    fn get_table_files_for_select(&self, table_id: i64, snapshot_id: i64) -> DlResult<Vec<DuckLakeTableFile>> {
        let mut files = self.inner.get_table_files_for_select(table_id, snapshot_id)?;
        if self.ours(table_id) {
            files.retain(|f| self.keep.contains(&f.data_file_id));
        }
        Ok(files)
    }

    fn get_partition_spec(&self, table_id: i64, snapshot_id: i64) -> DlResult<Option<PartitionSpec>> {
        self.inner.get_partition_spec(table_id, snapshot_id)
    }

    fn get_sort_spec(&self, table_id: i64, snapshot_id: i64) -> DlResult<Option<SortSpec>> {
        self.inner.get_sort_spec(table_id, snapshot_id)
    }

    fn get_table_statistics(&self, table_id: i64, snapshot_id: i64) -> DlResult<DuckLakeStatistics> {
        if self.ours(table_id) {
            // Table-level counts describe all files: unknown for a subset.
            let mut files = self.inner.get_table_statistics(table_id, snapshot_id)?.files;
            files.retain(|s| self.keep.contains(&s.data_file_id));
            return Ok(DuckLakeStatistics { files, ..Default::default() });
        }
        self.inner.get_table_statistics(table_id, snapshot_id)
    }

    fn get_table_summary_statistics(&self, table_id: i64, snapshot_id: i64) -> DlResult<DuckLakeStatistics> {
        if self.ours(table_id) {
            return Ok(DuckLakeStatistics::default());
        }
        self.inner.get_table_summary_statistics(table_id, snapshot_id)
    }

    fn get_table_file_metadata_page(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
    ) -> DlResult<Vec<DuckLakeFileMetadata>> {
        if !self.ours(table_id) {
            return self.inner.get_table_file_metadata_page(table_id, snapshot_id, after_data_file_id, limit);
        }
        self.page(after_data_file_id, limit, |after| {
            self.inner.get_table_file_metadata_page(table_id, snapshot_id, after, limit)
        })
    }

    fn get_table_file_metadata_page_filtered(
        &self,
        table_id: i64,
        snapshot_id: i64,
        after_data_file_id: Option<i64>,
        limit: usize,
        filter: Option<&StatsFilter>,
    ) -> DlResult<Vec<DuckLakeFileMetadata>> {
        if !self.ours(table_id) {
            return self.inner.get_table_file_metadata_page_filtered(table_id, snapshot_id, after_data_file_id, limit, filter);
        }
        self.page(after_data_file_id, limit, |after| {
            self.inner.get_table_file_metadata_page_filtered(table_id, snapshot_id, after, limit, filter)
        })
    }

    fn get_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> DlResult<Vec<datafusion::arrow::record_batch::RecordBatch>> {
        if self.no_inlined(table_id) {
            return Ok(Vec::new());
        }
        self.inner.get_inlined_data(table_id, snapshot_id, columns)
    }

    fn scan_inlined_data(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
        filter: Option<&InlinedFilter>,
    ) -> DlResult<InlinedDataScan> {
        if self.no_inlined(table_id) {
            return Ok(InlinedDataScan::default());
        }
        self.inner.scan_inlined_data(table_id, snapshot_id, columns, filter)
    }

    fn get_inlined_deletes(&self, table_id: i64, snapshot_id: i64) -> DlResult<Vec<DuckLakeInlinedDelete>> {
        self.inner.get_inlined_deletes(table_id, snapshot_id)
    }

    fn get_inlined_data_with_row_ids(
        &self,
        table_id: i64,
        snapshot_id: i64,
        columns: &[DuckLakeTableColumn],
    ) -> DlResult<Vec<DuckLakeInlinedData>> {
        if self.no_inlined(table_id) {
            return Ok(Vec::new());
        }
        self.inner.get_inlined_data_with_row_ids(table_id, snapshot_id, columns)
    }

    // get_table_row_count: the trait default, which counts from the
    // (filtered) file listing and inlined reads above.

    fn get_schema_by_name(&self, name: &str, snapshot_id: i64) -> DlResult<Option<SchemaMetadata>> {
        self.inner.get_schema_by_name(name, snapshot_id)
    }

    fn get_table_by_name(&self, schema_id: i64, name: &str, snapshot_id: i64) -> DlResult<Option<TableMetadata>> {
        self.inner.get_table_by_name(schema_id, name, snapshot_id)
    }

    fn get_view_by_name(&self, schema_id: i64, name: &str, snapshot_id: i64) -> DlResult<Option<ViewMetadata>> {
        self.inner.get_view_by_name(schema_id, name, snapshot_id)
    }

    fn table_exists(&self, schema_id: i64, name: &str, snapshot_id: i64) -> DlResult<bool> {
        self.inner.table_exists(schema_id, name, snapshot_id)
    }

    fn get_tags(&self, target: TagTarget, snapshot_id: i64) -> DlResult<Vec<DuckLakeTag>> {
        self.inner.get_tags(target, snapshot_id)
    }

    fn get_view_id_by_name(&self, schema_id: i64, name: &str, snapshot_id: i64) -> DlResult<Option<i64>> {
        self.inner.get_view_id_by_name(schema_id, name, snapshot_id)
    }

    fn list_all_tables(&self, snapshot_id: i64) -> DlResult<Vec<TableWithSchema>> {
        self.inner.list_all_tables(snapshot_id)
    }

    fn list_all_views(&self, snapshot_id: i64) -> DlResult<Vec<ViewWithSchema>> {
        self.inner.list_all_views(snapshot_id)
    }

    fn list_all_columns(&self, snapshot_id: i64) -> DlResult<Vec<ColumnWithTable>> {
        self.inner.list_all_columns(snapshot_id)
    }

    fn list_all_object_tags(&self, snapshot_id: i64) -> DlResult<Vec<ObjectTag>> {
        self.inner.list_all_object_tags(snapshot_id)
    }

    fn list_all_column_tags(&self, snapshot_id: i64) -> DlResult<Vec<ColumnTag>> {
        self.inner.list_all_column_tags(snapshot_id)
    }

    fn list_all_files(&self, snapshot_id: i64) -> DlResult<Vec<FileWithTable>> {
        self.inner.list_all_files(snapshot_id)
    }

    fn get_data_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> DlResult<Vec<DataFileChange>> {
        self.inner.get_data_files_added_between_snapshots(table_id, start_snapshot, end_snapshot)
    }

    fn get_delete_files_added_between_snapshots(
        &self,
        table_id: i64,
        start_snapshot: i64,
        end_snapshot: i64,
    ) -> DlResult<Vec<DeleteFileChange>> {
        self.inner.get_delete_files_added_between_snapshots(table_id, start_snapshot, end_snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::snapshot_time;

    #[test]
    fn parses_catalog_timestamps() {
        let t = snapshot_time("2026-11-24 10:00:00.123+00").unwrap();
        assert_eq!(t, snapshot_time("2026-11-24T10:00:00.123Z").unwrap());
        assert_eq!(t % 1000, 123);
        assert_eq!(snapshot_time("2026-11-24 10:00:00"), Some(t - 123));
        assert_eq!(snapshot_time("not a time"), None);
    }
}
