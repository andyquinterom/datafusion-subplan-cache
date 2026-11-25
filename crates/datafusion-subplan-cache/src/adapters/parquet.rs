//! Plain Parquet files with unique, never-rewritten names.
//!
//! Two ways to say which files make up the table:
//! * [`ParquetFiles::open`]: every `*.parquet` file under a directory or
//!   bucket prefix, re-listed on each refresh (an append-only folder, a
//!   stream sink, …);
//! * [`ParquetFiles::from_files`] / [`ParquetFiles::set_files`]: an explicit
//!   list the caller gets from its own catalog, e.g. the data files of an
//!   Iceberg snapshot (DuckLake has its own adapter, `DuckLakeSource`).
//!
//! This adapter applies no row-level deletes. A table format with delete
//! files (Iceberg merge-on-read) needs its own
//! [`crate::ImmutableFileTable`] whose file ids include the delete files that
//! apply to each data file.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::catalog::TableProvider;
use datafusion::catalog::empty::EmptyTable;
use datafusion::arrow::array::Array;
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::prelude::SessionContext;
use datafusion::parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use datafusion::parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
#[allow(deprecated)] // simplest footer reader over an ObjectStore; DataFusion uses it too
use datafusion::parquet::arrow::async_reader::ParquetObjectReader;
use datafusion::parquet::file::metadata::ParquetMetaData;
use futures::{StreamExt, TryStreamExt};
use object_store::ObjectStore;
use object_store::path::Path;

use crate::store::stable_hash;
use crate::table::{ColumnStats, DataFile, FileSnapshot, ImmutableFileTable};

enum Files {
    /// Every `*.parquet` under this prefix of the store.
    Listed(Path),
    /// Exactly these (object path, identity) pairs.
    Fixed(Vec<(String, String)>),
}

/// A table made of Parquet files that are never modified once written.
pub struct ParquetFiles {
    store: Arc<dyn ObjectStore>,
    store_url: ObjectStoreUrl,
    schema: SchemaRef,
    options: ListingOptions,
    files: Mutex<Files>,
    /// Column statistics per file id, read from each file's footer once.
    stats: Mutex<HashMap<String, Arc<HashMap<String, ColumnStats>>>>,
    collect_stats: bool,
}

fn options() -> ListingOptions {
    ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet")
}

impl ParquetFiles {
    /// Every `*.parquet` file under `url`: a local directory (`file:///…/`
    /// or a plain path) or a prefix in an object store registered with
    /// `ctx` (`gs://bucket/prefix/`, `s3://…`). The schema is inferred from
    /// the files present now.
    pub async fn open(ctx: &SessionContext, url: &str) -> Result<Self> {
        let table_url = ListingTableUrl::parse(url)?;
        let store = ctx.runtime_env().object_store(&table_url)?;
        let options = options();
        let schema = options.infer_schema(&ctx.state(), &table_url).await?;
        Ok(Self {
            store,
            store_url: table_url.object_store(),
            schema,
            options,
            files: Mutex::new(Files::Listed(table_url.prefix().clone())),
            stats: Default::default(),
            collect_stats: true,
        })
    }

    /// Exactly the given files, as URLs or paths in the store of `root`
    /// (e.g. root `gs://bucket/`, file `gs://bucket/t/data/0001.parquet` or
    /// `t/data/0001.parquet`), each with its size in bytes. Names must be
    /// unique for the lifetime of the table. The schema is inferred from the
    /// first file.
    pub async fn from_files(ctx: &SessionContext, root: &str, files: Vec<(String, u64)>) -> Result<Self> {
        let root_url = ListingTableUrl::parse(root)?;
        let store_url = root_url.object_store();
        let store = ctx.runtime_env().object_store(&root_url)?;
        let files = Self::fixed(&store_url, files);
        let first = files
            .first()
            .ok_or_else(|| DataFusionError::Plan("ParquetFiles::from_files: no files to infer the schema from".into()))?;
        let options = options();
        let schema = options.infer_schema(&ctx.state(), &file_url(&store_url, &first.0)?).await?;
        Ok(Self {
            store,
            store_url,
            schema,
            options,
            files: Mutex::new(Files::Fixed(files)),
            stats: Default::default(),
            collect_stats: true,
        })
    }

    /// Replace the file list (for tables made with [`Self::from_files`]):
    /// the next [`crate::SubplanCache::refresh`] picks it up.
    pub fn set_files(&self, files: Vec<(String, u64)>) {
        *self.files.lock().unwrap() = Files::Fixed(Self::fixed(&self.store_url, files));
    }

    fn fixed(store_url: &ObjectStoreUrl, files: Vec<(String, u64)>) -> Vec<(String, String)> {
        let base = store_url.as_str();
        files
            .into_iter()
            .map(|(p, size)| {
                let path = p.strip_prefix(base).unwrap_or(&p).trim_start_matches('/').to_string();
                (path, size.to_string())
            })
            .collect()
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Whether to read each new file's footer for column statistics
    /// (default on). They let the cache decide filters like
    /// `ts >= now() - interval '1 hour'` per file; each footer is read once.
    pub fn with_column_stats(mut self, on: bool) -> Self {
        self.collect_stats = on;
        self
    }

    /// Exact statistics of every column of the file, from its footer.
    #[allow(deprecated)]
    async fn read_stats(&self, path: &str, size: u64) -> Result<HashMap<String, ColumnStats>> {
        let mut reader = ParquetObjectReader::new(self.store.clone(), Path::from(path)).with_file_size(size);
        let meta = ArrowReaderMetadata::load_async(&mut reader, ArrowReaderOptions::new()).await?;
        Ok(file_stats(meta.schema(), meta.metadata(), &self.schema))
    }

    /// `files` with their statistics attached (read concurrently, memoized).
    async fn attach_stats(&self, files: Vec<(DataFile, u64)>) -> Result<Vec<DataFile>> {
        if !self.collect_stats {
            return Ok(files.into_iter().map(|(f, _)| f).collect());
        }
        let known = self.stats.lock().unwrap().clone();
        let out: Vec<DataFile> = futures::stream::iter(files)
            .map(|(f, size)| {
                let known = known.get(&f.id).cloned();
                async move {
                    let stats = match known {
                        Some(s) => s,
                        // Unreadable footers only cost the file its stats.
                        None => Arc::new(self.read_stats(&f.path, size).await.unwrap_or_default()),
                    };
                    Ok::<_, DataFusionError>(f.with_all_stats(stats))
                }
            })
            .buffered(16)
            .try_collect()
            .await?;
        let mut memo = self.stats.lock().unwrap();
        memo.retain(|id, _| out.iter().any(|f| &f.id == id));
        for f in &out {
            memo.entry(f.id.clone()).or_insert_with(|| f.stats.clone());
        }
        Ok(out)
    }
}

/// Fold the row-group statistics of one file into exact per-column stats,
/// typed as the table schema says. Columns whose stats are missing, inexact
/// (e.g. truncated strings) or floating point are left out.
fn file_stats(file_schema: &Schema, meta: &ParquetMetaData, table_schema: &Schema) -> HashMap<String, ColumnStats> {
    let mut out = HashMap::new();
    let groups = meta.row_groups();
    let row_count: u64 = groups.iter().map(|g| g.num_rows() as u64).sum();
    for field in table_schema.fields() {
        let ty = field.data_type();
        if ty.is_floating() || ty.is_nested() {
            continue;
        }
        let Some(stats) = (|| -> Option<ColumnStats> {
            let conv = StatisticsConverter::try_new(field.name(), file_schema, meta.file_metadata().schema_descr())
                .ok()?
                .with_missing_null_counts_as_zero(false);
            let mins = conv.row_group_mins(groups).ok()?;
            let maxes = conv.row_group_maxes(groups).ok()?;
            let min_exact = conv.row_group_is_min_value_exact(groups).ok()?;
            let max_exact = conv.row_group_is_max_value_exact(groups).ok()?;
            let nulls = conv.row_group_null_counts(groups).ok()?;
            let mut min: Option<ScalarValue> = None;
            let mut max: Option<ScalarValue> = None;
            let mut null_count = 0;
            for (i, g) in groups.iter().enumerate() {
                if nulls.is_null(i) {
                    return None;
                }
                let n = nulls.value(i);
                null_count += n;
                if n == g.num_rows() as u64 {
                    continue; // all null: no min/max to fold
                }
                if mins.is_null(i) || maxes.is_null(i) || !min_exact.value(i) || !max_exact.value(i) {
                    return None;
                }
                let lo = ScalarValue::try_from_array(&mins, i).ok()?;
                let hi = ScalarValue::try_from_array(&maxes, i).ok()?;
                if min.as_ref().is_none_or(|m| lo.partial_cmp(m).is_some_and(|o| o.is_lt())) {
                    min = Some(lo);
                }
                if max.as_ref().is_none_or(|m| hi.partial_cmp(m).is_some_and(|o| o.is_gt())) {
                    max = Some(hi);
                }
            }
            let null = ScalarValue::try_from(ty).ok()?;
            Some(ColumnStats {
                min: min.map_or(Some(null.clone()), |v| v.cast_to(ty).ok())?,
                max: max.map_or(Some(null), |v| v.cast_to(ty).ok())?,
                null_count,
                row_count,
            })
        })() else {
            continue;
        };
        out.insert(field.name().clone(), stats);
    }
    out
}

fn file_url(store_url: &ObjectStoreUrl, path: &str) -> Result<ListingTableUrl> {
    ListingTableUrl::parse(format!("{}{}", store_url.as_str(), path))
}

#[async_trait]
impl ImmutableFileTable for ParquetFiles {
    async fn latest(&self) -> Result<Arc<dyn FileSnapshot>> {
        let listed = match &*self.files.lock().unwrap() {
            Files::Fixed(f) => Ok(f.clone()),
            Files::Listed(prefix) => Err(prefix.clone()),
        };
        let pairs = match listed {
            Ok(f) => f,
            Err(prefix) => {
                let metas: Vec<_> = self
                    .store
                    .list(Some(&prefix))
                    .try_collect()
                    .await
                    .map_err(|e| DataFusionError::ObjectStore(Box::new(e)))?;
                metas
                    .into_iter()
                    .filter(|m| {
                        let name = m.location.filename().unwrap_or_default();
                        name.ends_with(".parquet") && !name.starts_with(['.', '_'])
                    })
                    .map(|m| {
                        let version = m.e_tag.clone().unwrap_or_else(|| m.last_modified.timestamp_millis().to_string());
                        (m.location.to_string(), format!("{}|{version}", m.size))
                    })
                    .collect()
            }
        };
        // The identity starts with the size in both cases.
        let sized = pairs.iter().map(|(p, identity)| {
            let size = identity.split('|').next().and_then(|s| s.parse().ok()).unwrap_or(0);
            (DataFile::new(p.clone(), identity), size)
        });
        let mut files = self.attach_stats(sized.collect()).await?;
        files.sort_by(|a, b| a.path.cmp(&b.path));
        let ids: Vec<&str> = files.iter().map(|f| f.id.as_str()).collect();
        Ok(Arc::new(ParquetSnapshot {
            version: stable_hash(&ids.join(","))[..8].to_string(),
            files,
            store_url: self.store_url.clone(),
            schema: self.schema.clone(),
            options: self.options.clone(),
        }))
    }
}

/// A fixed set of Parquet files.
pub struct ParquetSnapshot {
    version: String,
    files: Vec<DataFile>,
    store_url: ObjectStoreUrl,
    schema: SchemaRef,
    options: ListingOptions,
}

impl std::fmt::Debug for ParquetSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParquetSnapshot({}, {} files)", self.version, self.files.len())
    }
}

impl ParquetSnapshot {
    fn table(&self, files: &[&DataFile]) -> Result<Arc<dyn TableProvider>> {
        if files.is_empty() {
            return Ok(Arc::new(EmptyTable::new(self.schema.clone())));
        }
        let urls = files.iter().map(|f| file_url(&self.store_url, &f.path)).collect::<Result<Vec<_>>>()?;
        let config = ListingTableConfig::new_with_multi_paths(urls)
            .with_listing_options(self.options.clone())
            .with_schema(self.schema.clone());
        Ok(Arc::new(ListingTable::try_new(config)?))
    }
}

#[async_trait]
impl FileSnapshot for ParquetSnapshot {
    fn version(&self) -> String {
        self.version.clone()
    }

    fn files(&self) -> &[DataFile] {
        &self.files
    }

    async fn provider(&self) -> Result<Arc<dyn TableProvider>> {
        self.table(&self.files.iter().collect::<Vec<_>>())
    }

    async fn provider_for(&self, files: &[&DataFile]) -> Result<Option<Arc<dyn TableProvider>>> {
        self.table(files).map(Some)
    }
}
