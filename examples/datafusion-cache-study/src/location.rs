//! Where the tables live: a local directory (default `data`) or, with the
//! `gcs` / `s3` features, a bucket prefix such as `gs://bucket/prefix`.
//!
//!     cargo run --release --features gcs -- --data-dir gs://bucket/prefix …

use std::sync::{Arc, OnceLock};

use datafusion::common::{DataFusionError, Result};
use datafusion::prelude::SessionContext;
use futures::{StreamExt, TryStreamExt};
use object_store::{ObjectStore, ObjectStoreExt};
use object_store::path::Path;
use url::Url;

static DATA: OnceLock<String> = OnceLock::new();

pub fn set_data_dir(loc: &str) {
    let _ = DATA.set(loc.trim_end_matches('/').to_string());
    #[cfg(feature = "gcs")]
    deltalake::gcp::register_handlers(None);
}

pub fn data_dir() -> &'static str {
    DATA.get().map(String::as_str).unwrap_or("data")
}

pub fn is_remote(loc: &str) -> bool {
    loc.contains("://")
}

fn oe(e: object_store::Error) -> DataFusionError {
    DataFusionError::ObjectStore(Box::new(e))
}

/// The object store holding `loc` and the path of `loc` inside it.
fn remote(loc: &str) -> Result<(Arc<dyn ObjectStore>, Path, Url)> {
    let url = Url::parse(loc).map_err(|e| DataFusionError::Plan(format!("bad location {loc}: {e}")))?;
    let env = std::env::vars().map(|(k, v)| (k.to_ascii_lowercase(), v));
    let (store, path) = object_store::parse_url_opts(&url, env).map_err(oe)?;
    Ok((Arc::from(store), path, url))
}

/// Let DataFusion read and write the data location (Delta registers its own
/// store; plain Parquet files need this).
pub fn register(ctx: &SessionContext) -> Result<()> {
    let loc = data_dir();
    if is_remote(loc) {
        let (store, _, url) = remote(loc)?;
        let root = Url::parse(&format!("{}://{}", url.scheme(), url.host_str().unwrap_or_default())).unwrap();
        ctx.register_object_store(&root, store);
    }
    Ok(())
}

/// Does `loc` (a file) exist?
pub async fn exists(loc: &str) -> Result<bool> {
    if !is_remote(loc) {
        return Ok(std::path::Path::new(loc).exists());
    }
    let (store, path, _) = remote(loc)?;
    match store.head(&path).await {
        Ok(_) => Ok(true),
        Err(object_store::Error::NotFound { .. }) => Ok(false),
        Err(e) => Err(oe(e)),
    }
}

/// Delete everything under `loc` (a directory or prefix).
pub async fn remove_all(loc: &str) -> Result<()> {
    if !is_remote(loc) {
        let _ = std::fs::remove_dir_all(loc);
        return Ok(());
    }
    let (store, path, _) = remote(loc)?;
    let paths = store.list(Some(&path)).map_ok(|m| m.location).boxed();
    store.delete_stream(paths).try_collect::<Vec<_>>().await.map_err(oe)?;
    Ok(())
}
