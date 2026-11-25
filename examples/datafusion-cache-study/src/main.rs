//! Demo and benchmark binary for the `datafusion-subplan-cache` library.

mod delta;
mod example;
mod location;
mod timeline;

use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::Result;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::*;
use datafusion_subplan_cache::{PersistentStore, RewriteReport, SubplanCache};
use tokio::io::{AsyncBufReadExt, BufReader};

use delta::DeltaTables;

fn sales_dir() -> String {
    format!("{}/sales_delta", location::data_dir())
}
const SALES_COMMITS: u64 = 20;
const ROWS_PER_COMMIT: u64 = 1_500_000;
const CUSTOMERS: u64 = 2_000_000;

/// A SessionContext with a subplan cache installed, plus the demo's
/// reporting around each query.
pub struct CachingContext {
    ctx: SessionContext,
    cache: SubplanCache,
    delta: DeltaTables,
    pub show_plan: bool,
}

pub struct QueryOutcome {
    batches: Vec<RecordBatch>,
    report: RewriteReport,
    events: Vec<String>,
    elapsed: Duration,
    physical_plan: Option<String>,
}

impl CachingContext {
    pub fn new() -> Self {
        Self::with_store(None)
    }

    /// A context whose cache also persists to (and reads from) `store`.
    pub fn with_store(store: Option<Arc<PersistentStore>>) -> Self {
        let mut builder = SubplanCache::builder();
        if let Some(store) = store {
            builder = builder.with_store(store);
        }
        let cache = builder.build();
        Self { ctx: cache.session_context(), cache, delta: DeltaTables::default(), show_plan: false }
    }

    pub async fn sql(&mut self, sql: &str) -> Result<QueryOutcome> {
        let start = Instant::now();
        let loads_before = self.cache.loads();
        // Drop stats of queries run outside sql() (e.g. by append's max()).
        self.cache.take_reports();
        self.cache.take_segment_uses();
        // Pick up commits by other writers.
        self.cache.refresh(&self.ctx).await?;
        let df = self.ctx.sql(sql).await?;
        let physical = df.create_physical_plan().await?;
        let physical_plan = self.show_plan.then(|| displayable(physical.as_ref()).indent(false).to_string());
        let batches = collect(physical, self.ctx.task_ctx()).await?;
        // New entries are uploaded in the background while the query runs;
        // wait for them so a restart right after this query finds them.
        self.cache.flush().await?;
        let mut report = RewriteReport::default();
        for r in self.cache.take_reports() {
            report.hits.extend(r.hits);
            report.writes.extend(r.writes);
            report.unregistered.extend(r.unregistered);
        }
        let mut events = self.cache.take_events();
        let loaded = self.cache.loads() - loads_before;
        if loaded > 0 {
            events.push(format!("  ⇣ read {loaded} cached result(s) from the persistent store"));
        }
        Ok(QueryOutcome { batches, report, events, elapsed: start.elapsed(), physical_plan })
    }

    /// Run without any caching (reference result and timing).
    pub async fn sql_uncached(&mut self, sql: &str) -> Result<(Vec<RecordBatch>, Duration)> {
        self.cache.refresh(&self.ctx).await?;
        let was = self.cache.enabled();
        self.cache.set_enabled(false);
        let start = Instant::now();
        let res = async { self.ctx.sql(sql).await?.collect().await }.await;
        self.cache.set_enabled(was);
        self.cache.take_events();
        Ok((res?, start.elapsed()))
    }

    pub fn print_cache(&self) {
        let entries = self.cache.entries();
        if entries.is_empty() {
            println!("(cache is empty)");
            return;
        }
        println!("{} entries, {:.1} MiB total", entries.len(), self.cache.total_bytes() as f64 / (1 << 20) as f64);
        if let Some(store) = self.cache.store() {
            println!("persistent store: {}", store.location);
        }
        for e in entries {
            let mem = if e.data.get().is_some() { "mem" } else { "   " };
            let disk = if e.persisted.load(std::sync::atomic::Ordering::Relaxed) { "store" } else { "     " };
            println!(
                "  {}  {mem} {disk}  rows={:<8} {:>7.2} MiB  hits={:<3} built in {:>7.1?}  {}",
                e.id,
                e.rows,
                e.bytes as f64 / (1 << 20) as f64,
                e.hits.load(std::sync::atomic::Ordering::Relaxed),
                e.produced_in,
                e.label
            );
        }
    }

    pub fn print_files(&self) {
        let Some(s) = self.cache.snapshot("sales") else { return };
        println!("sales v{}: {} files (fileset {})", s.version(), s.files().len(), s.fileset);
        for f in s.files() {
            println!("  {}", f.path);
        }
    }

    fn sales_version(&self) -> String {
        self.cache.snapshot("sales").map(|s| s.version()).unwrap_or_default()
    }

    async fn append(&mut self, rows: u64) -> Result<()> {
        let next: i64 = {
            let b = self.ctx.sql("SELECT max(order_id) FROM sales").await?.collect().await?;
            datafusion::arrow::array::cast::as_primitive_array::<datafusion::arrow::datatypes::Int64Type>(b[0].column(0)).value(0) + 1
        };
        let batches = self.ctx.sql(&sales_sql(next as u64, next as u64 + rows)).await?.collect().await?;
        self.delta.append(&self.cache, &self.ctx, "sales", batches).await?;
        println!("  appended {rows} rows → sales v{}", self.sales_version());
        Ok(())
    }

    async fn delete(&mut self, predicate: &str) -> Result<()> {
        let m = self.delta.delete(&self.cache, &self.ctx, "sales", predicate).await?;
        println!("  DELETE WHERE {predicate}: {m} → sales v{}", self.sales_version());
        Ok(())
    }

    /// Delete persisted entries that no snapshot committed in the last
    /// `grace` uses (readers may lag behind by up to that much).
    pub async fn purge(&mut self, grace: Duration) -> Result<()> {
        self.cache.refresh(&self.ctx).await?;
        let n = self.cache.purge(grace).await?;
        println!("deleted {n} persisted entries (grace {} min)", grace.as_secs() / 60);
        Ok(())
    }

    async fn optimize(&mut self) -> Result<()> {
        let (removed, added) = self.delta.optimize(&self.cache, &self.ctx, "sales", None).await?;
        println!("  OPTIMIZE: {removed} files removed, {added} files added → sales v{}", self.sales_version());
        Ok(())
    }
}

fn print_outcome(out: &QueryOutcome, max_rows: usize) {
    let rows: usize = out.batches.iter().map(|b| b.num_rows()).sum();
    if let Some(p) = &out.physical_plan {
        println!("{p}");
    }
    if !out.batches.is_empty() {
        let mut shown = Vec::new();
        let mut left = max_rows;
        for b in &out.batches {
            if left == 0 {
                break;
            }
            let n = b.num_rows().min(left);
            shown.push(b.slice(0, n));
            left -= n;
        }
        println!("{}", pretty_format_batches(&shown).unwrap());
        if rows > max_rows {
            println!("... {} more rows", rows - max_rows);
        }
    }
    for e in &out.events {
        println!("{e}");
    }
    for e in &out.report.hits {
        println!("  ✔ cache HIT   {}  ({} rows, originally took {:.1?})  {}", e.id, e.rows, e.produced_in, e.label);
    }
    for (id, lbl) in &out.report.writes {
        println!("  ✎ caching     {id}  {lbl}");
    }
    println!("  ⏱  {:.1?}", out.elapsed);
}

/// Rows of the sales fact table for order ids in [start, end).
fn sales_sql(start: u64, end: u64) -> String {
    format!(
        "SELECT value AS order_id,
                (value * 7919) % {CUSTOMERS} AS customer_id,
                value % 1000 AS product_id,
                CAST(CAST((value * 31) % 100000 AS DECIMAL(12,2)) / 100 AS DECIMAL(12,2)) AS amount,
                2020 + CAST((value / 7) % 5 AS INT) AS year
         FROM range({start}, {end})"
    )
}

async fn ensure_data(cc: &mut CachingContext) -> Result<()> {
    let data = location::data_dir();
    if !location::is_remote(data) {
        std::fs::create_dir_all(data)?;
    }
    location::register(&cc.ctx)?;
    let customers = format!("{data}/customers.parquet");
    if !location::exists(&customers).await? {
        println!("Generating {CUSTOMERS} customers …");
        cc.ctx
            .sql(&format!(
                "COPY (SELECT value AS customer_id,
                         CASE (value / 3) % 5 WHEN 0 THEN 'NA' WHEN 1 THEN 'EU' WHEN 2 THEN 'APAC'
                                              WHEN 3 THEN 'LATAM' ELSE 'MEA' END AS region,
                         CASE value % 3 WHEN 0 THEN 'consumer' WHEN 1 THEN 'smb' ELSE 'enterprise' END AS segment
                       FROM range({CUSTOMERS}))
                 TO '{customers}' STORED AS PARQUET"
            ))
            .await?
            .collect()
            .await?;
    }
    let sales = sales_dir();
    if !location::exists(&format!("{sales}/_delta_log/00000000000000000000.json")).await? {
        println!("Creating Delta table sales: {SALES_COMMITS} commits × {ROWS_PER_COMMIT} rows …");
        let mut commits = vec![];
        for i in 0..SALES_COMMITS {
            let sql = sales_sql(i * ROWS_PER_COMMIT, (i + 1) * ROWS_PER_COMMIT);
            commits.push(cc.ctx.sql(&sql).await?.collect().await?);
        }
        delta::create_with_commits(&sales, commits).await?;
    }
    // customers never changes: register it as static so joins with it stay
    // cacheable. sales is a Delta table the cache follows file by file.
    cc.ctx.register_parquet("customers", &customers, ParquetReadOptions::default()).await?;
    cc.cache.mark_static(&cc.ctx, "customers").await?;
    cc.delta.open(&cc.cache, &cc.ctx, "sales", &sales).await?;
    Ok(())
}

const Q1: &str = "SELECT c.region, sum(s.amount) AS revenue, count(*) AS orders
     FROM sales s JOIN customers c ON s.customer_id = c.customer_id
     WHERE s.year = 2024
     GROUP BY c.region
     ORDER BY revenue DESC";

const Q2: &str = "SELECT c.region, sum(s.amount) AS revenue, count(*) AS orders
     FROM sales s JOIN customers c ON s.customer_id = c.customer_id
     WHERE s.year = 2024
     GROUP BY c.region
     HAVING count(*) > 1000
     ORDER BY c.region
     LIMIT 3";

const Q3: &str = "SELECT c.segment, avg(s.amount) AS avg_amount, max(s.amount) AS biggest, count(DISTINCT s.product_id) AS products
     FROM sales s JOIN customers c ON s.customer_id = c.customer_id
     GROUP BY c.segment
     ORDER BY c.segment";

enum Step {
    Query(&'static str, &'static str),
    Append(u64),
    Delete(&'static str),
    Optimize,
}

const DEMO: &[Step] = &[
    Step::Query("Q1 cold: every file computed, partial states cached per segment", Q1),
    Step::Query("Q1 again: exact repeat, same snapshot", Q1),
    Step::Query("Q2: same aggregate, different HAVING/ORDER/LIMIT", Q2),
    Step::Query("Q3 cold: a different aggregate (with count distinct)", Q3),
    Step::Append(ROWS_PER_COMMIT),
    Step::Query("Q1 after append: old segments reused, only the new file computed", Q1),
    Step::Query("Q3 after append", Q3),
    Step::Delete("order_id >= 4500000 AND order_id < 4500100"),
    Step::Query("Q1 after DELETE: only the segment holding the rewritten file recomputed", Q1),
    Step::Query("Q3 after DELETE", Q3),
    Step::Optimize,
    Step::Query("Q1 after OPTIMIZE: same rows, all-new files → everything recomputed", Q1),
    Step::Query("Q1 again after OPTIMIZE", Q1),
];

async fn run_demo(cc: &mut CachingContext) -> Result<()> {
    for step in DEMO {
        match step {
            Step::Append(n) => {
                println!("━━ APPEND");
                cc.append(*n).await?
            }
            Step::Delete(p) => {
                println!("━━ DELETE");
                cc.delete(p).await?
            }
            Step::Optimize => {
                println!("━━ OPTIMIZE");
                cc.optimize().await?
            }
            Step::Query(title, sql) => {
                println!("━━ {title}");
                let out = cc.sql(sql).await?;
                print_outcome(&out, 5);
                let (reference, t) = cc.sql_uncached(sql).await?;
                let same = pretty_format_batches(&out.batches)?.to_string() == pretty_format_batches(&reference)?.to_string();
                println!("  {} uncached run took {:.1?}\n", if same { "✓ identical to uncached result;" } else { "✗ DIFFERS from uncached result!" }, t);
                if !same {
                    println!("{}", pretty_format_batches(&reference)?);
                }
            }
        }
    }
    println!("━━ cache contents");
    cc.print_cache();
    Ok(())
}

async fn repl(cc: &mut CachingContext) -> Result<()> {
    println!("Tables: sales (Delta: order_id, customer_id, product_id, amount, year), customers (parquet: customer_id, region, segment)");
    println!("End statements with ';'.");
    println!("  \\cache  \\clear  \\files  \\plan  \\off / \\on  \\segsize N  \\demo  \\q");
    println!("  \\append [rows]   \\delete <predicate>   \\optimize");
    println!("  \\purge [minutes]  delete persisted cache entries no snapshot of the last N minutes (default 60) uses\n");
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut buf = String::new();
    loop {
        use std::io::Write;
        print!("{}", if buf.is_empty() { "sql> " } else { "...> " });
        std::io::stdout().flush().ok();
        let Some(line) = lines.next_line().await? else { break };
        let t = line.trim();
        if buf.is_empty() && t.starts_with('\\') {
            let (cmd, arg) = t.split_once(' ').unwrap_or((t, ""));
            let res = match cmd {
                "\\q" => break,
                "\\cache" => Ok(cc.print_cache()),
                "\\clear" => Ok(cc.cache.clear()),
                "\\files" => Ok(cc.print_files()),
                "\\plan" => {
                    cc.show_plan = !cc.show_plan;
                    Ok(println!("show plan: {}", cc.show_plan))
                }
                "\\off" => Ok(cc.cache.set_enabled(false)),
                "\\on" => Ok(cc.cache.set_enabled(true)),
                "\\segsize" => Ok(cc.cache.set_segment_files(arg.parse().unwrap_or(4))),
                "\\append" => cc.append(arg.parse().unwrap_or(ROWS_PER_COMMIT)).await,
                "\\delete" => cc.delete(arg).await,
                "\\optimize" => cc.optimize().await,
                "\\demo" => run_demo(cc).await,
                "\\purge" => cc.purge(Duration::from_secs(60 * arg.parse::<u64>().unwrap_or(60))).await,
                _ => Ok(println!("unknown command")),
            };
            if let Err(e) = res {
                println!("error: {e}");
            }
            continue;
        }
        buf.push_str(&line);
        buf.push('\n');
        if t.ends_with(';') {
            let sql = std::mem::take(&mut buf);
            match cc.sql(&sql).await {
                Ok(out) => {
                    print_outcome(&out, 50);
                    println!();
                }
                Err(e) => println!("error: {e}\n"),
            }
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let store = match flag("--cache-dir") {
        Some(location) => Some(Arc::new(PersistentStore::open(&location)?)),
        None => None,
    };
    if let Some(d) = flag("--data-dir") {
        location::set_data_dir(&d);
    }
    println!("Data: {}", location::data_dir());
    let mut cc = CachingContext::with_store(store);
    if let Some(n) = args.iter().position(|a| a == "--segsize").and_then(|i| args.get(i + 1)) {
        cc.cache.set_segment_files(n.parse().unwrap_or(4));
    }
    if args.get(1).map(String::as_str) == Some("example") {
        return example::run().await;
    }
    if matches!(args.get(1).map(String::as_str), Some("demo") | Some("timeline")) {
        // The demo mutates the table; always start from the original 20 commits.
        location::remove_all(&sales_dir()).await?;
    }
    ensure_data(&mut cc).await?;
    match args.get(1).map(String::as_str) {
        Some("demo") => run_demo(&mut cc).await,
        Some("timeline") => timeline::run(&mut cc, args.get(2).map(String::as_str).filter(|a| !a.starts_with("--")), flag("--cache-dir")).await,
        _ => repl(&mut cc).await,
    }
}
