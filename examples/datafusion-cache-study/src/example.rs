//! A tiny, readable walkthrough of per-file aggregate caching on Delta.
//!
//!     cargo run --release -- example
//!
//! Table `orders(order_id, store, amount)` with one data file per commit.
//! Files are labelled A, B, C, … in the order they appear. Every step prints
//! which files each query read and which it took from the cache, the cached
//! partial aggregate states per file, and checks the result against an
//! uncached run.

use std::collections::HashMap;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::util::display::array_value_to_string;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::Result;

use datafusion_subplan_cache::SegmentUse;

use crate::CachingContext;
use crate::delta;

const DIR: &str = "data/example_orders";

const Q_ALL: (&str, &str) = (
    "Q_all  ",
    "SELECT store, sum(amount) AS total, count(*) AS orders FROM orders GROUP BY store ORDER BY store",
);
const Q_NORTH: (&str, &str) = (
    "Q_north",
    "SELECT store, sum(amount) AS total FROM orders WHERE store = 'north' GROUP BY store ORDER BY store",
);

struct Example {
    cc: CachingContext,
    /// file path → label
    labels: HashMap<String, String>,
    next: u8,
}

impl Example {
    /// Assign labels to files we have not seen yet; print the table state.
    async fn show_table(&mut self, change: &str) -> Result<()> {
        let snap = self.cc.cache.snapshot("orders").unwrap();
        let before: Vec<String> = self.labels.keys().cloned().collect();
        let mut new_paths: Vec<&String> =
            snap.files().iter().map(|f| &f.path).filter(|p| !self.labels.contains_key(*p)).collect();
        new_paths.sort();
        let mut added = vec![];
        for p in new_paths {
            let l = ((b'A' + self.next) as char).to_string();
            self.next += 1;
            self.labels.insert(p.clone(), l.clone());
            added.push(l);
        }
        let current: Vec<&String> = snap.files().iter().map(|f| &f.path).collect();
        let mut removed: Vec<String> =
            before.iter().filter(|p| !current.contains(p)).map(|p| self.labels[p].clone()).collect();
        removed.sort();
        // Keep removed labels around only to report them once.
        self.labels.retain(|p, _| current.contains(&p));

        println!("  {change}");
        println!(
            "  → orders v{}: files {}   (removed: {}, added: {})",
            snap.version(),
            self.sorted_labels(snap.files().iter().map(|f| f.path.as_str())).join(" "),
            if removed.is_empty() { "-".into() } else { removed.join(" ") },
            if added.is_empty() { "-".into() } else { added.join(" ") },
        );
        let mut files = snap.files().to_vec();
        files.sort_by_key(|f| self.labels[&f.path].clone());
        for f in &files {
            let provider = snap.inner.provider_for(&[f]).await?.expect("Delta scans can be restricted");
            let rows = self.cc.ctx.read_table(provider)?.sort(vec![datafusion::prelude::col("order_id").sort(true, false)])?.collect().await?;
            println!("       {}: {}", self.labels[&f.path], format_rows(&rows, ", ", "(", ")"));
        }
        Ok(())
    }

    fn sorted_labels<'a>(&self, paths: impl Iterator<Item = &'a str>) -> Vec<String> {
        let mut v: Vec<String> = paths.map(|p| self.labels[p].clone()).collect();
        v.sort();
        v
    }

    fn label_of_id(&self, id: &str) -> String {
        let snap = self.cc.cache.snapshot("orders").unwrap();
        snap.files().iter().find(|f| snap.file_dep(f) == id).map(|f| self.labels[&f.path].clone()).unwrap_or("?".into())
    }

    async fn query(&mut self, (name, sql): (&str, &str)) -> Result<()> {
        let out = self.cc.sql(sql).await?;
        let uses = self.cc.cache.take_segment_uses();
        let (reference, _) = self.cc.sql_uncached(sql).await?;
        let same = pretty_format_batches(&out.batches)?.to_string() == pretty_format_batches(&reference)?.to_string();

        println!("  {name}  {sql}");
        if !out.report.hits.is_empty() && uses.is_empty() {
            println!("           whole result reused from cache: no files read, nothing re-aggregated");
        }
        for u in &uses {
            self.print_segments(u);
        }
        println!(
            "           result: {}   {}",
            format_rows(&out.batches, "  ", "[", "]"),
            if same { "✓ same as uncached" } else { "✗ DIFFERS FROM UNCACHED" }
        );
        Ok(())
    }

    /// One line per file: was it read or reused, and its cached partial state.
    fn print_segments(&self, u: &SegmentUse) {
        let mut lines: Vec<(String, &str, String)> = vec![];
        for (kind, segs) in [("reused  ", &u.reused), ("computed", &u.computed)] {
            for deps in segs {
                let labels: Vec<String> = deps.iter().map(|d| self.label_of_id(d)).collect();
                let key = format!("{}\n-- segment: {}", u.base_key, deps.join(","));
                let state = match self.cc.cache.get(&key) {
                    Some(e) => {
                        let batches = e.data.get().map(|d| d.partitions.concat()).unwrap_or_default();
                        let rows = format_rows(&batches, "  ", "{", "}");
                        if rows.is_empty() { "{} no matching rows".to_string() } else { rows }
                    }
                    None => "(not stored)".to_string(),
                };
                lines.push((labels.join("+"), kind, state));
            }
        }
        lines.sort();
        for (label, kind, state) in lines {
            println!("           file {label}  {kind}  partial state: {state}");
        }
    }
}

fn format_rows(batches: &[RecordBatch], sep: &str, open: &str, close: &str) -> String {
    let mut out = vec![];
    for b in batches {
        for r in 0..b.num_rows() {
            let vals: Vec<String> =
                b.columns().iter().map(|c| array_value_to_string(c, r).unwrap_or_default()).collect();
            out.push(format!("{open}{}{close}", vals.join(", ")));
        }
    }
    out.join(sep)
}

async fn values(cc: &CachingContext, rows: &str) -> Result<Vec<RecordBatch>> {
    cc.ctx
        .sql(&format!("SELECT * FROM (VALUES {rows}) AS t(order_id, store, amount)"))
        .await?
        .collect()
        .await
}

fn step(n: u32, title: &str, expect: &str) {
    println!("\n━━ {n}. {title}");
    println!("   expect: {expect}");
}

pub async fn run() -> Result<()> {
    let mut cc = CachingContext::new();
    cc.cache.set_segment_files(1); // one cache segment per file
    let _ = std::fs::remove_dir_all(DIR);

    let commits = vec![
        values(&cc, "(1, 'north', 10), (2, 'north', 20), (3, 'south', 5)").await?,
        values(&cc, "(4, 'north', 40), (5, 'east', 7), (6, 'east', 8)").await?,
        values(&cc, "(7, 'south', 15), (8, 'south', 25)").await?,
    ];
    delta::create_with_commits(DIR, commits).await?;
    cc.delta.open(&cc.cache, &cc.ctx, "orders", DIR).await?;
    let mut ex = Example { cc, labels: HashMap::new(), next: 0 };

    println!("Partial state columns: Q_all {{store, sum, count}}, Q_north {{store, sum}}");

    step(1, "Create the table: 3 commits, one file each", "both queries read every file; one partial state cached per file");
    ex.show_table("3 commits").await?;
    ex.query(Q_ALL).await?;
    ex.query(Q_NORTH).await?;

    step(2, "Run Q_all again, nothing changed", "the final result itself is reused");
    ex.query(Q_ALL).await?;

    step(3, "APPEND a file containing only a brand-new group (west)",
        "only the new file is read; A, B and C are reused untouched; the merge adds the 'west' group");
    let batches = values(&ex.cc, "(9, 'west', 100), (10, 'west', 50)").await?;
    ex.cc.delta.append(&ex.cc.cache, &ex.cc.ctx, "orders", batches).await?;
    ex.show_table("append (9, west, 100), (10, west, 50)").await?;
    ex.query(Q_ALL).await?;
    ex.query(Q_NORTH).await?;

    step(4, "DELETE one row (order 5, east)", "Delta rewrites only file B; only that rewritten file is read");
    let m = ex.cc.delta.delete(&ex.cc.cache, &ex.cc.ctx, "orders", "order_id = 5").await?;
    ex.show_table(&format!("DELETE WHERE order_id = 5: {m}")).await?;
    ex.query(Q_ALL).await?;
    ex.query(Q_NORTH).await?;

    step(5, "UPDATE: double the amounts of store 'south'",
        "Delta rewrites the files that contain 'south' rows; only those are read");
    let m = ex.cc.delta.update(&ex.cc.cache, &ex.cc.ctx, "orders", "store = 'south'", "amount", "amount * 2").await?;
    ex.show_table(&format!("UPDATE SET amount = amount * 2 WHERE store = 'south': {m}")).await?;
    ex.query(Q_ALL).await?;
    ex.query(Q_NORTH).await?;

    println!("\n━━ Cache contents at the end (stale entries from removed files were dropped)");
    ex.cc.print_cache();
    Ok(())
}
