//! Query latency over a sequence of table changes, as a line chart.
//!
//!     cargo run --release -- timeline [--segsize N]
//!
//!     cargo run --release -- timeline partial-optimize [--segsize N]
//!
//! Runs Q1 a few times, changes the Delta table, runs Q1 again, and so on.
//! Every run is timed twice: through the subplan cache, and without any
//! caching as a baseline. Writes `<name>.csv` and a self-contained
//! `<name>.html` chart.
//!
//! Scenarios:
//!   * `mixed` (default): append, delete, update, append, full OPTIMIZE.
//!   * `partial-optimize`: small appends, then OPTIMIZE with a target size
//!     the original files already exceed, so only the small files are
//!     compacted and the big ones stay untouched.
//!   * `restart`: the cache persists to disk (`--cache-dir`, default
//!     `data/timeline_cache`, wiped first); RESTART replaces the whole
//!     context (empty memory, same store), like a new process would.

use std::sync::Arc;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::Result;

use datafusion_subplan_cache::PersistentStore;

use crate::{CachingContext, Q1, ROWS_PER_COMMIT};

const REPEATS: usize = 3;

struct Run {
    n: usize,
    phase: String,
    cached_ms: f64,
    uncached_ms: f64,
    /// Same query through a memory-only cache (restart scenario only).
    memory_ms: Option<f64>,
    /// How the cached run got its answer.
    how: String,
    same: bool,
}

struct Event {
    /// The event happened after this many runs.
    after: usize,
    label: String,
    detail: String,
}

enum Change {
    Append(u64),
    Delete(&'static str),
    Update(&'static str, &'static str, &'static str),
    /// OPTIMIZE with an optional target file size in bytes (default: the
    /// table's, 100 MiB). Files larger than the target are left alone.
    Optimize(Option<u64>),
    /// Throw away the context, memory cache included; keep the store.
    Restart,
}

struct Scenario {
    name: &'static str,
    file: &'static str,
    description: &'static str,
    script: &'static [(&'static str, Change)],
}

const SMALL_APPEND: u64 = 50_000;
/// The original files are ~18.9 MB, so they count as "already optimized".
const TARGET_SIZE: u64 = 16 << 20;

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "mixed",
        file: "timeline",
        description: "Appends, a delete, an update, and a full OPTIMIZE that rewrites nearly every file.",
        script: &[
            ("APPEND", Change::Append(ROWS_PER_COMMIT)),
            ("DELETE", Change::Delete("order_id >= 4500000 AND order_id < 4500100")),
            ("UPDATE", Change::Update("order_id >= 12000000 AND order_id < 12000100", "amount", "amount + 1")),
            ("APPEND", Change::Append(ROWS_PER_COMMIT)),
            ("OPTIMIZE", Change::Optimize(None)),
        ],
    },
    Scenario {
        name: "partial-optimize",
        file: "timeline_partial_optimize",
        description: "Small appends (50k rows each), then OPTIMIZE with a 16 MiB target. The 20 original files \
                      (~18.9 MB) are already above the target, so OPTIMIZE leaves them alone and compacts only the \
                      small files.",
        script: &[
            ("APPEND small", Change::Append(SMALL_APPEND)),
            ("APPEND small", Change::Append(SMALL_APPEND)),
            ("APPEND small", Change::Append(SMALL_APPEND)),
            ("OPTIMIZE", Change::Optimize(Some(TARGET_SIZE))),
            ("APPEND small", Change::Append(SMALL_APPEND)),
            ("OPTIMIZE", Change::Optimize(Some(TARGET_SIZE))),
        ],
    },
    Scenario {
        name: "restart",
        file: "timeline_restart",
        description: "Two caches side by side: one also persisted to a store (local directory or bucket), one memory-only. RESTART \
                      replaces both with fresh contexts (empty memory), as a new process would; the store \
                      is kept.",
        script: &[
            ("RESTART", Change::Restart),
            ("APPEND", Change::Append(ROWS_PER_COMMIT)),
            ("RESTART", Change::Restart),
            ("DELETE", Change::Delete("order_id >= 4500000 AND order_id < 4500100")),
            ("RESTART", Change::Restart),
        ],
    },
];

const DEFAULT_STORE: &str = "data/timeline_cache";

/// A context like `cc` (same segment size) with the given store.
async fn fresh(cc: &CachingContext, store: Option<Arc<PersistentStore>>) -> Result<CachingContext> {
    let mut new = CachingContext::with_store(store);
    new.cache.set_segment_files(cc.cache.segment_files());
    crate::ensure_data(&mut new).await?;
    Ok(new)
}

async fn time_runs(cc: &mut CachingContext, mut mem: Option<&mut CachingContext>, phase: &str, runs: &mut Vec<Run>) -> Result<()> {
    for _ in 0..REPEATS {
        let loads = cc.cache.loads();
        let out = cc.sql(Q1).await?;
        let loaded = cc.cache.loads() - loads;
        let from_store = if loaded > 0 { format!(" ({loaded} read from store)") } else { String::new() };
        let uses = cc.cache.take_segment_uses();
        let mut memory_ms = None;
        let mut results = vec![out.batches.clone()];
        if let Some(m) = mem.as_deref_mut() {
            let o = m.sql(Q1).await?;
            memory_ms = Some(o.elapsed.as_secs_f64() * 1e3);
            results.push(o.batches);
        }
        let (reference, t) = cc.sql_uncached(Q1).await?;
        let reference = pretty_format_batches(&reference)?.to_string();
        let mut same = true;
        for r in &results {
            same &= pretty_format_batches(r)?.to_string() == reference;
        }
        let how = if uses.is_empty() {
            if out.report.hits.is_empty() { "computed, no cache".to_string() } else { format!("whole result from cache{from_store}") }
        } else {
            let files = |s: &Vec<Vec<String>>| s.iter().map(Vec::len).sum::<usize>();
            let computed: usize = uses.iter().map(|u| files(&u.computed)).sum();
            let reused: usize = uses.iter().map(|u| files(&u.reused)).sum();
            format!("{computed} files read, {reused} reused{from_store}")
        };
        let run = Run {
            n: runs.len() + 1,
            phase: phase.to_string(),
            cached_ms: out.elapsed.as_secs_f64() * 1e3,
            uncached_ms: t.as_secs_f64() * 1e3,
            memory_ms,
            how,
            same,
        };
        println!(
            "  run {:>2}  cached {:>7.1} ms   {}uncached {:>7.1} ms   {}{}",
            run.n,
            run.cached_ms,
            run.memory_ms.map(|m| format!("memory-only {m:>7.1} ms   ")).unwrap_or_default(),
            run.uncached_ms,
            run.how,
            if run.same { "" } else { "   ✗ DIFFERS FROM UNCACHED" }
        );
        runs.push(run);
    }
    Ok(())
}

pub async fn run(cc: &mut CachingContext, scenario: Option<&str>, cache_dir: Option<String>) -> Result<()> {
    let name = scenario.unwrap_or("mixed");
    let Some(sc) = SCENARIOS.iter().find(|s| s.name == name) else {
        let names: Vec<&str> = SCENARIOS.iter().map(|s| s.name).collect();
        return Err(datafusion::common::DataFusionError::Plan(format!("unknown scenario {name}; try one of {names:?}")));
    };
    println!("Scenario {}: {}", sc.name, sc.description);
    let restarts = sc.script.iter().any(|(_, c)| matches!(c, Change::Restart));
    if restarts && cc.cache.store().is_none() {
        // Fresh store so the run is reproducible.
        let _ = std::fs::remove_dir_all(DEFAULT_STORE);
        *cc = fresh(cc, Some(Arc::new(PersistentStore::open(DEFAULT_STORE)?))).await?;
    }
    // For comparison: the same cache without a store, restarted alongside.
    let mut mem = if restarts { Some(fresh(cc, None).await?) } else { None };
    if let Some(store) = cc.cache.store() {
        println!("Persistent cache: {}{}", store.location, if cache_dir.is_some() { "" } else { " (wiped)" });
    }
    let segsize = cc.cache.segment_files();
    let mut runs = vec![];
    let mut events = vec![];

    println!("━━ initial table");
    time_runs(cc, mem.as_mut(), "initial table", &mut runs).await?;
    for (label, change) in sc.script {
        println!("━━ {label}");
        let detail = match change {
            Change::Append(rows) => {
                cc.append(*rows).await?;
                format!("{rows} rows in one new file")
            }
            Change::Delete(p) => {
                let m = cc.delta.delete(&cc.cache, &cc.ctx, "sales", p).await?;
                format!("DELETE WHERE {p} ({m})")
            }
            Change::Update(p, col, val) => {
                let m = cc.delta.update(&cc.cache, &cc.ctx, "sales", p, col, val).await?;
                format!("UPDATE SET {col} = {val} WHERE {p} ({m})")
            }
            Change::Restart => {
                *cc = fresh(cc, cc.cache.store().cloned()).await?;
                if mem.is_some() {
                    mem = Some(fresh(cc, None).await?);
                }
                "new contexts: memory caches empty; the persistent store is kept".to_string()
            }
            Change::Optimize(target) => {
                let before = cc.cache.snapshot("sales").unwrap().files().len() as u64;
                let (removed, added) = cc.delta.optimize(&cc.cache, &cc.ctx, "sales", *target).await?;
                let target = target.map(|t| format!(" with a {} MiB target", t >> 20)).unwrap_or_default();
                format!("OPTIMIZE{target}: {removed} files compacted into {added}, {} left untouched", before - removed)
            }
        };
        println!("  {detail}");
        events.push(Event { after: runs.len(), label: label.to_string(), detail });
        time_runs(cc, mem.as_mut(), &format!("after {label}"), &mut runs).await?;
    }

    // CSV
    let mut csv = String::from("run,phase,cached_ms,memory_only_ms,uncached_ms,how,same_result\n");
    for r in &runs {
        let m = r.memory_ms.map(|m| format!("{m:.2}")).unwrap_or_default();
        csv += &format!("{},{},{:.2},{m},{:.2},{},{}\n", r.n, r.phase, r.cached_ms, r.uncached_ms, r.how, r.same);
    }
    std::fs::write(format!("{}.csv", sc.file), csv)?;

    // HTML: the template draws the chart from a JSON blob.
    let json_runs: Vec<String> = runs
        .iter()
        .map(|r| {
            format!(
                "{{\"n\":{},\"phase\":{:?},\"cached\":{:.2},\"uncached\":{:.2},\"memory\":{},\"how\":{:?},\"same\":{}}}",
                r.n,
                r.phase,
                r.cached_ms,
                r.uncached_ms,
                r.memory_ms.map(|m| format!("{m:.2}")).unwrap_or("null".into()),
                r.how,
                r.same
            )
        })
        .collect();
    let json_events: Vec<String> = events
        .iter()
        .map(|e| format!("{{\"after\":{},\"label\":{:?},\"detail\":{:?}}}", e.after, e.label, e.detail))
        .collect();
    let mut description = sc.description.to_string();
    if let Some(store) = cc.cache.store() {
        description += &format!(" Cache store: {}.", store.location);
    }
    description += &format!(" Data: {}.", crate::location::data_dir());
    let data = format!(
        "{{\"segsize\":{segsize},\"scenario\":{:?},\"query\":{:?},\"runs\":[{}],\"events\":[{}]}}",
        description,
        Q1.split_whitespace().collect::<Vec<_>>().join(" "),
        json_runs.join(","),
        json_events.join(",")
    );
    let html = include_str!("timeline.html").replace("/*__DATA__*/null", &data);
    std::fs::write(format!("{}.html", sc.file), html)?;
    println!("\nwrote {0}.csv and {0}.html", sc.file);
    if runs.iter().any(|r| !r.same) {
        println!("✗ some cached results differed from the uncached ones");
    }
    Ok(())
}
