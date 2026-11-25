//! The logical-plan rewrite: swap cached subtrees in, wrap expensive ones in
//! cache writers.
//!
//! Every subtree of an *optimized* logical plan has a canonical text form
//! (`display_indent_schema`). That text plus the identity of every snapshot
//! the subtree reads is its cache key. Scans are recognized by their
//! provider (not by table name): a scan is cacheable only if its provider
//! was registered with the cache, as an immutable-file table at a known
//! snapshot or as a static table.
//!
//! The rewrite works top-down: the largest cached subtree wins. A subtree
//! that reads a table the cache does not know is neither read from nor
//! written to the cache, but the rewrite still descends into it, so the
//! parts of it that read only known tables are cached as usual.


use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchemaRef, DataFusionError, Result};
use datafusion::logical_expr::{Expr, Extension, JoinType, LogicalPlan, UserDefinedLogicalNodeCore};

use crate::UnregisteredTables;
use crate::cache::{Cache, CacheEntry};
use crate::lift::{self, Lifted};
use crate::store::stable_hash;
use crate::table::{Binding, Snapshot};

pub fn cache_key(plan: &LogicalPlan) -> String {
    plan.display_indent_schema().to_string()
}

pub fn short_id(key: &str) -> String {
    stable_hash(key)[..8].to_string()
}

pub fn label(plan: &LogicalPlan) -> String {
    let s = plan.display().to_string();
    if s.len() > 100 { format!("{}…", &s[..s.floor_char_boundary(100)]) } else { s }
}

/// Subtrees worth materializing as a whole: they are where the work happens.
fn is_cacheable(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::Aggregate(_) | LogicalPlan::Join(_) | LogicalPlan::Window(_) | LogicalPlan::Distinct(_)
    ) && !has_volatile_exprs(plan)
}

/// `random()`, `now()` etc. anywhere in the subtree make results non-reusable,
/// and so do references to an enclosing query (correlated subqueries).
fn has_volatile_exprs(plan: &LogicalPlan) -> bool {
    let mut volatile = false;
    let _ = plan.apply_with_subqueries(|node| {
        if node.expressions().iter().any(|e| e.is_volatile() || e.contains_outer()) {
            volatile = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    volatile
}

/// What a subtree reads.
enum Reads {
    /// Only known data: the snapshots of file tables, and static tokens.
    Known { files: Vec<Arc<Snapshot>>, statics: Vec<String> },
    /// Something the cache cannot identify (named for the event).
    Unknown(String),
}

impl Reads {
    fn deps(&self) -> Vec<String> {
        match self {
            Reads::Known { files, statics } => {
                let mut deps: Vec<String> = files.iter().map(|s| s.fileset_dep()).chain(statics.iter().cloned()).collect();
                deps.sort();
                deps
            }
            Reads::Unknown(_) => vec![],
        }
    }
}

/// The plans the rewrite needs to look at.
pub struct Rewriter<'a> {
    pub cache: &'a Cache,
    pub report: RewriteReport,
}

impl Rewriter<'_> {
    fn policy(&self) -> UnregisteredTables {
        self.cache.policy
    }

    /// What one scan reads; `None` if it is unknown and that matters.
    fn resolve_scan(&self, scan: &datafusion::logical_expr::TableScan) -> Option<Option<Binding>> {
        match self.cache.registry.resolve(&scan.source) {
            Some(b) => Some(Some(b)),
            None if self.policy() == UnregisteredTables::AssumeStatic => Some(None),
            None => None,
        }
    }

    /// Everything `plan` reads, subqueries included.
    fn reads(&self, plan: &LogicalPlan) -> Reads {
        let mut files: Vec<Arc<Snapshot>> = vec![];
        let mut statics: Vec<String> = vec![];
        let mut unknown = None;
        let _ = plan.apply_with_subqueries(|node| {
            match node {
                LogicalPlan::TableScan(scan) => match self.resolve_scan(scan) {
                    Some(Some(Binding::Files(s))) => {
                        if !files.iter().any(|f| Arc::ptr_eq(f, &s)) {
                            files.push(s);
                        }
                    }
                    Some(Some(Binding::Static(dep))) => {
                        if !statics.contains(&dep) {
                            statics.push(dep);
                        }
                    }
                    Some(None) => {} // assumed static, no token
                    None => {
                        unknown = Some(scan.table_name.to_string());
                        return Ok(TreeNodeRecursion::Stop);
                    }
                },
                // Someone else's plan node: we cannot know what it reads.
                LogicalPlan::Extension(e) if self.policy() != UnregisteredTables::AssumeStatic => {
                    unknown = Some(format!("<{}>", e.node.name()));
                    return Ok(TreeNodeRecursion::Stop);
                }
                _ => {}
            }
            Ok(TreeNodeRecursion::Continue)
        });
        match unknown {
            Some(name) => Reads::Unknown(name),
            None => Reads::Known { files, statics },
        }
    }

    /// Do the subquery expressions of this one node read only static data?
    fn node_subqueries_static(&self, plan: &LogicalPlan) -> bool {
        let mut ok = true;
        let _ = plan.apply_subqueries(|sub| {
            if !matches!(self.reads(sub), Reads::Known { ref files, .. } if files.is_empty()) {
                ok = false;
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        });
        ok
    }

    /// Can this subtree be evaluated file-by-file over a single file table,
    /// with the union of the per-file results equal to the result over all
    /// files?
    ///   Ok(Some(t)) – yes, splitting along the snapshot `t`
    ///   Ok(None)    – the subtree reads only static data (fine to repeat)
    ///   Err(())     – no
    fn split_table(&self, plan: &LogicalPlan) -> Result<Option<Arc<Snapshot>>, ()> {
        match plan {
            LogicalPlan::TableScan(scan) => match self.resolve_scan(scan) {
                Some(Some(Binding::Files(s))) => Ok(Some(s)),
                Some(_) => Ok(None),
                None => Err(()),
            },
            LogicalPlan::Projection(_) | LogicalPlan::Filter(_) | LogicalPlan::SubqueryAlias(_) => {
                if !self.node_subqueries_static(plan) {
                    return Err(());
                }
                self.split_table(plan.inputs()[0])
            }
            LogicalPlan::Join(join) => {
                if !self.node_subqueries_static(plan) {
                    return Err(());
                }
                let l = self.split_table(&join.left)?;
                let r = self.split_table(&join.right)?;
                use JoinType::*;
                match (l, r) {
                    (Some(_), Some(_)) => Err(()),
                    (None, None) => Ok(None),
                    // Splitting a side is only valid if that side's rows are
                    // never needed "globally" (e.g. to emit unmatched rows).
                    (Some(t), None) if matches!(join.join_type, Inner | Left | LeftSemi | LeftAnti | LeftMark) => Ok(Some(t)),
                    (None, Some(t)) if matches!(join.join_type, Inner | Right | RightSemi | RightAnti | RightMark) => Ok(Some(t)),
                    _ => Err(()),
                }
            }
            other => match self.reads(other) {
                Reads::Known { files, .. } if files.is_empty() => Ok(None),
                _ => Err(()),
            },
        }
    }

    /// Rewrite an optimized logical plan to read from / write to the cache.
    pub fn rewrite(&mut self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let reads = self.reads(&plan);
        let cacheable = is_cacheable(&plan);
        if let Reads::Unknown(name) = &reads {
            if cacheable {
                match self.policy() {
                    UnregisteredTables::Error => {
                        return Err(DataFusionError::Plan(format!(
                            "subplan cache: query reads table {name}, which is not registered with the cache \
                             (use SubplanCache::register_table or register_static, or a different UnregisteredTables policy)"
                        )));
                    }
                    _ => {
                        if !self.report.unregistered.contains(name) {
                            self.cache.event(format!("  · not cached: reads unregistered table {name}"));
                            self.report.unregistered.push(name.clone());
                        }
                    }
                }
            }
            return self.recurse(plan);
        }

        let base_key = cache_key(&plan);
        let deps = reads.deps();
        let key = if deps.is_empty() { base_key.clone() } else { format!("{base_key}\n-- data: {}", deps.join(", ")) };

        if let Some(entry) = self.cache.get(&key) {
            entry.hits.fetch_add(1, Ordering::Relaxed);
            self.report.hits.push(entry.clone());
            return Ok(LogicalPlan::Extension(Extension {
                node: Arc::new(CacheReadNode { schema: plan.schema().clone(), entry }),
            }));
        }

        let mut lbl = cacheable.then(|| label(&plan));
        let plan = match &plan {
            // Aggregate over one file table: cache partial states per segment.
            LogicalPlan::Aggregate(agg) if cacheable => match self.split_table(&agg.input) {
                Ok(Some(snap)) => {
                    let fraction = self.cache.lift_max_edge_fraction();
                    let (base_key, lifted, windowless) = match lift::lift(&plan, &snap, &self.cache.registry, fraction)? {
                        // The whole result is keyed by the lifted literals
                        // (e.g. a folded `now()`), so it is rarely asked
                        // for again: cache only the per-file states.
                        Some(l) => {
                            lbl = None;
                            (cache_key(&l.windowless), l.lifted, l.windowless)
                        }
                        None => (base_key, vec![], plan.clone()),
                    };
                    LogicalPlan::Extension(Extension {
                        node: Arc::new(SegmentedAggregateNode {
                            id: short_id(&base_key),
                            base_key,
                            snap,
                            lifted,
                            windowless,
                            input: plan,
                        }),
                    })
                }
                _ => self.recurse(plan)?,
            },
            _ => self.recurse(plan)?,
        };

        if let Some(lbl) = lbl {
            let id = short_id(&key);
            self.report.writes.push((id.clone(), lbl.clone()));
            return Ok(LogicalPlan::Extension(Extension {
                node: Arc::new(CacheWriteNode { key, id, label: lbl, deps, input: plan }),
            }));
        }
        Ok(plan)
    }

    fn recurse(&mut self, plan: LogicalPlan) -> Result<LogicalPlan> {
        Ok(plan
            .map_children(|child| Ok(Transformed::yes(self.rewrite(child)?)))?
            .data
            // Also look inside subquery expressions, e.g. `(SELECT sum(x) FROM cte)`.
            .map_subqueries(|sub| Ok(Transformed::yes(self.rewrite(sub)?)))?
            .data)
    }
}

/// What the rewrite of one query decided.
#[derive(Default, Clone)]
pub struct RewriteReport {
    /// Cached whole results used.
    pub hits: Vec<Arc<CacheEntry>>,
    /// Results that will be cached when the query completes: (id, label).
    pub writes: Vec<(String, String)>,
    /// Tables that kept a subtree from being cached.
    pub unregistered: Vec<String>,
}

impl fmt::Debug for RewriteReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hits: Vec<&str> = self.hits.iter().map(|e| e.id.as_str()).collect();
        f.debug_struct("RewriteReport")
            .field("hits", &hits)
            .field("writes", &self.writes)
            .field("unregistered", &self.unregistered)
            .finish()
    }
}


// ---------------------------------------------------------------------------
// Logical extension nodes
// ---------------------------------------------------------------------------

/// Pass-through node: records its input's output under `key`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheWriteNode {
    pub key: String,
    pub id: String,
    pub label: String,
    pub deps: Vec<String>,
    pub input: LogicalPlan,
}

impl PartialOrd for CacheWriteNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.key.partial_cmp(&other.key)
    }
}

impl UserDefinedLogicalNodeCore for CacheWriteNode {
    fn name(&self) -> &str {
        "CacheWrite"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }
    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }
    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }
    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CacheWrite: id={}", self.id)
    }
    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, mut inputs: Vec<LogicalPlan>) -> Result<Self> {
        Ok(Self { input: inputs.swap_remove(0), ..self.clone() })
    }
}

/// Leaf node: produces a previously cached result.
#[derive(Clone)]
pub struct CacheReadNode {
    /// The exact (qualified) schema of the subtree it replaces, so parents
    /// keep resolving their column references unchanged.
    pub schema: DFSchemaRef,
    pub entry: Arc<CacheEntry>,
}

impl fmt::Debug for CacheReadNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CacheRead({})", self.entry.id)
    }
}
impl PartialEq for CacheReadNode {
    fn eq(&self, other: &Self) -> bool {
        self.entry.key == other.entry.key
    }
}
impl Eq for CacheReadNode {}
impl Hash for CacheReadNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entry.key.hash(state)
    }
}
impl PartialOrd for CacheReadNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.entry.key.partial_cmp(&other.entry.key)
    }
}

impl UserDefinedLogicalNodeCore for CacheReadNode {
    fn name(&self) -> &str {
        "CacheRead"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![]
    }
    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }
    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }
    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CacheRead: id={} rows={}", self.entry.id, self.entry.rows)
    }
    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, _inputs: Vec<LogicalPlan>) -> Result<Self> {
        Ok(self.clone())
    }
}

/// An `Aggregate` whose partial states are cached per segment of the file
/// table snapshot `snap`. `input` is the original Aggregate node; it is planned once
/// normally (as a template) and once per missing segment with a scan that
/// reads only that segment's files.
///
/// With `lifted` conjuncts, segments are computed from `windowless` (the
/// aggregate without them) over files the conjuncts keep entirely, and
/// `base_key` is the key of `windowless`; files they keep partly are read
/// with `input`, files they exclude are skipped.
#[derive(Clone)]
pub struct SegmentedAggregateNode {
    pub base_key: String,
    pub id: String,
    pub snap: Arc<Snapshot>,
    pub lifted: Vec<Lifted>,
    pub windowless: LogicalPlan,
    pub input: LogicalPlan,
}

impl fmt::Debug for SegmentedAggregateNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SegmentedAggregate({} over {}@{})", self.id, self.snap.name, self.snap.fileset)
    }
}
impl PartialEq for SegmentedAggregateNode {
    fn eq(&self, other: &Self) -> bool {
        self.base_key == other.base_key && self.snap.fileset == other.snap.fileset && self.input == other.input
    }
}
impl Eq for SegmentedAggregateNode {}
impl Hash for SegmentedAggregateNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base_key.hash(state);
        self.snap.fileset.hash(state);
    }
}
impl PartialOrd for SegmentedAggregateNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (&self.base_key, &self.snap.fileset).partial_cmp(&(&other.base_key, &other.snap.fileset))
    }
}

impl UserDefinedLogicalNodeCore for SegmentedAggregateNode {
    fn name(&self) -> &str {
        "SegmentedAggregate"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }
    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }
    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }
    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SegmentedAggregate: id={} table={} files={}", self.id, self.snap.name, self.snap.files().len())?;
        if !self.lifted.is_empty() {
            write!(f, " lifted={:?}", self.lifted)?;
        }
        Ok(())
    }
    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, mut inputs: Vec<LogicalPlan>) -> Result<Self> {
        Ok(Self { input: inputs.swap_remove(0), ..self.clone() })
    }
}
