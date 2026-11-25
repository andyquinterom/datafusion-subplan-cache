//! Physical side: the tee operator that fills the cache, and the planner
//! hook that turns the logical cache nodes into execution plans.

use std::cmp::Reverse;
use std::collections::HashSet;
use std::fmt;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll};
use std::time::Instant;

use async_trait::async_trait;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::provider_as_source;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SessionState, SessionStateBuilder};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, ReplaceChildrenOptions,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use futures::{Stream, StreamExt};

use crate::cache::{Cache, CacheEntry, EntryData, SegmentUse, Stored};
use crate::rewrite::{CacheReadNode, CacheWriteNode, Rewriter, SegmentedAggregateNode, short_id};
use crate::lift::{self, Class};
use crate::table::{Binding, DataFile};

// ---------------------------------------------------------------------------
// CacheWriteExec: passes batches through and stores a copy
// ---------------------------------------------------------------------------

/// What a CacheWriteExec will store its result as.
#[derive(Debug)]
pub struct WriteMeta {
    pub key: String,
    pub id: String,
    pub label: String,
    pub deps: Vec<String>,
    pub segment_of: Option<String>,
}

#[derive(Debug)]
struct Sink {
    started: Mutex<Option<Instant>>,
    parts: Mutex<Vec<Option<Vec<RecordBatch>>>>,
}

pub struct CacheWriteExec {
    meta: Arc<WriteMeta>,
    input: Arc<dyn ExecutionPlan>,
    cache: Arc<Cache>,
    sink: Arc<Sink>,
}

impl fmt::Debug for CacheWriteExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CacheWriteExec({})", self.meta.id)
    }
}

impl CacheWriteExec {
    pub fn new(meta: Arc<WriteMeta>, input: Arc<dyn ExecutionPlan>, cache: Arc<Cache>) -> Self {
        let n = input.output_partitioning().partition_count();
        let sink = Arc::new(Sink { started: Mutex::new(None), parts: Mutex::new(vec![None; n]) });
        Self { meta, input, cache, sink }
    }

    /// Called when one partition's stream ran to completion. When every
    /// partition has completed, the full result is committed to the cache.
    /// Streams that are dropped early (LIMIT, errors, cancellation) never
    /// report, so a partial result can never be cached.
    fn finish_partition(&self, partition: usize, batches: Vec<RecordBatch>) {
        let mut parts = self.sink.parts.lock().unwrap();
        parts[partition] = Some(batches);
        if parts.iter().any(Option::is_none) {
            return;
        }
        let partitions: Vec<Vec<RecordBatch>> = parts.iter_mut().map(|p| p.take().unwrap()).collect();
        let rows = partitions.iter().flatten().map(|b| b.num_rows()).sum();
        let bytes = partitions.iter().flatten().map(|b| b.get_array_memory_size()).sum();
        let started = self.sink.started.lock().unwrap().unwrap_or_else(Instant::now);
        let m = &self.meta;
        let partition_batches = partitions.iter().map(Vec::len).collect();
        self.cache.insert(CacheEntry {
            stored: Mutex::new(Stored { data_file: String::new(), partition_batches }),
            key: m.key.clone(),
            id: m.id.clone(),
            label: m.label.clone(),
            data: OnceLock::from(Arc::new(EntryData { schema: self.input.schema(), partitions })),
            persisted: AtomicBool::new(false),
            rows,
            bytes,
            produced_in: started.elapsed(),
            hits: AtomicUsize::new(0),
            deps: m.deps.clone(),
            segment_of: m.segment_of.clone(),
        });
    }
}

impl DisplayAs for CacheWriteExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CacheWriteExec: id={} {}", self.meta.id, self.meta.label)
    }
}

impl ExecutionPlan for CacheWriteExec {
    fn name(&self) -> &str {
        "CacheWriteExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(self.meta.clone(), children.swap_remove(0), self.cache.clone())))
    }
    #[allow(deprecated)]
    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(children, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
    }
    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        self.sink.started.lock().unwrap().get_or_insert_with(Instant::now);
        let inner = self.input.execute(partition, context)?;
        let exec = Arc::new(Self {
            meta: self.meta.clone(),
            input: self.input.clone(),
            cache: self.cache.clone(),
            sink: self.sink.clone(), // shared across partitions
        });
        Ok(Box::pin(TeeStream { inner, buf: Vec::new(), partition, exec, done: false }))
    }
}

struct TeeStream {
    inner: SendableRecordBatchStream,
    buf: Vec<RecordBatch>,
    partition: usize,
    exec: Arc<CacheWriteExec>,
    done: bool,
}

impl Stream for TeeStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        match this.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                this.buf.push(batch.clone()); // cheap: Arc'd buffers
                Poll::Ready(Some(Ok(batch)))
            }
            Poll::Ready(None) => {
                if !this.done {
                    this.done = true;
                    this.exec.finish_partition(this.partition, std::mem::take(&mut this.buf));
                }
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

impl RecordBatchStream for TeeStream {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

// ---------------------------------------------------------------------------
// Segmented aggregates
// ---------------------------------------------------------------------------

fn first_aggregate(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    let mut found = None;
    let _ = plan.apply(|n| {
        if n.is::<AggregateExec>() {
            found = Some(n.clone());
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

fn mode(plan: &Arc<dyn ExecutionPlan>) -> Option<AggregateMode> {
    plan.downcast_ref::<AggregateExec>().map(|a| *a.mode())
}

/// The plan of a single aggregate is `Final(…Partial(input))`. Returns the
/// Partial node, or None if DataFusion chose a single-stage aggregate.
fn partial_aggregate(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    let top = first_aggregate(plan)?;
    if !matches!(mode(&top), Some(AggregateMode::Final | AggregateMode::FinalPartitioned)) {
        return None;
    }
    let partial = first_aggregate(&top.children()[0].clone())?;
    matches!(mode(&partial), Some(AggregateMode::Partial)).then_some(partial)
}

impl CacheExtensionPlanner {
    /// Build `Final( union( cached segment states…, Partial(missing files)… ) )`.
    async fn plan_segmented(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &SegmentedAggregateNode,
        template: Arc<dyn ExecutionPlan>,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let cache = &self.cache;
        let snap = &node.snap;
        let Some(template_partial) = partial_aggregate(&template) else {
            cache.event(format!("  · {}: single-stage aggregate, no per-segment caching", node.id));
            return Ok(template);
        };

        // 0. With lifted conjuncts, split the files: inside ones go through
        //    the cached unfiltered segments, edge ones are read with the full
        //    predicate, outside ones are not read at all.
        let mut inside: Vec<&DataFile> = vec![];
        let mut edge: Vec<&DataFile> = vec![];
        let mut skipped = 0;
        for f in snap.files() {
            match lift::classify(f, &node.lifted) {
                Class::Inside => inside.push(f),
                Class::Edge => edge.push(f),
                Class::Outside => skipped += 1,
            }
        }

        // 1. Reuse cached segments whose files are all still in the snapshot
        //    (and kept whole by the lifted conjuncts). Larger segments first;
        //    segments never overlap within one plan.
        let live: HashSet<String> = inside.iter().map(|f| snap.file_dep(f)).collect();
        let mut segments = cache.segments(&node.base_key);
        segments.sort_by_key(|e| Reverse(e.deps.len()));
        let mut covered: HashSet<String> = HashSet::new();
        let mut children: Vec<Arc<dyn ExecutionPlan>> = vec![];
        let mut reused: Vec<Vec<String>> = vec![];
        let mut computed: Vec<Vec<String>> = vec![];
        let mut chosen = vec![];
        let mut claimed: HashSet<&str> = HashSet::new();
        for e in &segments {
            if e.deps.iter().all(|d| live.contains(d) && !claimed.contains(d.as_str())) {
                claimed.extend(e.deps.iter().map(String::as_str));
                chosen.push(e.clone());
            }
        }
        // Segments only in the persistent store are fetched concurrently. A
        // segment purged meanwhile comes back as None: its files are then
        // computed below like any other uncovered file.
        let loaded = futures::future::try_join_all(chosen.iter().map(|e| cache.try_data(e))).await?;
        for (e, d) in chosen.iter().zip(loaded) {
            let Some(d) = d else { continue };
            covered.extend(e.deps.iter().cloned());
            e.hits.fetch_add(1, Ordering::Relaxed);
            reused.push(e.deps.clone());
            children.push(MemorySourceConfig::try_new_exec(&d.partitions, d.schema.clone(), None)?);
        }

        // 2. Plan the partial aggregate over the remaining files, in segments.
        //    Sorted by the first lifted column, so that segments hold
        //    neighbouring values and a moving window invalidates few of them.
        let mut missing: Vec<&DataFile> = inside.iter().copied().filter(|f| !covered.contains(&snap.file_dep(f))).collect();
        if let Some(first) = node.lifted.first() {
            let key = |f: &DataFile| f.stats.get(&first.column).map(|s| s.min.clone());
            missing.sort_by(|a, b| match (key(a), key(b)) {
                (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
                (x, y) => x.is_none().cmp(&y.is_none()),
            });
        }
        let per_segment = cache.segment_files.load(Ordering::Relaxed).max(1);
        // Segment plans are spliced into the outer plan, which is physically
        // optimized as a whole afterwards. Optimizing them here too would run
        // rules twice (e.g. join input swaps after dynamic filters exist).
        let unoptimized = match session.as_any().downcast_ref::<SessionState>() {
            Some(state) => SessionStateBuilder::new_from_existing(state.clone())
                .with_physical_optimizer_rules(vec![])
                .build(),
            None => return Ok(template),
        };
        // The partial aggregate of `plan` over just `files`, or None if that
        // cannot be planned (then fall back to the template).
        let restricted_partial = async |plan: &LogicalPlan, files: &[&DataFile]| -> Result<Option<Arc<dyn ExecutionPlan>>> {
            let Some(provider) = snap.inner.provider_for(files).await? else {
                cache.event(format!("  · {}: {} cannot scan a subset of its files, no per-segment caching", node.id, snap.name));
                return Ok(None);
            };
            let source = provider_as_source(provider);
            // Replace the scans of exactly this snapshot (matched by provider).
            let restricted = plan
                .clone()
                .transform_up(|p| match p {
                    LogicalPlan::TableScan(mut scan)
                        if matches!(cache.registry.resolve(&scan.source), Some(Binding::Files(s)) if Arc::ptr_eq(&s, snap)) =>
                    {
                        scan.source = source.clone();
                        Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
                    }
                    other => Ok(Transformed::no(other)),
                })?
                .data;
            let physical = planner.create_physical_plan(&restricted, &unoptimized).await?;
            let Some(partial) = partial_aggregate(&physical) else {
                cache.event(format!("  · {}: segment planned single-stage, no per-segment caching", node.id));
                return Ok(None);
            };
            if partial.schema() != template_partial.schema() {
                cache.event(format!("  · {}: segment state schema differs from the query's, no per-segment caching", node.id));
                return Ok(None);
            }
            Ok(Some(partial))
        };

        // Edge files: the full predicate, read live, not cached.
        let mut edge_ids = vec![];
        for chunk in edge.chunks(per_segment) {
            let Some(partial) = restricted_partial(&node.input, chunk).await? else { return Ok(template) };
            edge_ids.extend(chunk.iter().map(|f| f.id.clone()));
            children.push(partial);
        }

        for chunk in missing.chunks(per_segment) {
            let Some(partial) = restricted_partial(&node.windowless, chunk).await? else { return Ok(template) };
            let deps: Vec<String> = chunk.iter().map(|f| snap.file_dep(f)).collect();
            let key = format!("{}\n-- segment: {}", node.base_key, deps.join(","));
            let meta = WriteMeta {
                id: short_id(&key),
                key,
                label: format!("partial states of {} for {} file(s)", node.id, chunk.len()),
                deps,
                segment_of: Some(node.base_key.clone()),
            };
            computed.push(meta.deps.clone());
            children.push(Arc::new(CacheWriteExec::new(Arc::new(meta), partial, cache.clone())));
        }

        let mut lifted: Vec<String> = node.lifted.iter().map(|l| l.expr.to_string()).collect();
        lifted.dedup(); // a BETWEEN lifts as two predicates sharing one expression
        let mut msg = format!(
            "  ▦ segmented {} over {} v{}: {} of {} files from {} cached segment(s), {} file(s) computed in {} new segment(s)",
            node.id,
            snap.name,
            snap.version(),
            covered.len(),
            snap.files().len(),
            reused.len(),
            missing.len(),
            computed.len()
        );
        if !lifted.is_empty() {
            msg += &format!(
                "; lifted [{}]: {} edge file(s) read live, {} file(s) skipped",
                lifted.join(" AND "),
                edge.len(),
                skipped
            );
        }
        cache.event(msg);
        cache.record_segment_use(SegmentUse {
            base_key: node.base_key.clone(),
            reused,
            computed,
            lifted,
            edge: edge_ids,
            skipped,
        });
        if children.is_empty() {
            return Ok(template); // empty table
        }

        // 3. Splice the union in where the template's Partial aggregate was.
        let merged: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(UnionExec::try_new(children)?));
        Ok(template
            .transform_down(|p| {
                if Arc::ptr_eq(&p, &template_partial) {
                    Ok(Transformed::new(merged.clone(), true, TreeNodeRecursion::Jump))
                } else {
                    Ok(Transformed::no(p))
                }
            })?
            .data)
    }
}

// ---------------------------------------------------------------------------
// Planner hooks
// ---------------------------------------------------------------------------

/// Turns the cache's logical nodes into execution plans. Add it to your own
/// `DefaultPhysicalPlanner` if you have a custom [`QueryPlanner`], and run
/// [`crate::SubplanCache::rewrite`] on the optimized plan first.
#[derive(Debug)]
pub struct CacheExtensionPlanner {
    cache: Arc<Cache>,
}

impl CacheExtensionPlanner {
    pub fn new(cache: &crate::SubplanCache) -> Self {
        Self { cache: cache.inner.clone() }
    }
}

#[async_trait]
impl ExtensionPlanner for CacheExtensionPlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session: &dyn Session,
        _ctx: &PhysicalPlanningContext,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(w) = node.as_any().downcast_ref::<CacheWriteNode>() {
            let meta = WriteMeta {
                key: w.key.clone(),
                id: w.id.clone(),
                label: w.label.clone(),
                deps: w.deps.clone(),
                segment_of: None,
            };
            return Ok(Some(Arc::new(CacheWriteExec::new(Arc::new(meta), physical_inputs[0].clone(), self.cache.clone()))));
        }
        if let Some(r) = node.as_any().downcast_ref::<CacheReadNode>() {
            let d = self.cache.data(&r.entry).await?;
            return Ok(Some(MemorySourceConfig::try_new_exec(&d.partitions, d.schema.clone(), None)?));
        }
        if let Some(s) = node.as_any().downcast_ref::<SegmentedAggregateNode>() {
            return Ok(Some(self.plan_segmented(planner, s, physical_inputs[0].clone(), session).await?));
        }
        Ok(None)
    }
}

/// The query planner [`crate::SubplanCache::install`] puts into a session:
/// rewrites every optimized plan to use the cache, then plans it.
#[derive(Debug)]
pub struct CachingQueryPlanner {
    pub(crate) cache: Arc<Cache>,
}

/// Plans the cache never touches.
fn passthrough(plan: &LogicalPlan) -> bool {
    matches!(
        plan,
        LogicalPlan::Dml(_)
            | LogicalPlan::Ddl(_)
            | LogicalPlan::Statement(_)
            | LogicalPlan::Explain(_)
            | LogicalPlan::Analyze(_)
            | LogicalPlan::DescribeTable(_)
    )
}

/// Rewrite `plan` to use the cache: pick up new entries from the store,
/// drop entries of snapshots no longer registered, then rewrite. A
/// whole-result hit whose data vanished from the store meanwhile is
/// forgotten by `try_data`; rewriting again then plans that subtree normally.
pub(crate) async fn rewrite_with_cache(cache: &Cache, plan: &LogicalPlan) -> Result<LogicalPlan> {
    let found = cache.sync().await?;
    if found > 0 {
        cache.event(format!("  ⇣ found {found} cache entries in {}", cache.store.as_ref().unwrap().location));
    }
    let dropped = cache.gc(&cache.registry.live_deps());
    if dropped > 0 {
        cache.event(format!("  🗑 dropped {dropped} cache entries built from files no longer in any registered snapshot"));
    }
    let mut attempts = 0;
    loop {
        let mut r = Rewriter { cache, report: Default::default() };
        let rewritten = r.rewrite(plan.clone())?;
        let hits = r.report.hits.clone();
        let loaded = futures::future::try_join_all(hits.into_iter().map(|e| async move { cache.try_data(&e).await })).await?;
        attempts += 1;
        if loaded.iter().all(Option::is_some) || attempts == 3 {
            cache.push_report(r.report);
            return Ok(rewritten);
        }
    }
}

#[async_trait]
impl QueryPlanner for CachingQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let planner = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(CacheExtensionPlanner {
            cache: self.cache.clone(),
        })]);
        if passthrough(logical_plan) || !self.cache.enabled.load(Ordering::Relaxed) {
            return planner.create_physical_plan(logical_plan, session_state).await;
        }
        let rewritten = rewrite_with_cache(&self.cache, logical_plan).await?;
        planner.create_physical_plan(&rewritten, session_state).await
    }
}
