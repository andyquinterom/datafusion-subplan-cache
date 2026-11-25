//! Predicate lifting: take simple filter conjuncts (`ts >= <literal>`,
//! `region = 'EU'`, `day IN (…)`, …) out of an aggregate's cache key, and
//! decide per data file from its column statistics whether the conjuncts
//! keep all of its rows, none, or some.
//!
//! * **Inside** (every lifted conjunct is true for every row): the file
//!   contributes exactly its *unfiltered* partial aggregate state, which is
//!   cached once and reused by every query whose window covers the file.
//! * **Outside** (some lifted conjunct is false for every row): the file
//!   contributes nothing and is not read.
//! * **Edge** (anything else, or no statistics): the file is read with the
//!   full predicate, uncached.
//!
//! So `WHERE ts >= now() - interval '1 hour'` (DataFusion folds `now()` into
//! a literal, which would otherwise change the cache key on every query)
//! reuses the cached state of every file fully inside the window.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, Result, ScalarValue};
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::utils::{conjunction, split_conjunction};
use datafusion::logical_expr::{Between, BinaryExpr, Expr, Filter, LogicalPlan, Operator, TableScan};

use crate::table::{Binding, ColumnStats, DataFile, Registry, Snapshot};

/// A predicate on one column, decidable from that column's statistics.
#[derive(Debug, Clone, PartialEq)]
pub enum Pred {
    /// `column <op> value` with op one of `= != < <= > >=`.
    Cmp(Operator, ScalarValue),
    In(Vec<ScalarValue>),
    IsNull,
    IsNotNull,
}

/// One lifted conjunct.
#[derive(Clone, PartialEq)]
pub struct Lifted {
    pub column: String,
    pub pred: Pred,
    /// The conjunct as it appeared in the plan.
    pub expr: Expr,
}

impl fmt::Debug for Lifted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

/// How a file relates to a set of lifted conjuncts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Inside,
    Outside,
    Edge,
}

fn cmp(a: &ScalarValue, b: &ScalarValue) -> Option<Ordering> {
    // Dictionary-encoded columns (e.g. partition columns) compare by value.
    fn plain(v: &ScalarValue) -> &ScalarValue {
        match v {
            ScalarValue::Dictionary(_, inner) => plain(inner),
            v => v,
        }
    }
    plain(a).partial_cmp(plain(b))
}

/// Classify a file's column (given its statistics) against one predicate.
pub fn classify_stats(s: &ColumnStats, pred: &Pred) -> Class {
    use Class::*;
    if s.row_count == 0 {
        return Outside;
    }
    let all_null = s.null_count == s.row_count;
    let no_null = s.null_count == 0;
    match pred {
        Pred::IsNull => return if all_null { Inside } else if no_null { Outside } else { Edge },
        Pred::IsNotNull => return if no_null { Inside } else if all_null { Outside } else { Edge },
        _ => {}
    }
    // A comparison with NULL is never true: all-null files contribute nothing.
    if all_null {
        return Outside;
    }
    if s.min.is_null() || s.max.is_null() {
        return Edge;
    }
    // `inside` requires no nulls: a null row fails every comparison.
    let inside = |cond: bool| if cond && no_null { Inside } else { Edge };
    let res = (|| -> Option<Class> {
        Some(match pred {
            Pred::Cmp(op, v) => {
                let (lo, hi) = (cmp(&s.min, v)?, cmp(&s.max, v)?);
                match op {
                    Operator::Gt if hi.is_le() => Outside,
                    Operator::Gt => inside(lo.is_gt()),
                    Operator::GtEq if hi.is_lt() => Outside,
                    Operator::GtEq => inside(lo.is_ge()),
                    Operator::Lt if lo.is_ge() => Outside,
                    Operator::Lt => inside(hi.is_lt()),
                    Operator::LtEq if lo.is_gt() => Outside,
                    Operator::LtEq => inside(hi.is_le()),
                    Operator::Eq if lo.is_gt() || hi.is_lt() => Outside,
                    Operator::Eq => inside(lo.is_eq() && hi.is_eq()),
                    Operator::NotEq if lo.is_eq() && hi.is_eq() => Outside,
                    Operator::NotEq => inside(lo.is_gt() || hi.is_lt()),
                    _ => Edge,
                }
            }
            Pred::In(list) => {
                let mut any_in_range = false;
                for v in list {
                    let (lo, hi) = (cmp(&s.min, v)?, cmp(&s.max, v)?);
                    any_in_range |= lo.is_le() && hi.is_ge();
                }
                if !any_in_range {
                    Outside
                } else {
                    let single = cmp(&s.min, &s.max)?.is_eq();
                    inside(single && list.iter().any(|v| cmp(&s.min, v) == Some(Ordering::Equal)))
                }
            }
            Pred::IsNull | Pred::IsNotNull => unreachable!(),
        })
    })();
    res.unwrap_or(Edge)
}

/// Classify a file against one lifted conjunct.
pub fn classify_one(file: &DataFile, l: &Lifted) -> Class {
    match file.stats.get(&l.column) {
        Some(s) => classify_stats(s, &l.pred),
        None => Class::Edge,
    }
}

/// Classify a file against all lifted conjuncts (their AND).
pub fn classify(file: &DataFile, lifted: &[Lifted]) -> Class {
    let mut all_inside = true;
    for l in lifted {
        match classify_one(file, l) {
            Class::Outside => return Class::Outside,
            Class::Edge => all_inside = false,
            Class::Inside => {}
        }
    }
    if all_inside { Class::Inside } else { Class::Edge }
}

// ---------------------------------------------------------------------------
// Recognizing liftable conjuncts
// ---------------------------------------------------------------------------

/// The literal value of `e` (looking through casts), cast to `ty`.
fn literal(e: &Expr, ty: &DataType) -> Option<ScalarValue> {
    let v = match e {
        Expr::Literal(v, _) => v.clone(),
        Expr::Cast(c) => literal(&c.expr, c.field.data_type())?,
        Expr::TryCast(c) => literal(&c.expr, c.field.data_type())?,
        _ => return None,
    };
    if v.is_null() {
        return None;
    }
    let v = v.cast_to(ty).ok()?;
    (!v.is_null()).then_some(v)
}

/// A plain column of the scanned table (not a cast of it).
fn scan_column<'a>(e: &'a Expr, scan: &TableScan, schema: &'a Schema) -> Option<(&'a Column, &'a DataType)> {
    let Expr::Column(c) = e else { return None };
    if c.relation.as_ref().is_some_and(|r| r != &scan.table_name) {
        return None;
    }
    let field = schema.field_with_name(&c.name).ok()?;
    Some((c, field.data_type()))
}

fn sortable(ty: &DataType) -> bool {
    // Float statistics may leave out NaN, which compares greater than any
    // number: never decide float predicates from them.
    !(ty.is_floating() || ty.is_nested())
}

fn split_disjunction(e: &Expr) -> Vec<&Expr> {
    match e {
        Expr::BinaryExpr(BinaryExpr { left, op: Operator::Or, right }) => {
            let mut v = split_disjunction(left);
            v.extend(split_disjunction(right));
            v
        }
        e => vec![e],
    }
}

/// Parse one conjunct over the scan's columns.
fn parse(e: &Expr, scan: &TableScan, schema: &Schema) -> Vec<Lifted> {
    let one = |c: &Column, pred: Pred| vec![Lifted { column: c.name.clone(), pred, expr: e.clone() }];
    match e {
        // `c = a OR c = b …` (what the optimizer makes of short IN lists).
        Expr::BinaryExpr(BinaryExpr { op: Operator::Or, .. }) => {
            let mut column: Option<&Column> = None;
            let mut values = vec![];
            for d in split_disjunction(e) {
                let Some((c, ty, v)) = (|| {
                    let Expr::BinaryExpr(BinaryExpr { left, op: Operator::Eq, right }) = d else { return None };
                    let (left, right): (&Expr, &Expr) = (left, right);
                    let ((c, ty), v) = match scan_column(left, scan, schema) {
                        Some(ct) => (ct, right),
                        None => (scan_column(right, scan, schema)?, left),
                    };
                    Some((c, ty, literal(v, ty)?))
                })() else {
                    return vec![];
                };
                if !sortable(ty) || column.is_some_and(|x| x != c) {
                    return vec![];
                }
                column = Some(c);
                values.push(v);
            }
            match column {
                Some(c) => one(c, Pred::In(values)),
                None => vec![],
            }
        }
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            use Operator::*;
            if !matches!(op, Eq | NotEq | Lt | LtEq | Gt | GtEq) {
                return vec![];
            }
            let (c, ty, v, op) = if let Some((c, ty)) = scan_column(left, scan, schema) {
                (c, ty, right, *op)
            } else if let Some((c, ty)) = scan_column(right, scan, schema) {
                match op.swap() {
                    Some(op) => (c, ty, left, op),
                    None => return vec![],
                }
            } else {
                return vec![];
            };
            match (sortable(ty), literal(v, ty)) {
                (true, Some(v)) => one(c, Pred::Cmp(op, v)),
                _ => vec![],
            }
        }
        Expr::Between(Between { expr, negated: false, low, high }) => {
            let Some((c, ty)) = scan_column(expr, scan, schema) else { return vec![] };
            match (sortable(ty), literal(low, ty), literal(high, ty)) {
                // Both halves share the conjunct: lifted together or not at all.
                (true, Some(lo), Some(hi)) => vec![
                    Lifted { column: c.name.clone(), pred: Pred::Cmp(Operator::GtEq, lo), expr: e.clone() },
                    Lifted { column: c.name.clone(), pred: Pred::Cmp(Operator::LtEq, hi), expr: e.clone() },
                ],
                _ => vec![],
            }
        }
        Expr::InList(InList { expr, list, negated: false }) => {
            let Some((c, ty)) = scan_column(expr, scan, schema) else { return vec![] };
            if !sortable(ty) {
                return vec![];
            }
            match list.iter().map(|v| literal(v, ty)).collect::<Option<Vec<_>>>() {
                Some(vs) if !vs.is_empty() => one(c, Pred::In(vs)),
                _ => vec![],
            }
        }
        Expr::IsNull(inner) => match scan_column(inner, scan, schema) {
            Some((c, _)) => one(c, Pred::IsNull),
            None => vec![],
        },
        Expr::IsNotNull(inner) => match scan_column(inner, scan, schema) {
            Some((c, _)) => one(c, Pred::IsNotNull),
            None => vec![],
        },
        _ => vec![],
    }
}

fn is_snapshot_scan(registry: &Registry, scan: &TableScan, snap: &Arc<Snapshot>) -> bool {
    matches!(registry.resolve(&scan.source), Some(Binding::Files(s)) if Arc::ptr_eq(&s, snap))
}

/// The conjuncts that filter the scan of `snap` directly: a `Filter` right
/// above the scan (possibly stacked) and the scan's pushed-down filters.
fn candidate_conjuncts(plan: &LogicalPlan, registry: &Registry, snap: &Arc<Snapshot>) -> Vec<(Expr, Arc<TableScan>)> {
    fn filtered_scan<'a>(p: &'a LogicalPlan, registry: &Registry, snap: &Arc<Snapshot>) -> Option<&'a TableScan> {
        match p {
            LogicalPlan::TableScan(s) if is_snapshot_scan(registry, s, snap) => Some(s),
            LogicalPlan::Filter(f) => filtered_scan(&f.input, registry, snap),
            _ => None,
        }
    }
    let mut out: Vec<(Expr, Arc<TableScan>)> = vec![];
    let mut push = |e: &Expr, s: &TableScan| {
        if !out.iter().any(|(x, _)| x == e) {
            out.push((e.clone(), Arc::new(s.clone())));
        }
    };
    let _ = plan.apply(|node| {
        match node {
            LogicalPlan::Filter(f) => {
                if let Some(scan) = filtered_scan(&f.input, registry, snap) {
                    for c in split_conjunction(&f.predicate) {
                        push(c, scan);
                    }
                }
            }
            LogicalPlan::TableScan(s) if is_snapshot_scan(registry, s, snap) => {
                for f in &s.filters {
                    for c in split_conjunction(f) {
                        push(c, s);
                    }
                }
            }
            _ => {}
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    });
    out
}

/// What lifting decided for one aggregate.
pub struct Lift {
    pub lifted: Vec<Lifted>,
    /// The aggregate with the lifted conjuncts removed.
    pub windowless: LogicalPlan,
}

/// Choose the conjuncts of `agg` (an aggregate over `snap`, possibly joined
/// with static tables) to lift: those whose classification over the
/// snapshot's files leaves at most `max_edge_fraction` of them (and at
/// least two are always allowed) as Edge, and not all of them.
pub fn lift(agg: &LogicalPlan, snap: &Arc<Snapshot>, registry: &Registry, max_edge_fraction: f64) -> Result<Option<Lift>> {
    if max_edge_fraction <= 0.0 || snap.files().is_empty() {
        return Ok(None);
    }
    let n = snap.files().len();
    let allowed = ((max_edge_fraction * n as f64).floor() as usize).max(2).min(n - 1);
    let mut lifted: Vec<Lifted> = vec![];
    for (expr, scan) in candidate_conjuncts(agg, registry, snap) {
        let schema = scan.source.schema();
        let parts = parse(&expr, &scan, &schema);
        if parts.is_empty() {
            continue;
        }
        let edges = snap
            .files()
            .iter()
            .filter(|f| classify(f, &parts) == Class::Edge)
            .count();
        if edges <= allowed {
            lifted.extend(parts);
        }
    }
    if lifted.is_empty() {
        return Ok(None);
    }
    let remove: Vec<&Expr> = lifted.iter().map(|l| &l.expr).collect();
    let keep = |e: &Expr| -> Option<Expr> {
        let rest: Vec<Expr> = split_conjunction(e).into_iter().filter(|c| !remove.contains(c)).cloned().collect();
        conjunction(rest)
    };
    let windowless = agg
        .clone()
        .transform_up(|p| match p {
            LogicalPlan::Filter(f) => match keep(&f.predicate) {
                None => Ok(Transformed::yes(Arc::unwrap_or_clone(f.input))),
                Some(pred) if pred != f.predicate => Ok(Transformed::yes(LogicalPlan::Filter(Filter::try_new(pred, f.input)?))),
                Some(_) => Ok(Transformed::no(LogicalPlan::Filter(f))),
            },
            LogicalPlan::TableScan(mut s) if is_snapshot_scan(registry, &s, snap) => {
                s.filters = s.filters.iter().filter_map(&keep).collect();
                Ok(Transformed::yes(LogicalPlan::TableScan(s)))
            }
            other => Ok(Transformed::no(other)),
        })?
        .data;
    Ok(Some(Lift { lifted, windowless }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(min: i64, max: i64, nulls: u64, rows: u64) -> ColumnStats {
        ColumnStats { min: ScalarValue::Int64(Some(min)), max: ScalarValue::Int64(Some(max)), null_count: nulls, row_count: rows }
    }
    fn v(x: i64) -> ScalarValue {
        ScalarValue::Int64(Some(x))
    }
    use Class::*;

    #[test]
    fn comparisons() {
        let s = st(10, 20, 0, 100);
        let c = |op, x| classify_stats(&s, &Pred::Cmp(op, v(x)));
        assert_eq!(c(Operator::Gt, 9), Inside);
        assert_eq!(c(Operator::Gt, 10), Edge);
        assert_eq!(c(Operator::Gt, 20), Outside);
        assert_eq!(c(Operator::GtEq, 10), Inside);
        assert_eq!(c(Operator::GtEq, 21), Outside);
        assert_eq!(c(Operator::GtEq, 20), Edge);
        assert_eq!(c(Operator::Lt, 21), Inside);
        assert_eq!(c(Operator::Lt, 10), Outside);
        assert_eq!(c(Operator::LtEq, 20), Inside);
        assert_eq!(c(Operator::LtEq, 9), Outside);
        assert_eq!(c(Operator::Eq, 15), Edge);
        assert_eq!(c(Operator::Eq, 25), Outside);
        assert_eq!(c(Operator::NotEq, 25), Inside);
        assert_eq!(c(Operator::NotEq, 15), Edge);
        let one = st(7, 7, 0, 5);
        assert_eq!(classify_stats(&one, &Pred::Cmp(Operator::Eq, v(7))), Inside);
        assert_eq!(classify_stats(&one, &Pred::Cmp(Operator::NotEq, v(7))), Outside);
    }

    #[test]
    fn nulls() {
        let s = st(10, 20, 3, 100);
        assert_eq!(classify_stats(&s, &Pred::Cmp(Operator::Gt, v(0))), Edge, "null rows fail the comparison");
        assert_eq!(classify_stats(&s, &Pred::Cmp(Operator::Gt, v(30))), Outside);
        assert_eq!(classify_stats(&s, &Pred::IsNull), Edge);
        assert_eq!(classify_stats(&st(1, 1, 0, 4), &Pred::IsNull), Outside);
        assert_eq!(classify_stats(&st(1, 1, 0, 4), &Pred::IsNotNull), Inside);
        let all_null = ColumnStats { min: ScalarValue::Int64(None), max: ScalarValue::Int64(None), null_count: 4, row_count: 4 };
        assert_eq!(classify_stats(&all_null, &Pred::IsNull), Inside);
        assert_eq!(classify_stats(&all_null, &Pred::Cmp(Operator::Gt, v(0))), Outside);
    }

    #[test]
    fn in_list() {
        assert_eq!(classify_stats(&st(5, 5, 0, 3), &Pred::In(vec![v(1), v(5)])), Inside);
        assert_eq!(classify_stats(&st(5, 9, 0, 3), &Pred::In(vec![v(1), v(5)])), Edge);
        assert_eq!(classify_stats(&st(5, 9, 0, 3), &Pred::In(vec![v(1), v(10)])), Outside);
    }

    #[test]
    fn incomparable_is_edge() {
        let s = st(1, 2, 0, 3);
        assert_eq!(classify_stats(&s, &Pred::Cmp(Operator::Gt, ScalarValue::Utf8(Some("a".into())))), Edge);
    }
}
