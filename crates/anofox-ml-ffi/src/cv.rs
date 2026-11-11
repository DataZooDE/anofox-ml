//! Cross-validation splits behind [`anofox_ml_cv_splits`]: anofox-ml-core's
//! splitters, dispatched by method name, over rows in the caller's order.

use std::ffi::c_char;

use anofox_ml_core::{splitters as sp, utils as ut, CvSplit};
use ndarray::{Array1, Array2};
use serde::Deserialize;

use crate::{err, str_arg};

/// Splitter options (scikit-learn names; all optional).
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Options {
    n_splits: Option<usize>,
    n_repeats: Option<usize>,
    shuffle: Option<bool>,
    seed: Option<u64>,
    test_size: Option<serde_json::Value>,
    train_size: Option<f64>,
    gap: Option<usize>,
    max_train_size: Option<usize>,
    min_train_size: Option<usize>,
    n_bins: Option<usize>,
    p: Option<usize>,
    embargo: Option<usize>,
    n_groups: Option<usize>,
    n_test_groups: Option<usize>,
}

/// What a method needs besides the row count.
#[derive(Clone, Copy, PartialEq)]
enum Needs {
    Rows,
    Y,
    Groups,
    YAndGroups,
}

const METHODS: &[(&str, Needs)] = &[
    ("kfold", Needs::Rows),
    ("repeated_kfold", Needs::Rows),
    ("shuffle_split", Needs::Rows),
    ("leave_one_out", Needs::Rows),
    ("leave_p_out", Needs::Rows),
    ("time_series", Needs::Rows),
    ("stratified_kfold", Needs::Y),
    ("repeated_stratified_kfold", Needs::Y),
    ("stratified_shuffle_split", Needs::Y),
    ("stratified_kfold_continuous", Needs::Y),
    ("group_kfold", Needs::Groups),
    ("leave_one_group_out", Needs::Groups),
    ("leave_p_groups_out", Needs::Groups),
    ("group_shuffle_split", Needs::Groups),
    ("group_time_series", Needs::Groups),
    ("stratified_group_kfold", Needs::YAndGroups),
];

impl Options {
    fn fraction(&self, default: f64) -> Result<f64, String> {
        match &self.test_size {
            None => Ok(default),
            Some(v) => v
                .as_f64()
                .filter(|f| *f > 0.0 && *f < 1.0)
                .ok_or_else(|| format!("test_size must be a fraction in (0, 1) here, got {v}")),
        }
    }

    fn time_series(&self) -> Result<sp::TimeSeriesSplitOptions, String> {
        let test_size = match &self.test_size {
            None => None,
            Some(v) => Some(
                v.as_u64()
                    .ok_or_else(|| format!("test_size must be a row count here, got {v}"))?
                    as usize,
            ),
        };
        Ok(sp::TimeSeriesSplitOptions {
            n_splits: self.n_splits.unwrap_or(5),
            test_size,
            gap: self.gap.unwrap_or(0),
            max_train_size: self.max_train_size,
            min_train_size: self.min_train_size,
        })
    }
}

/// Splits over `n` rows; `y` / `groups` are given for the rows the method
/// uses (see [`Needs`]).
fn splits(
    method: &str,
    o: &Options,
    n: usize,
    y: &Array1<f64>,
    groups: &Array1<usize>,
) -> anofox_ml_core::Result<Vec<CvSplit>> {
    let k = o.n_splits.unwrap_or(5);
    let seed = o.seed.unwrap_or(0);
    let x = || Array2::<f64>::zeros((n, 1));
    let bad = |m: String| anofox_ml_core::RustMlError::InvalidParameter(m);
    Ok(match method {
        "kfold" if o.shuffle.unwrap_or(false) => ut::shuffled_k_fold(n, k, seed)?,
        "kfold" => ut::k_fold(n, k)?,
        "repeated_kfold" => ut::repeated_k_fold(n, k, o.n_repeats.unwrap_or(10), seed)?,
        "shuffle_split" => ut::shuffle_split(
            n,
            o.n_splits.unwrap_or(10),
            o.fraction(0.1).map_err(bad)?,
            seed,
        )?,
        "leave_one_out" => ut::leave_one_out(n),
        "leave_p_out" => ut::leave_p_out(n, o.p.ok_or_else(|| bad("leave_p_out needs p".into()))?)?,
        "time_series" => sp::time_series_split_with(n, &o.time_series().map_err(bad)?)?,
        "stratified_kfold" => ut::stratified_k_fold(&x(), y, k, seed)?,
        "repeated_stratified_kfold" => {
            ut::repeated_stratified_k_fold(&x(), y, k, o.n_repeats.unwrap_or(10), seed)?
        }
        "stratified_shuffle_split" => ut::stratified_shuffle_split(
            &x(),
            y,
            o.n_splits.unwrap_or(10),
            o.fraction(0.1).map_err(bad)?,
            seed,
        )?,
        "stratified_kfold_continuous" => {
            sp::stratified_k_fold_continuous(y, k, o.n_bins.unwrap_or(10), seed)?
        }
        "group_kfold" if o.shuffle.unwrap_or(false) => sp::group_k_fold_shuffled(groups, k, seed)?,
        "group_kfold" => sp::group_k_fold(groups, k)?,
        "leave_one_group_out" => sp::leave_one_group_out(groups)?,
        "leave_p_groups_out" => sp::leave_p_groups_out(
            groups,
            o.p.ok_or_else(|| bad("leave_p_groups_out needs p".into()))?,
        )?,
        "group_shuffle_split" => sp::group_shuffle_split(
            groups,
            o.n_splits.unwrap_or(5),
            o.fraction(0.2).map_err(bad)?,
            o.train_size,
            seed,
        )?,
        "group_time_series" => sp::group_time_series_split(
            groups.as_slice().expect("contiguous"),
            &o.time_series().map_err(bad)?,
        )?,
        "stratified_group_kfold" => {
            sp::stratified_group_k_fold(y, groups, k, o.shuffle.unwrap_or(false).then_some(seed))?
        }
        _ => unreachable!("checked by the caller"),
    })
}

/// `(row, split, is_test)` triples over the `n` input rows. Rows without the
/// target (`NaN`) or group (`u64::MAX`) a method needs are left out.
pub(crate) fn cv_splits(
    method: &str,
    options: &str,
    y: Option<&[f64]>,
    groups: Option<&[u64]>,
    n: usize,
) -> Result<Vec<(usize, usize, bool)>, String> {
    let needs = METHODS
        .iter()
        .find(|(m, _)| *m == method)
        .map(|(_, n)| *n)
        .ok_or_else(|| {
            let names: Vec<&str> = METHODS.iter().map(|(m, _)| *m).collect();
            format!("unknown method '{method}' (known: {names:?})")
        })?;
    let o: Options = if options.trim().is_empty() {
        Options::default()
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    let use_y = matches!(needs, Needs::Y | Needs::YAndGroups);
    let use_groups = matches!(needs, Needs::Groups | Needs::YAndGroups);
    if use_y && y.is_none() {
        return Err(format!("{method} needs a target y"));
    }
    if use_groups && groups.is_none() {
        return Err(format!("{method} needs groups"));
    }
    // Rows the method can use, as indices into the input.
    let keep: Vec<usize> = (0..n)
        .filter(|&i| !use_y || !y.expect("checked")[i].is_nan())
        .filter(|&i| !use_groups || groups.expect("checked")[i] != u64::MAX)
        .collect();
    if keep.is_empty() {
        return Err("no rows".into());
    }
    let yk = Array1::from_iter(keep.iter().map(|&i| y.map_or(0.0, |y| y[i])));
    let gk = Array1::from_iter(keep.iter().map(|&i| groups.map_or(0, |g| g[i] as usize)));
    let s = splits(method, &o, keep.len(), &yk, &gk).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (split, (train, test)) in s.iter().enumerate() {
        out.extend(train.iter().map(|&r| (keep[r], split, false)));
        out.extend(test.iter().map(|&r| (keep[r], split, true)));
    }
    Ok(out)
}

/// One split entry: (row, split, is_test, backtest path or -1).
type Entry = (usize, usize, bool, i32);

/// Purged splits of rows in time order whose labels span
/// `[t_start[i], t_end[i]]`: `purged_kfold` or `combinatorial_purged`
/// (whose test rows are listed once per backtest path they belong to).
pub(crate) fn purged_splits(
    method: &str,
    options: &str,
    t_start: &[f64],
    t_end: &[f64],
) -> Result<Vec<Entry>, String> {
    let o: Options = if options.trim().is_empty() {
        Options::default()
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    let embargo = o.embargo.unwrap_or(0);
    let mut out = Vec::new();
    match method {
        "purged_kfold" => {
            let s = sp::purged_k_fold(t_start, t_end, o.n_splits.unwrap_or(5), embargo)
                .map_err(|e| e.to_string())?;
            for (split, (train, test)) in s.iter().enumerate() {
                out.extend(train.iter().map(|&r| (r, split, false, -1)));
                out.extend(test.iter().map(|&r| (r, split, true, -1)));
            }
        }
        "combinatorial_purged" => {
            let c = sp::combinatorial_purged_cv(
                t_start,
                t_end,
                o.n_groups.unwrap_or(6),
                o.n_test_groups.unwrap_or(2),
                embargo,
            )
            .map_err(|e| e.to_string())?;
            let group_of = |r: usize| {
                c.group_ranges
                    .iter()
                    .position(|&(a, b)| r >= a && r < b)
                    .expect("every row is in a group")
            };
            for (split, (train, test)) in c.splits.iter().enumerate() {
                out.extend(train.iter().map(|&r| (r, split, false, -1)));
                for &r in test {
                    let g = group_of(r);
                    for (path, p) in c.paths.iter().enumerate() {
                        if p[g] == split {
                            out.push((r, split, true, path as i32));
                        }
                    }
                }
            }
        }
        _ => {
            return Err(format!(
                "unknown method '{method}' (known: [\"purged_kfold\", \"combinatorial_purged\"])"
            ))
        }
    }
    Ok(out)
}

/// Purged cross-validation splits of `n` rows in time order, labels spanning
/// `[t_start[i], t_end[i]]`. As [`anofox_ml_cv_splits`]; `path` holds the
/// backtest path of a combinatorial-purged test entry, else -1.
///
/// # Safety
/// `method` / `options` are NUL-terminated (`options` may be NULL);
/// `t_start` / `t_end` hold `n` doubles; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_purged_cv_splits(
    method: *const c_char,
    options: *const c_char,
    n: usize,
    t_start: *const f64,
    t_end: *const f64,
    out: *mut AnofoxMlSplits,
) -> *mut c_char {
    let run = || -> Result<Vec<Entry>, String> {
        let method = str_arg(method, "method")?;
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        if t_start.is_null() || t_end.is_null() || n == 0 {
            return Err("no rows".into());
        }
        purged_splits(
            method,
            options,
            std::slice::from_raw_parts(t_start, n),
            std::slice::from_raw_parts(t_end, n),
        )
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(v)) => {
            *out = splits_out(&v);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(p) => err(crate::panic_message("while splitting", p)),
    }
}

fn splits_out(v: &[Entry]) -> AnofoxMlSplits {
    AnofoxMlSplits {
        row: leak(v.iter().map(|e| e.0 as u64).collect()),
        split: leak(v.iter().map(|e| e.1 as u32).collect()),
        test: leak(v.iter().map(|e| u8::from(e.2)).collect()),
        path: leak(v.iter().map(|e| e.3).collect()),
        len: v.len(),
    }
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SpatialOptions {
    n_splits: Option<usize>,
    seed: Option<u64>,
    /// Cell width of `spatial_block_kfold`, in coordinate units.
    block_size: Option<f64>,
    /// Remove training points within this distance of a test point (radius
    /// of `buffered_loo`).
    buffer: Option<f64>,
    /// `"euclidean"` (default) or `"haversine"` (km; coordinates
    /// `[lat, lon]` in degrees).
    metric: Option<String>,
}

/// Spatial splits of rows with coordinates `coords` (`n × d`):
/// `spatial_block_kfold`, `kfold` (both with an optional `buffer`) or
/// `buffered_loo`.
pub(crate) fn spatial_splits(
    method: &str,
    options: &str,
    coords: &Array2<f64>,
) -> Result<Vec<Entry>, String> {
    let o: SpatialOptions = if options.trim().is_empty() {
        SpatialOptions::default()
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    let metric = match o.metric.as_deref() {
        None | Some("euclidean") => sp::SpatialMetric::Euclidean,
        Some("haversine") => {
            if coords.ncols() != 2 {
                return Err("haversine needs [latitude, longitude] coordinates".into());
            }
            sp::SpatialMetric::Haversine
        }
        Some(m) => {
            return Err(format!(
                "metric must be \"euclidean\" or \"haversine\", got \"{m}\""
            ))
        }
    };
    let k = o.n_splits.unwrap_or(5);
    let seed = o.seed.unwrap_or(0);
    let e = |e: anofox_ml_core::RustMlError| e.to_string();
    let mut splits = match method {
        "spatial_block_kfold" => {
            let size = o.block_size.ok_or("spatial_block_kfold needs block_size")?;
            sp::spatial_block_k_fold(coords, size, k, seed).map_err(e)?
        }
        "kfold" => ut::shuffled_k_fold(coords.nrows(), k, seed).map_err(e)?,
        "buffered_loo" => {
            let r = o.buffer.ok_or("buffered_loo needs buffer (the radius)")?;
            sp::buffered_leave_one_out(coords, r, metric).map_err(e)?
        }
        _ => {
            return Err(format!(
                "unknown method '{method}' (known: [\"spatial_block_kfold\", \"kfold\", \"buffered_loo\"])"
            ))
        }
    };
    if method != "buffered_loo" {
        if let Some(r) = o.buffer {
            splits = sp::spatial_buffer(&splits, coords, r, metric).map_err(e)?;
        }
    }
    let mut out = Vec::new();
    for (split, (train, test)) in splits.iter().enumerate() {
        out.extend(train.iter().map(|&r| (r, split, false, -1)));
        out.extend(test.iter().map(|&r| (r, split, true, -1)));
    }
    Ok(out)
}

/// Spatial cross-validation splits of `n` rows with `n × d` row-major
/// coordinates. As [`anofox_ml_cv_splits`].
///
/// # Safety
/// `method` / `options` are NUL-terminated (`options` may be NULL);
/// `coords` holds `n * d` doubles; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_spatial_cv_splits(
    method: *const c_char,
    options: *const c_char,
    n: usize,
    d: usize,
    coords: *const f64,
    out: *mut AnofoxMlSplits,
) -> *mut c_char {
    let run = || -> Result<Vec<Entry>, String> {
        let method = str_arg(method, "method")?;
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        if coords.is_null() || n == 0 || d == 0 {
            return Err("no rows".into());
        }
        let c = Array2::from_shape_vec((n, d), std::slice::from_raw_parts(coords, n * d).to_vec())
            .map_err(|e| e.to_string())?;
        spatial_splits(method, options, &c)
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(v)) => {
            *out = splits_out(&v);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(p) => err(crate::panic_message("while splitting", p)),
    }
}

#[repr(C)]
pub struct AnofoxMlSplits {
    /// Input row index of each entry.
    pub row: *mut u64,
    /// Split number (0-based).
    pub split: *mut u32,
    /// 1 for a test row, 0 for a training row.
    pub test: *mut u8,
    /// Backtest path of a combinatorial-purged test entry, else -1.
    pub path: *mut i32,
    pub len: usize,
}

fn leak<T>(v: Vec<T>) -> *mut T {
    Box::into_raw(v.into_boxed_slice()).cast()
}

/// Cross-validation splits of `n` rows (in the caller's order) by `method`
/// with JSON `options` (may be NULL). `y` holds `n` targets or is NULL (a
/// `NaN` row is left out where the method needs `y`); `groups` holds `n`
/// group codes or is NULL (`UINT64_MAX` = no group). `out` lists every
/// (row, split, train/test) entry.
///
/// # Safety
/// `method` / `options` are NUL-terminated strings (`options` may be NULL);
/// `y` / `groups` hold `n` values or are NULL; `out` is writable. Release it
/// with [`anofox_ml_free_splits`].
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_cv_splits(
    method: *const c_char,
    options: *const c_char,
    n: usize,
    y: *const f64,
    groups: *const u64,
    out: *mut AnofoxMlSplits,
) -> *mut c_char {
    let run = || -> Result<Vec<(usize, usize, bool)>, String> {
        let method = str_arg(method, "method")?;
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        let y = (!y.is_null()).then(|| std::slice::from_raw_parts(y, n));
        let groups = (!groups.is_null()).then(|| std::slice::from_raw_parts(groups, n));
        cv_splits(method, options, y, groups, n)
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(v)) => {
            let v: Vec<Entry> = v.into_iter().map(|(r, s, t)| (r, s, t, -1)).collect();
            *out = splits_out(&v);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(p) => err(crate::panic_message("while splitting", p)),
    }
}

/// # Safety
/// `s` was filled by [`anofox_ml_cv_splits`].
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_free_splits(s: AnofoxMlSplits) {
    if !s.row.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            s.row, s.len,
        )));
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            s.split, s.len,
        )));
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            s.test, s.len,
        )));
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            s.path, s.len,
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sets(v: &[(usize, usize, bool)]) -> Vec<Vec<usize>> {
        let n = v.iter().map(|e| e.1).max().map_or(0, |m| m + 1);
        let mut out = vec![Vec::new(); n];
        for &(r, s, t) in v {
            if t {
                out[s].push(r);
            }
        }
        out.iter_mut().for_each(|s| s.sort_unstable());
        out
    }

    #[test]
    fn kfold_partitions_the_rows() {
        let v = cv_splits("kfold", r#"{"n_splits": 3}"#, None, None, 6).unwrap();
        assert_eq!(test_sets(&v), vec![vec![0, 1], vec![2, 3], vec![4, 5]]);
        assert_eq!(v.len(), 3 * 6); // every row once per split
    }

    #[test]
    fn time_series_tests_follow_training() {
        let v = cv_splits("time_series", r#"{"n_splits": 2, "gap": 1}"#, None, None, 9).unwrap();
        for s in 0..2 {
            let max_train = v
                .iter()
                .filter(|e| e.1 == s && !e.2)
                .map(|e| e.0)
                .max()
                .unwrap();
            let min_test = v
                .iter()
                .filter(|e| e.1 == s && e.2)
                .map(|e| e.0)
                .min()
                .unwrap();
            assert!(max_train + 1 < min_test, "gap respected");
        }
    }

    #[test]
    fn groups_never_straddle_and_missing_rows_are_left_out() {
        let groups = [0, 0, 1, 1, 2, 2, u64::MAX];
        let v = cv_splits("group_kfold", r#"{"n_splits": 3}"#, None, Some(&groups), 7).unwrap();
        assert!(v.iter().all(|e| e.0 != 6));
        for t in test_sets(&v) {
            let gs: std::collections::BTreeSet<_> = t.iter().map(|&r| groups[r]).collect();
            assert_eq!(gs.len(), 1);
        }
        let e = cv_splits("stratified_kfold", "", None, None, 4).unwrap_err();
        assert!(e.contains("needs a target"), "{e}");
        let e = cv_splits("kfold", r#"{"k": 3}"#, None, None, 4).unwrap_err();
        assert!(e.contains("unknown field"), "{e}");
    }

    #[test]
    fn purged_folds_drop_overlapping_labels() {
        // 12 rows; every label spans 2 steps, so the training rows right
        // before each test block overlap it and are purged.
        let t: Vec<f64> = (0..12).map(f64::from).collect();
        let e: Vec<f64> = t.iter().map(|v| v + 2.0).collect();
        let v = purged_splits("purged_kfold", r#"{"n_splits": 3}"#, &t, &e).unwrap();
        let train1: Vec<usize> = v.iter().filter(|x| x.1 == 1 && !x.2).map(|x| x.0).collect();
        assert!(!train1.contains(&2) && !train1.contains(&3), "{train1:?}");
        // Combinatorial: every path covers every row exactly once.
        let v = purged_splits(
            "combinatorial_purged",
            r#"{"n_groups": 4, "n_test_groups": 2}"#,
            &t,
            &e,
        )
        .unwrap();
        let n_paths = v.iter().map(|x| x.3).max().unwrap() + 1;
        assert_eq!(n_paths, 3); // C(3, 1)
        for p in 0..n_paths {
            let mut rows: Vec<usize> = v.iter().filter(|x| x.2 && x.3 == p).map(|x| x.0).collect();
            rows.sort_unstable();
            assert_eq!(rows, (0..12).collect::<Vec<_>>());
        }
    }

    #[test]
    fn spatial_blocks_and_buffers() {
        // A 10 x 10 grid of points, 1 unit apart.
        let c = Array2::from_shape_fn((100, 2), |(i, j)| {
            if j == 0 {
                (i % 10) as f64
            } else {
                (i / 10) as f64
            }
        });
        let v = spatial_splits(
            "spatial_block_kfold",
            r#"{"block_size": 5, "n_splits": 4}"#,
            &c,
        )
        .unwrap();
        // 4 blocks of 25 points: each split tests exactly one block.
        for s in 0..4 {
            assert_eq!(v.iter().filter(|e| e.1 == s && e.2).count(), 25);
        }
        // A buffer of 1.5 removes the training points next to the test block.
        let b = spatial_splits(
            "spatial_block_kfold",
            r#"{"block_size": 5, "n_splits": 4, "buffer": 1.5}"#,
            &c,
        )
        .unwrap();
        for s in 0..4 {
            let near = |r: usize| {
                b.iter().filter(|e| e.1 == s && e.2).any(|t| {
                    let d = (c[[r, 0]] - c[[t.0, 0]]).hypot(c[[r, 1]] - c[[t.0, 1]]);
                    d <= 1.5
                })
            };
            assert!(b.iter().filter(|e| e.1 == s && !e.2).all(|e| !near(e.0)));
            assert!(b.iter().filter(|e| e.1 == s && !e.2).count() < 75);
        }
        let l = spatial_splits("buffered_loo", r#"{"buffer": 1.0}"#, &c).unwrap();
        assert_eq!(l.iter().filter(|e| e.2).count(), 100);
        assert!(spatial_splits("spatial_block_kfold", "{}", &c)
            .unwrap_err()
            .contains("block_size"));
    }
}
