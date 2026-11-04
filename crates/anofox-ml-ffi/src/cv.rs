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

#[repr(C)]
pub struct AnofoxMlSplits {
    /// Input row index of each entry.
    pub row: *mut u64,
    /// Split number (0-based).
    pub split: *mut u32,
    /// 1 for a test row, 0 for a training row.
    pub test: *mut u8,
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
            let len = v.len();
            *out = AnofoxMlSplits {
                row: leak(v.iter().map(|e| e.0 as u64).collect()),
                split: leak(v.iter().map(|e| e.1 as u32).collect()),
                test: leak(v.iter().map(|e| u8::from(e.2)).collect()),
                len,
            };
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while splitting"),
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
}
