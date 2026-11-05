//! Multiple imputation behind `ml_multiple_impute_agg`: Hmisc's
//! `aregImpute` (additive regression with predictive mean matching) through
//! anofox-ml's `AregImputer`. Missing values are `NaN`; the result is
//! `n_impute` completed copies of the data.

use std::ffi::c_char;

use anofox_ml_preprocessing::{AregImputer, AregMatch, AregVariable};
use serde::Deserialize;

use crate::{err, into_doubles, matrix, str_arg, AnofoxMlDoubles};

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Options {
    /// Completed data sets (default 5).
    n_impute: Option<usize>,
    /// Spline knots (default 3; 0 = linear).
    nk: Option<usize>,
    /// Model only the linear part of the imputed variable (default true).
    tlinear: Option<bool>,
    burnin: Option<usize>,
    /// `"weighted"` (default) or `"closest"`.
    #[serde(rename = "match")]
    matching: Option<String>,
    /// Scale of the weighted donor draw (default 0.2).
    fweighted: Option<f64>,
    /// Per column: `"numeric"` (default), `"linear"` or `"categorical"`.
    kinds: Option<Vec<String>>,
    /// Seed of R's generator (default 1, as `set.seed(1)`).
    seed: Option<i32>,
}

fn imputer(options: &str, n_cols: usize) -> Result<AregImputer, String> {
    let o: Options = if options.trim().is_empty() {
        Options::default()
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    let mut a = AregImputer::new(o.n_impute.unwrap_or(5));
    if let Some(nk) = o.nk {
        a = a.with_nk(nk);
    }
    if let Some(t) = o.tlinear {
        a.tlinear = t;
    }
    if let Some(b) = o.burnin {
        a = a.with_burnin(b);
    }
    a = a.with_matching(match o.matching.as_deref() {
        None | Some("weighted") => AregMatch::Weighted(o.fweighted.unwrap_or(0.2)),
        Some("closest") => AregMatch::Closest,
        Some(m) => {
            return Err(format!(
                "match must be \"weighted\" or \"closest\", got \"{m}\""
            ))
        }
    });
    if let Some(kinds) = o.kinds {
        if kinds.len() != n_cols {
            return Err(format!(
                "kinds has {} entries for {n_cols} columns",
                kinds.len()
            ));
        }
        a = a.with_kinds(
            kinds
                .iter()
                .map(|k| match k.as_str() {
                    "numeric" => Ok(AregVariable::Numeric),
                    "linear" => Ok(AregVariable::Linear),
                    "categorical" => Ok(AregVariable::Categorical),
                    _ => Err(format!(
                        "kinds: \"{k}\" is not \"numeric\", \"linear\" or \"categorical\""
                    )),
                })
                .collect::<Result<_, _>>()?,
        );
    }
    Ok(a.with_seed(o.seed.unwrap_or(1)))
}

/// Multiply impute `n_rows × n_cols` row-major `x` (`NaN` = missing). `out`
/// receives `n_impute` completed copies one after the other
/// (`n_impute × n_rows × n_cols`), `n_impute` their count.
///
/// # Safety
/// `options` is NUL-terminated or NULL; `x` holds `n_rows * n_cols`
/// doubles; `out` / `n_impute` are writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_multiple_impute(
    options: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    out: *mut AnofoxMlDoubles,
    n_impute: *mut usize,
) -> *mut c_char {
    let run = || -> Result<(Vec<f64>, usize), String> {
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        let x = matrix(x, n_rows, n_cols)?;
        let r = imputer(options, n_cols)?
            .impute(x)
            .map_err(|e| e.to_string())?;
        let m = r.datasets.len();
        let v = r
            .datasets
            .iter()
            .flat_map(|d| d.as_standard_layout().iter().copied().collect::<Vec<_>>())
            .collect();
        Ok((v, m))
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok((v, m))) => {
            *out = into_doubles(v);
            *n_impute = m;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while imputing"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn fills_only_the_holes_and_replays_the_seed() {
        let mut x = Array2::from_shape_fn((40, 2), |(i, j)| {
            if j == 0 {
                i as f64
            } else {
                2.0 * i as f64 + (i % 3) as f64
            }
        });
        for i in [3, 11, 25] {
            x[[i, 1]] = f64::NAN;
        }
        let a = imputer(r#"{"n_impute": 3}"#, 2)
            .unwrap()
            .impute(&x)
            .unwrap();
        assert_eq!(a.datasets.len(), 3);
        for d in &a.datasets {
            assert!(d.iter().all(|v| v.is_finite()));
            for i in 0..40 {
                if !x[[i, 1]].is_nan() {
                    assert_eq!(d[[i, 1]], x[[i, 1]]);
                }
            }
        }
        let b = imputer(r#"{"n_impute": 3}"#, 2)
            .unwrap()
            .impute(&x)
            .unwrap();
        assert_eq!(a.datasets, b.datasets);
        assert!(imputer(r#"{"kinds": ["numeric"]}"#, 2)
            .unwrap_err()
            .contains("2 columns"));
    }
}
