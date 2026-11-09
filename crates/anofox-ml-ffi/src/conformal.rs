//! Split conformal prediction on a fitted model BLOB: calibration scores are
//! stored in the BLOB metadata, so intervals / sets work for any `alpha`.
//!
//! - Regressors: score `|y - ŷ|`; interval `ŷ ± q`.
//! - Classifiers with `predict_proba`: score `1 - p(true class)` (LAC); the
//!   set holds every class with `1 - p ≤ q`.
//!
//! `q` is the finite-sample conformal quantile of the scores
//! (`anofox_ml_conformal::conformal_quantile`); with too few calibration
//! rows for `alpha` it is unbounded (infinite interval, every class).
//!
//! Models of the registry's conformal kinds (`SplitConformalRegressor`,
//! `CrossConformalRegressor`, `ConformalizedQuantileRegressor`,
//! `MondrianConformalClassifier`, …) calibrate themselves when fitted; when
//! such a model has no calibration scores of its own here, its
//! `predict_interval` / `predict_set` method answers.

use std::ffi::c_char;

use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};

use crate::{
    apply_impl, decode, encode, err, flatten, into_doubles, into_strings, label_of, matrix,
    strings, with_model, AnofoxMlBytes, AnofoxMlDoubles, AnofoxMlStrings, POOL,
};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Score {
    /// `|y - ŷ|` of a regressor.
    AbsoluteResidual,
    /// `1 - p(true class)` of a classifier.
    Lac,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct Conformity {
    pub score: Score,
    pub scores: Vec<f64>,
}

fn quantile(c: &Conformity, alpha: f64) -> Result<f64, String> {
    anofox_ml_conformal::conformal_quantile(&c.scores, alpha)
        .map(|q| q.unwrap_or(f64::INFINITY))
        .map_err(|e| e.to_string())
}

/// Calibrate `blob` on `n_rows` held-out rows; returns the new BLOB.
fn conformalize(
    blob: &[u8],
    x: ndarray::ArrayView2<f64>,
    y: ArrayView1<f64>,
) -> Result<Vec<u8>, String> {
    let (mut meta, bytes) = decode(blob)?;
    let (score, scores) = match &meta.classes {
        Some(classes) => {
            let (p, k) = apply_impl(blob, "predict_proba", x)?;
            let scores = y
                .iter()
                .enumerate()
                .map(|(r, &t)| {
                    let j = classes.iter().position(|&c| c == t).ok_or_else(|| {
                        format!("calibration label {t} is not a class of the model")
                    })?;
                    Ok(1.0 - p[r * k + j])
                })
                .collect::<Result<Vec<f64>, String>>()?;
            (Score::Lac, scores)
        }
        None => {
            let (p, k) = apply_impl(blob, "predict", x)?;
            if k != 1 {
                return Err(format!("{} predicts {k} values per row", meta.kind));
            }
            (
                Score::AbsoluteResidual,
                y.iter().zip(&p).map(|(t, p)| (t - p).abs()).collect(),
            )
        }
    };
    if scores.iter().any(|s| s.is_nan()) {
        return Err("calibration scores contain NaN".into());
    }
    meta.conformity = Some(Conformity { score, scores });
    Ok(encode(&meta, bytes))
}

/// `method` (`predict_interval` / `predict_set`) of a conformal-kind model
/// that has no calibration scores from `ml_conformalize_agg`: the
/// row-major output and its column count, or `None` for other models.
fn native(
    blob: &[u8],
    method: &str,
    x: ndarray::ArrayView2<f64>,
    alpha: f64,
) -> Result<Option<(Vec<f64>, usize)>, String> {
    if decode(blob)?.0.conformity.is_some() {
        return Ok(None);
    }
    with_model(blob, |meta, model| {
        if !model.capabilities().contains(&method) {
            return Ok(None);
        }
        if x.ncols() != meta.n_features {
            return Err(format!(
                "{} was fitted on {} features, got {}",
                meta.kind,
                meta.n_features,
                x.ncols()
            ));
        }
        let out = POOL.with(|p| {
            p.install(|| match method {
                "predict_interval" => model.predict_interval(x, alpha),
                _ => model.predict_set(x, alpha),
            })
        });
        let out = out.map_err(|e| {
            format!(
                "{e} (or calibrate the model with ml_conformalize_agg, or fit a conformal kind)"
            )
        })?;
        Ok(Some(flatten(out)))
    })
}

fn conformity_of(blob: &[u8], want: Score) -> Result<Conformity, String> {
    let (meta, _) = decode(blob)?;
    let c = meta.conformity.ok_or(
        "the model is not conformalized; calibrate it with ml_conformalize_agg, \
         or fit a conformal kind such as SplitConformalRegressor",
    )?;
    if c.score != want {
        return Err(match want {
            Score::AbsoluteResidual => "prediction intervals need a regressor; use ml_predict_set",
            Score::Lac => "prediction sets need a classifier; use ml_predict_interval",
        }
        .into());
    }
    Ok(c)
}

/// Calibrate a model BLOB on `n_rows` rows of `x` and `y`: `out` receives a
/// copy of the model with its conformity scores.
///
/// # Safety
/// `model` holds `model_len` bytes, `x` `n_rows * n_cols` doubles, `y`
/// `n_rows` doubles; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_conformalize(
    model: *const u8,
    model_len: usize,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const f64,
    out: *mut AnofoxMlBytes,
) -> *mut c_char {
    let run = || -> Result<Vec<u8>, String> {
        if model.is_null() || y.is_null() {
            return Err("model or y is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        let y = ArrayView1::from(std::slice::from_raw_parts(y, n_rows));
        conformalize(blob, matrix(x, n_rows, n_cols)?, y)
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(bytes)) => {
            let mut b = bytes.into_boxed_slice();
            *out = AnofoxMlBytes {
                data: b.as_mut_ptr(),
                len: b.len(),
            };
            std::mem::forget(b);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while calibrating"),
    }
}

/// [`anofox_ml_conformalize`] for a model fitted on string labels: `y`
/// holds `n_rows` NUL-terminated labels.
///
/// # Safety
/// As [`anofox_ml_conformalize`], with `y` as strings.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_conformalize_labels(
    model: *const u8,
    model_len: usize,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const *const c_char,
    out: *mut AnofoxMlBytes,
) -> *mut c_char {
    let run = || -> Result<Vec<f64>, String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let (meta, _) = decode(std::slice::from_raw_parts(model, model_len))?;
        let labels = meta
            .labels
            .ok_or("the model was not fitted on string labels")?;
        strings(y, n_rows, "y")?
            .iter()
            .map(|l| {
                labels
                    .iter()
                    .position(|m| m == l)
                    .map(|i| i as f64)
                    .ok_or_else(|| format!("calibration label '{l}' is not a label of the model"))
            })
            .collect()
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(codes)) => {
            anofox_ml_conformalize(model, model_len, x, n_rows, n_cols, codes.as_ptr(), out)
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while calibrating"),
    }
}

/// Conformal prediction intervals at miscoverage `alpha`: `out` receives
/// `n_rows × 2` doubles (lower, upper), row-major.
///
/// # Safety
/// As [`crate::anofox_ml_predict`].
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_predict_interval(
    model: *const u8,
    model_len: usize,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    alpha: f64,
    out: *mut AnofoxMlDoubles,
) -> *mut c_char {
    let run = || -> Result<Vec<f64>, String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        let x = matrix(x, n_rows, n_cols)?;
        if let Some((v, _)) = native(blob, "predict_interval", x, alpha)? {
            return Ok(v);
        }
        let q = quantile(&conformity_of(blob, Score::AbsoluteResidual)?, alpha)?;
        let (p, _) = apply_impl(blob, "predict", x)?;
        Ok(p.iter().flat_map(|&p| [p - q, p + q]).collect())
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(v)) => {
            *out = into_doubles(v);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while predicting"),
    }
}

/// Conformal prediction sets at miscoverage `alpha`: `labels` receives the
/// members of every row's set (as text, see `ml_predict_label`), one row
/// after the other, and `lengths` the set size of each row.
///
/// # Safety
/// As [`crate::anofox_ml_predict`]; release the outputs with
/// `anofox_ml_free_strings` / `anofox_ml_free_doubles`.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_predict_set(
    model: *const u8,
    model_len: usize,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    alpha: f64,
    labels: *mut AnofoxMlStrings,
    lengths: *mut AnofoxMlDoubles,
) -> *mut c_char {
    let run = || -> Result<(Vec<String>, Vec<f64>), String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        let x = matrix(x, n_rows, n_cols)?;
        let (meta, _) = decode(blob)?;
        let (mut members, mut lens) = (Vec::new(), Vec::with_capacity(n_rows));
        if let Some((sets, k)) = native(blob, "predict_set", x, alpha)? {
            let classes = meta.classes.as_ref().ok_or("the model has no classes")?;
            // n × classes membership (1 = in the set), columns in class order.
            if k != classes.len() {
                return Err(format!(
                    "{} returned sets over {k} classes, the model has {}",
                    meta.kind,
                    classes.len()
                ));
            }
            for row in sets.chunks(k) {
                let before = members.len();
                for (j, &m) in row.iter().enumerate() {
                    if m > 0.5 {
                        members.push(label_of(&meta, classes[j])?);
                    }
                }
                lens.push((members.len() - before) as f64);
            }
            return Ok((members, lens));
        }
        let q = quantile(&conformity_of(blob, Score::Lac)?, alpha)?;
        let classes = meta.classes.as_ref().ok_or("the model has no classes")?;
        let (p, k) = apply_impl(blob, "predict_proba", x)?;
        for row in p.chunks(k) {
            let before = members.len();
            for (j, &pj) in row.iter().enumerate() {
                if 1.0 - pj <= q {
                    members.push(label_of(&meta, classes[j])?);
                }
            }
            lens.push((members.len() - before) as f64);
        }
        Ok((members, lens))
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok((m, l))) => {
            *labels = into_strings(&m);
            *lengths = into_doubles(l);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while predicting"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit_impl;
    use ndarray::Array2;

    #[test]
    fn intervals_cover_and_widen_with_confidence() {
        // y = x + noise in [-1, 1]; calibrate on 99 rows.
        let n = 99;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let noise = |i: usize| ((i * 37) % 21) as f64 / 10.0 - 1.0;
        let y: Vec<f64> = (0..n).map(|i| x[i] + noise(i)).collect();
        let xv = Array2::from_shape_vec((n, 1), x.clone()).unwrap();
        let blob = fit_impl(
            "OlsRegressor",
            "",
            xv.view(),
            Some(ArrayView1::from(&y)),
            None,
            None,
            1,
        )
        .unwrap();
        let cal = conformalize(&blob, xv.view(), ArrayView1::from(&y)).unwrap();
        let c = conformity_of(&cal, Score::AbsoluteResidual).unwrap();
        let (q10, q50) = (quantile(&c, 0.1).unwrap(), quantile(&c, 0.5).unwrap());
        assert!(q10 > q50 && q10 <= 1.0 + 1e-9, "{q10} {q50}");
        assert_eq!(quantile(&c, 0.001).unwrap(), f64::INFINITY);
        assert!(conformity_of(&cal, Score::Lac)
            .unwrap_err()
            .contains("classifier"));
        assert!(conformity_of(&blob, Score::AbsoluteResidual)
            .unwrap_err()
            .contains("conformalize"));
    }
}
