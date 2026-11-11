//! Permutation feature importance of a model BLOB on a table of rows
//! (scikit-learn `permutation_importance`): the drop in a metric when one
//! feature column is shuffled, averaged over `n_repeats` shuffles.

use std::ffi::c_char;

use ndarray::{Array1, Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Deserialize;

use crate::{
    apply_impl, decode, err, into_doubles, matrix, metrics, str_arg, strings, AnofoxMlDoubles,
};

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Options {
    /// Metric name as in the metric aggregates (default: accuracy for
    /// classifiers, r2 otherwise).
    metric: Option<String>,
    /// Options of the metric (average, pos_label, ...).
    metric_options: Option<serde_json::Value>,
    n_repeats: Option<usize>,
    seed: Option<u64>,
}

/// Metrics where lower is better: their drop is negated, so a positive
/// importance always means the feature helps.
const LOSSES: &[&str] = &[
    "mse",
    "rmse",
    "mae",
    "mape",
    "median_absolute_error",
    "max_error",
    "mean_squared_log_error",
    "log_loss",
    "brier_score",
];

/// Metrics computed on probabilities rather than predictions.
const ON_PROBA: &[&str] = &["roc_auc", "log_loss", "brier_score", "average_precision"];

/// `(mean, std)` importance of every feature.
fn importance(
    blob: &[u8],
    x: ArrayView2<f64>,
    y: Array1<f64>,
    options: &str,
) -> Result<Vec<(f64, f64)>, String> {
    let o: Options = if options.trim().is_empty() {
        Options::default()
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    let (meta, _) = decode(blob)?;
    let classifier = meta.classes.is_some() || meta.labels.is_some();
    let metric = o
        .metric
        .clone()
        .unwrap_or_else(|| if classifier { "accuracy" } else { "r2" }.into());
    let mut metric_options = o.metric_options.clone().unwrap_or(serde_json::json!({}));
    let proba = ON_PROBA.contains(&metric.as_str());
    if proba && metric == "log_loss" {
        // Probability columns follow the stored classes.
        if let (Some(c), Some(obj)) = (&meta.classes, metric_options.as_object_mut()) {
            obj.entry("labels").or_insert(serde_json::json!(c));
        }
    }
    let metric_options = metric_options.to_string();
    let sign = if LOSSES.contains(&metric.as_str()) {
        -1.0
    } else {
        1.0
    };
    let score = |xm: ArrayView2<f64>| -> Result<f64, String> {
        let (p, k) = apply_impl(blob, if proba { "predict_proba" } else { "predict" }, xm)?;
        let n = xm.nrows();
        let pred = if proba && k == 2 && metric != "log_loss" {
            // binary: the score of the positive (greater) class
            Array2::from_shape_vec((n, 1), p.chunks(2).map(|r| r[1]).collect())
        } else {
            Array2::from_shape_vec((n, k), p)
        }
        .map_err(|e| e.to_string())?;
        Ok(sign * metrics::metric(&metric, &metric_options, y.clone(), pred, None)?)
    };
    let baseline = score(x)?;
    let n_repeats = o.n_repeats.unwrap_or(5).max(1);
    let seed = o.seed.unwrap_or(0);
    let mut out = Vec::with_capacity(x.ncols());
    for j in 0..x.ncols() {
        // A per-feature seed, as in anofox-ml-core's permutation_importance.
        let mut rng = StdRng::seed_from_u64(
            seed.wrapping_add((j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        let original: Vec<f64> = x.column(j).to_vec();
        let mut xp = x.to_owned();
        let mut idx: Vec<usize> = (0..x.nrows()).collect();
        let mut drops = Vec::with_capacity(n_repeats);
        for _ in 0..n_repeats {
            idx.shuffle(&mut rng);
            for (i, &p) in idx.iter().enumerate() {
                xp[[i, j]] = original[p];
            }
            drops.push(baseline - score(xp.view())?);
        }
        let mean = drops.iter().sum::<f64>() / n_repeats as f64;
        let var = drops.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n_repeats as f64;
        out.push((mean, var.sqrt()));
    }
    Ok(out)
}

/// Permutation importance of a model BLOB on `n_rows` rows: `y` holds
/// `n_rows` doubles, or `y_labels` `n_rows` strings for a model fitted on
/// string labels (pass the other as NULL). `out` receives `n_cols × 2`
/// doubles (mean, std per feature), row-major.
///
/// # Safety
/// `model` holds `model_len` bytes, `x` `n_rows * n_cols` doubles, `y` /
/// `y_labels` as described; `options` is NUL-terminated or NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_permutation_importance(
    model: *const u8,
    model_len: usize,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const f64,
    y_labels: *const *const c_char,
    options: *const c_char,
    out: *mut AnofoxMlDoubles,
) -> *mut c_char {
    let run = || -> Result<Vec<f64>, String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        let y = if !y_labels.is_null() {
            let (meta, _) = decode(blob)?;
            let labels = meta
                .labels
                .ok_or("the model was not fitted on string labels")?;
            strings(y_labels, n_rows, "y")?
                .iter()
                .map(|l| {
                    labels
                        .iter()
                        .position(|m| m == l)
                        .map(|i| i as f64)
                        .ok_or_else(|| format!("label '{l}' is not a label of the model"))
                })
                .collect::<Result<Vec<f64>, String>>()?
        } else if !y.is_null() {
            std::slice::from_raw_parts(y, n_rows).to_vec()
        } else {
            return Err("y is NULL".into());
        };
        let imp = importance(blob, matrix(x, n_rows, n_cols)?, Array1::from(y), options)?;
        Ok(imp.into_iter().flat_map(|(m, s)| [m, s]).collect())
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(v)) => {
            *out = into_doubles(v);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(p) => err(crate::panic_message("while computing importances", p)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit_impl;
    use ndarray::ArrayView1;

    #[test]
    fn noise_features_are_unimportant() {
        // y depends on feature 0 only; feature 1 is noise.
        let n = 200;
        let x: Vec<f64> = (0..n)
            .flat_map(|i| [(i % 20) as f64, ((i * 7919) % 13) as f64])
            .collect();
        let y: Vec<f64> = (0..n).map(|i| 3.0 * (i % 20) as f64).collect();
        let xv = Array2::from_shape_vec((n, 2), x).unwrap();
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
        let imp = importance(&blob, xv.view(), Array1::from(y.clone()), r#"{"seed": 1}"#).unwrap();
        assert!(imp[0].0 > 1.0, "{imp:?}");
        assert!(imp[1].0.abs() < 1e-6, "{imp:?}");
        // a loss metric is negated: still positive for the useful feature
        let imp = importance(&blob, xv.view(), Array1::from(y), r#"{"metric": "rmse"}"#).unwrap();
        assert!(imp[0].0 > 0.0 && imp[1].0.abs() < 1e-6, "{imp:?}");
    }
}
