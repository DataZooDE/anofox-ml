//! Recursive feature elimination (scikit-learn's `RFE`): fit, rank the
//! features by importance (`feature_importances`, or the absolute
//! coefficients summed over classes), drop the `step` weakest, repeat until
//! `n_features_to_select` are left; then fit on the selected features.

use std::ffi::c_char;

use ndarray::{ArrayView1, ArrayView2, Axis};
use serde::Deserialize;

use crate::{
    encode_fitted, encode_labels, err, fit_model, fitted_array, into_doubles, matrix, str_arg,
    strings, AnofoxMlBytes, AnofoxMlDoubles, DynModel,
};

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RfeOptions {
    /// Features to keep: a count, or a fraction in (0, 1) (default: half).
    n_features_to_select: Option<f64>,
    /// Features dropped per round: a count, or a fraction in (0, 1) of all
    /// features (default 1).
    step: Option<f64>,
}

/// Importance of each of the `d` features of a fitted model.
fn importances(kind: &str, model: &dyn DynModel, d: usize) -> Result<Vec<f64>, String> {
    if model.capabilities().contains(&"feature_importances") {
        let (v, _) = crate::flatten(model.feature_importances().map_err(|e| e.to_string())?);
        return Ok(v);
    }
    let json: serde_json::Value =
        serde_json::from_str(&model.fitted_json().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let (_, _, coef) = fitted_array(&json, &["coef", "weights"]).ok_or_else(|| {
        format!("{kind} has neither feature_importances nor coefficients, which RFE needs")
    })?;
    if coef.len() % d != 0 {
        return Err(format!(
            "{kind}: {} coefficients for {d} features",
            coef.len()
        ));
    }
    // Rows are classes: the L1 norm over classes, as scikit-learn.
    let mut imp = vec![0.0; d];
    for (i, c) in coef.iter().enumerate() {
        imp[i % d] += c.abs();
    }
    Ok(imp)
}

/// `(support, ranking, BLOB of the model on the selected features)`.
type Selection = (Vec<bool>, Vec<usize>, Vec<u8>);

fn rfe(
    kind: &str,
    params: &str,
    options: &str,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    labels: Option<Vec<String>>,
    threads: usize,
) -> Result<Selection, String> {
    let o: RfeOptions = if options.trim().is_empty() {
        RfeOptions::default()
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    let d = x.ncols();
    let count = |v: f64, what: &str| -> Result<usize, String> {
        if v > 0.0 && v < 1.0 {
            Ok(((v * d as f64) as usize).max(1))
        } else if v >= 1.0 && v.fract() == 0.0 {
            Ok(v as usize)
        } else {
            Err(format!(
                "{what} must be a positive count or a fraction in (0, 1), got {v}"
            ))
        }
    };
    let keep = match o.n_features_to_select {
        Some(v) => count(v, "n_features_to_select")?.min(d),
        None => (d / 2).max(1),
    };
    let step = count(o.step.unwrap_or(1.0), "step")?;
    let mut support = vec![true; d];
    let mut ranking = vec![1usize; d];
    loop {
        let features: Vec<usize> = (0..d).filter(|&j| support[j]).collect();
        if features.len() <= keep {
            break;
        }
        let xs = x.select(Axis(1), &features);
        let model = fit_model(kind, params, xs.view(), Some(y), None, threads)?;
        let imp = importances(kind, model.as_ref(), features.len())?;
        // Weakest first; ties keep column order (a stable sort).
        let mut order: Vec<usize> = (0..features.len()).collect();
        order.sort_by(|&a, &b| imp[a].total_cmp(&imp[b]));
        for &k in order.iter().take(step.min(features.len() - keep)) {
            support[features[k]] = false;
        }
        for j in 0..d {
            if !support[j] {
                ranking[j] += 1;
            }
        }
    }
    let features: Vec<usize> = (0..d).filter(|&j| support[j]).collect();
    let xs = x.select(Axis(1), &features);
    let model = fit_model(kind, params, xs.view(), Some(y), None, threads)?;
    let blob = encode_fitted(kind, model.as_ref(), features.len(), Some(y), labels)?;
    Ok((support, ranking, blob))
}

#[repr(C)]
pub struct AnofoxMlRfeResult {
    /// `n_cols` values: 1 for a selected feature.
    pub support: AnofoxMlDoubles,
    /// `n_cols` values: 1 for selected features, then 2, 3, … in the order
    /// they were dropped (last dropped = 2).
    pub ranking: AnofoxMlDoubles,
    /// The model fitted on the selected features.
    pub model: AnofoxMlBytes,
}

/// Recursive feature elimination of `kind` on `n_rows × n_cols` `x` with
/// targets `y`, or string labels `y_labels` when `y` is NULL.
///
/// # Safety
/// As `anofox_ml_fit`; `options` may be NULL. Release `out` with
/// [`anofox_ml_free_rfe_result`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_rfe(
    kind: *const c_char,
    params: *const c_char,
    options: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const f64,
    y_labels: *const *const c_char,
    threads: usize,
    out: *mut AnofoxMlRfeResult,
) -> *mut c_char {
    let run = || -> Result<Selection, String> {
        let kind = str_arg(kind, "kind")?;
        let opt = |p: *const c_char, what| {
            if p.is_null() {
                Ok("")
            } else {
                str_arg(p, what)
            }
        };
        let (params, options) = (opt(params, "params")?, opt(options, "options")?);
        let x = matrix(x, n_rows, n_cols)?;
        if y.is_null() {
            let (labels, codes) = encode_labels(&strings(y_labels, n_rows, "y")?);
            rfe(
                kind,
                params,
                options,
                x,
                ArrayView1::from(&codes),
                Some(labels),
                threads,
            )
        } else {
            let y = ArrayView1::from(std::slice::from_raw_parts(y, n_rows));
            rfe(kind, params, options, x, y, None, threads)
        }
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok((support, ranking, blob))) => {
            let mut b = blob.into_boxed_slice();
            *out = AnofoxMlRfeResult {
                support: into_doubles(support.iter().map(|&s| s as u8 as f64).collect()),
                ranking: into_doubles(ranking.iter().map(|&r| r as f64).collect()),
                model: AnofoxMlBytes {
                    data: b.as_mut_ptr(),
                    len: b.len(),
                },
            };
            std::mem::forget(b);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(p) => err(crate::panic_message("during feature elimination", p)),
    }
}

/// # Safety
/// `r` was filled by [`anofox_ml_rfe`].
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_free_rfe_result(r: AnofoxMlRfeResult) {
    crate::anofox_ml_free_doubles(r.support);
    crate::anofox_ml_free_doubles(r.ranking);
    crate::anofox_ml_free_bytes(r.model);
}
