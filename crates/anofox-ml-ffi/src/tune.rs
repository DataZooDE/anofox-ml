//! Hyperparameter search behind `ml_tune_agg`: grid or seeded random search
//! over a JSON search space, every candidate scored by k-fold
//! cross-validation with a metric of the metric aggregates, the best one
//! refitted on all rows.
//!
//! Candidates are generated up front from the seed and each
//! (candidate, fold) is scored independently, so the result does not depend
//! on how many threads score them.

use std::ffi::c_char;

use anofox_ml_core::utils as ut;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::{
    encode_labels, err, fit_impl, fit_model, matrix, metrics, run_fit, str_arg, strings,
    AnofoxMlBytes,
};

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Options {
    /// `"random"` (default) or `"grid"`.
    search: Option<String>,
    /// Candidates of a random search (default 20).
    n_trials: Option<usize>,
    /// Folds (default 5).
    cv: Option<usize>,
    /// Stratify the folds on the target (default: for classifiers).
    stratify: Option<bool>,
    /// Metric name of the metric aggregates (default: accuracy for
    /// classifiers, r2 otherwise); losses are minimised.
    metric: Option<String>,
    metric_options: Option<Value>,
    /// Parameters fixed for every candidate.
    params: Option<Map<String, Value>>,
    seed: Option<u64>,
}

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

/// One dimension of the search space.
enum Dim {
    Int { lo: i64, hi: i64, log: bool },
    Float { lo: f64, hi: f64, log: bool },
    Choice(Vec<Value>),
}

fn parse_space(space: &str) -> Result<Vec<(String, Dim)>, String> {
    let v: Value =
        serde_json::from_str(space).map_err(|e| format!("invalid search space JSON: {e}"))?;
    let obj = v
        .as_object()
        .ok_or("the search space must be a JSON object")?;
    if obj.is_empty() {
        return Err("the search space is empty".into());
    }
    obj.iter()
        .map(|(name, d)| {
            let bad = || format!("search space '{name}': expected a list of values, {{\"int\": [lo, hi]}}, {{\"float\": [lo, hi]}} or {{\"categorical\": [...]}}");
            let dim = match d {
                Value::Array(vals) if !vals.is_empty() => Dim::Choice(vals.clone()),
                Value::Object(o) => {
                    let log = o.get("log").and_then(Value::as_bool).unwrap_or(false);
                    if let Some(Value::Array(r)) = o.get("int") {
                        let (lo, hi) = (r.first().and_then(Value::as_i64), r.get(1).and_then(Value::as_i64));
                        match (lo, hi) {
                            (Some(lo), Some(hi)) if lo <= hi && (!log || lo > 0) => Dim::Int { lo, hi, log },
                            _ => return Err(bad()),
                        }
                    } else if let Some(Value::Array(r)) = o.get("float") {
                        let (lo, hi) = (r.first().and_then(Value::as_f64), r.get(1).and_then(Value::as_f64));
                        match (lo, hi) {
                            (Some(lo), Some(hi)) if lo <= hi && (!log || lo > 0.0) => Dim::Float { lo, hi, log },
                            _ => return Err(bad()),
                        }
                    } else if let Some(Value::Array(c)) = o.get("categorical") {
                        if c.is_empty() {
                            return Err(bad());
                        }
                        Dim::Choice(c.clone())
                    } else {
                        return Err(bad());
                    }
                }
                _ => return Err(bad()),
            };
            Ok((name.clone(), dim))
        })
        .collect()
}

fn sample(dim: &Dim, rng: &mut StdRng) -> Value {
    match dim {
        Dim::Int { lo, hi, log: false } => json!(rng.gen_range(*lo..=*hi)),
        Dim::Int { lo, hi, log: true } => {
            let v = rng
                .gen_range((*lo as f64).ln()..=(*hi as f64 + 1.0 - 1e-9).ln())
                .exp()
                .floor() as i64;
            json!(v.clamp(*lo, *hi))
        }
        Dim::Float { lo, hi, log: false } => json!(if lo == hi {
            *lo
        } else {
            rng.gen_range(*lo..*hi)
        }),
        Dim::Float { lo, hi, log: true } => json!(if lo == hi {
            *lo
        } else {
            rng.gen_range(lo.ln()..hi.ln()).exp()
        }),
        Dim::Choice(c) => c[rng.gen_range(0..c.len())].clone(),
    }
}

/// Candidate parameter sets (without the fixed params).
fn candidates(space: &[(String, Dim)], o: &Options) -> Result<Vec<Map<String, Value>>, String> {
    match o.search.as_deref().unwrap_or("random") {
        "grid" => {
            let mut out = vec![Map::new()];
            for (name, dim) in space {
                let Dim::Choice(vals) = dim else {
                    return Err(format!("grid search needs a list of values for '{name}'"));
                };
                out = out
                    .into_iter()
                    .flat_map(|c| {
                        vals.iter().map(move |v| {
                            let mut c = c.clone();
                            c.insert(name.clone(), v.clone());
                            c
                        })
                    })
                    .collect();
            }
            Ok(out)
        }
        "random" => {
            let mut rng = StdRng::seed_from_u64(o.seed.unwrap_or(0));
            Ok((0..o.n_trials.unwrap_or(20).max(1))
                .map(|_| {
                    space
                        .iter()
                        .map(|(n, d)| (n.clone(), sample(d, &mut rng)))
                        .collect()
                })
                .collect())
        }
        s => Err(format!(
            "search must be \"random\" or \"grid\", got \"{s}\""
        )),
    }
}

#[derive(Debug)]
pub(crate) struct TuneResult {
    pub best_params: Value,
    pub best_score: f64,
    pub model: Vec<u8>,
    pub trials: Value,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tune(
    kind: &str,
    space: &str,
    options: &str,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    labels: Option<Vec<String>>,
    threads: usize,
) -> Result<TuneResult, String> {
    let o: Options = if options.trim().is_empty() {
        Options::default()
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    let space = parse_space(space)?;
    let cands = candidates(&space, &o)?;
    let probe = anofox_ml_registry::make(kind, &json!({})).map_err(|e| e.to_string())?;
    let classifier = labels.is_some() || probe.capabilities().contains(&"predict_proba");
    let metric = o
        .metric
        .clone()
        .unwrap_or_else(|| if classifier { "accuracy" } else { "r2" }.into());
    let minimise = LOSSES.contains(&metric.as_str());
    let proba = matches!(
        metric.as_str(),
        "roc_auc" | "log_loss" | "brier_score" | "average_precision"
    );
    let k = o.cv.unwrap_or(5);
    let seed = o.seed.unwrap_or(0);
    let folds = if o.stratify.unwrap_or(classifier) {
        ut::stratified_k_fold(x, y, k, seed)
    } else {
        ut::shuffled_k_fold(x.nrows(), k, seed)
    }
    .map_err(|e| e.to_string())?;
    // Classes of the whole training set: the column order of probabilities
    // and the labels of log_loss, the same in every fold.
    let mut classes: Vec<f64> = y.to_vec();
    classes.sort_by(f64::total_cmp);
    classes.dedup();
    let mut metric_options = o.metric_options.clone().unwrap_or(json!({}));
    if metric == "log_loss" {
        if let Some(m) = metric_options.as_object_mut() {
            m.entry("labels").or_insert(json!(classes));
        }
    }
    let metric_options = metric_options.to_string();
    let fixed = o.params.clone().unwrap_or_default();
    let full_params: Vec<Map<String, Value>> = cands
        .iter()
        .map(|c| {
            let mut p = fixed.clone();
            p.extend(c.clone());
            p
        })
        .collect();

    // Score one (candidate, fold); each fit runs sequentially inside the
    // parallel loop.
    let score_fold =
        |p: &Map<String, Value>, (train, test): &(Vec<usize>, Vec<usize>)| -> Result<f64, String> {
            let (xtr, ytr) = (x.select(Axis(0), train), y.select(Axis(0), train));
            let (xte, yte) = (x.select(Axis(0), test), y.select(Axis(0), test));
            let params = Value::Object(p.clone()).to_string();
            let m = fit_model(kind, &params, xtr.view(), Some(ytr.view()), None, 1)?;
            let pred = if proba {
                let out = m.predict_proba(xte.view()).map_err(|e| e.to_string())?;
                let crate::Out::A2(p) = out else {
                    return Err("predict_proba returned a vector".into());
                };
                // Align the fold model's classes with the full set (a fold may miss one).
                let mut fold_classes: Vec<f64> = ytr.to_vec();
                fold_classes.sort_by(f64::total_cmp);
                fold_classes.dedup();
                let mut full = Array2::<f64>::zeros((p.nrows(), classes.len()));
                for (j, c) in fold_classes.iter().enumerate() {
                    let col = classes
                        .iter()
                        .position(|v| v == c)
                        .expect("fold class in the full set");
                    full.column_mut(col).assign(&p.column(j));
                }
                if classes.len() == 2 && metric != "log_loss" {
                    full.column(1).to_owned().insert_axis(Axis(1))
                } else {
                    full
                }
            } else {
                let out = m.predict(xte.view()).map_err(|e| e.to_string())?;
                let crate::Out::A1(p) = out else {
                    return Err("predict returned a matrix".into());
                };
                p.insert_axis(Axis(1))
            };
            metrics::metric(
                &metric,
                &metric_options,
                Array1::from(yte.to_vec()),
                pred,
                None,
            )
        };
    let jobs: Vec<(usize, usize)> = (0..full_params.len())
        .flat_map(|c| (0..folds.len()).map(move |f| (c, f)))
        .collect();
    let scores: Vec<Result<f64, String>> = run_fit(kind, threads, || {
        jobs.par_iter()
            .map(|&(c, f)| score_fold(&full_params[c], &folds[f]))
            .collect()
    })?;

    let mut trials = Vec::with_capacity(full_params.len());
    let mut best: Option<(usize, f64)> = None;
    for c in 0..full_params.len() {
        let fold_scores: Result<Vec<f64>, String> = scores[c * folds.len()..(c + 1) * folds.len()]
            .iter()
            .cloned()
            .collect();
        match fold_scores {
            Ok(s) => {
                let mean = s.iter().sum::<f64>() / s.len() as f64;
                let std =
                    (s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
                trials
                    .push(json!({"trial": c, "params": cands[c], "score": mean, "score_std": std}));
                let better = match best {
                    None => mean.is_finite(),
                    Some((_, b)) => mean.is_finite() && if minimise { mean < b } else { mean > b },
                };
                if better {
                    best = Some((c, mean));
                }
            }
            Err(e) => trials.push(json!({"trial": c, "params": cands[c], "error": e})),
        }
    }
    let (bi, best_score) = best.ok_or_else(|| {
        let first = trials
            .iter()
            .find_map(|t| t.get("error"))
            .cloned()
            .unwrap_or_default();
        format!("every trial failed (first error: {first})")
    })?;
    let params = Value::Object(full_params[bi].clone()).to_string();
    let model = fit_impl(kind, &params, x, Some(y), None, labels, threads)?;
    Ok(TuneResult {
        best_params: Value::Object(full_params[bi].clone()),
        best_score,
        model,
        trials: Value::Array(trials),
    })
}

#[repr(C)]
pub struct AnofoxMlTuneResult {
    pub best_params: *mut c_char,
    pub best_score: f64,
    pub model: AnofoxMlBytes,
    pub trials: *mut c_char,
}

/// Tune `kind` on `n_rows × n_cols` rows: `y` holds doubles, or `y_labels`
/// strings (the other NULL). `space` / `options` are JSON (`options` may be
/// NULL). Release `out` with [`anofox_ml_free_tune_result`].
///
/// # Safety
/// Strings are NUL-terminated; arrays hold the stated number of values;
/// `out` is writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_tune(
    kind: *const c_char,
    space: *const c_char,
    options: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const f64,
    y_labels: *const *const c_char,
    threads: usize,
    out: *mut AnofoxMlTuneResult,
) -> *mut c_char {
    let run = || -> Result<TuneResult, String> {
        let kind = str_arg(kind, "kind")?;
        let space = str_arg(space, "search space")?;
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        let x = matrix(x, n_rows, n_cols)?;
        if !y_labels.is_null() {
            let (labels, codes) = encode_labels(&strings(y_labels, n_rows, "y")?);
            tune(
                kind,
                space,
                options,
                x,
                ArrayView1::from(&codes),
                Some(labels),
                threads,
            )
        } else if !y.is_null() {
            let y = ArrayView1::from(std::slice::from_raw_parts(y, n_rows));
            tune(kind, space, options, x, y, None, threads)
        } else {
            Err("y is NULL".into())
        }
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(r)) => {
            let mut b = r.model.into_boxed_slice();
            *out = AnofoxMlTuneResult {
                best_params: crate::cstring(r.best_params.to_string()),
                best_score: r.best_score,
                model: AnofoxMlBytes {
                    data: b.as_mut_ptr(),
                    len: b.len(),
                },
                trials: crate::cstring(r.trials.to_string()),
            };
            std::mem::forget(b);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while tuning"),
    }
}

/// # Safety
/// `r` was filled by [`anofox_ml_tune`].
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_free_tune_result(r: AnofoxMlTuneResult) {
    crate::anofox_ml_free_string(r.best_params);
    crate::anofox_ml_free_string(r.trials);
    crate::anofox_ml_free_bytes(r.model);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y = step at x = 5 plus a little noise: a stump (max_depth 1) fits it,
    /// deeper trees only add variance; max_depth 0 cannot.
    fn data() -> (Array2<f64>, Array1<f64>) {
        let n = 200;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| (i % 10) as f64);
        let y = Array1::from_shape_fn(
            n,
            |i| if i % 10 >= 5 { 10.0 } else { 0.0 } + ((i * 37) % 7) as f64 / 10.0,
        );
        (x, y)
    }

    #[test]
    fn grid_search_finds_the_needed_depth_and_is_thread_independent() {
        let (x, y) = data();
        let r1 = tune(
            "DecisionTreeRegressor",
            r#"{"max_depth": [0, 1]}"#,
            r#"{"search": "grid", "cv": 4}"#,
            x.view(),
            y.view(),
            None,
            1,
        )
        .unwrap();
        assert_eq!(r1.best_params["max_depth"], 1, "{}", r1.trials);
        assert_eq!(r1.trials.as_array().unwrap().len(), 2);
        let r8 = tune(
            "DecisionTreeRegressor",
            r#"{"max_depth": [0, 1]}"#,
            r#"{"search": "grid", "cv": 4}"#,
            x.view(),
            y.view(),
            None,
            8,
        )
        .unwrap();
        assert_eq!(r1.trials, r8.trials);
        assert_eq!(r1.model, r8.model);
    }

    #[test]
    fn random_search_is_seeded_and_records_failures() {
        let (x, y) = data();
        let space =
            r#"{"max_depth": {"int": [1, 6]}, "min_samples_leaf": {"int": [1, 50], "log": true}}"#;
        let a = tune(
            "DecisionTreeRegressor",
            space,
            r#"{"n_trials": 6, "seed": 3, "metric": "rmse"}"#,
            x.view(),
            y.view(),
            None,
            4,
        )
        .unwrap();
        let b = tune(
            "DecisionTreeRegressor",
            space,
            r#"{"n_trials": 6, "seed": 3, "metric": "rmse"}"#,
            x.view(),
            y.view(),
            None,
            1,
        )
        .unwrap();
        assert_eq!(a.trials, b.trials);
        assert!(a.best_score < 1.0, "rmse is minimised: {}", a.best_score);
        // An invalid value becomes a failed trial, not an error.
        let r = tune(
            "DecisionTreeRegressor",
            r#"{"min_samples_leaf": [1, -5]}"#,
            r#"{"search": "grid"}"#,
            x.view(),
            y.view(),
            None,
            1,
        )
        .unwrap();
        assert!(r.trials[1].get("error").is_some(), "{}", r.trials);
        assert!(tune(
            "DecisionTreeRegressor",
            r#"{"min_samples_leaf": [-5]}"#,
            r#"{"search": "grid"}"#,
            x.view(),
            y.view(),
            None,
            1
        )
        .unwrap_err()
        .contains("every trial failed"));
        assert!(parse_space(r#"{"a": {"int": [5, 1]}}"#).is_err());
    }
}
