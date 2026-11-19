//! Model selection over a registry kind, behind the `ml_*_agg` aggregates
//! of the same names: `learning_curve`, `validation_curve`,
//! `permutation_test`, `nested_cv` (anofox-ml-core's
//! `nested_cross_validate` of a grid) and `bootstrap_validate` (Efron's
//! .632 / .632+ and Harrell's optimism-corrected estimates).
//!
//! Every method scores with a metric of the metric aggregates (default:
//! accuracy for classifiers, r2 otherwise; the probability metrics use the
//! positive-class probability of a binary classifier). Results come back as
//! a [`Table`]: one text per row (parameters as JSON, or empty) and
//! row-major doubles.

use std::ffi::c_char;

use anofox_ml_core::{utils as ut, validation as va, CvSplit, Result as MlResult, RustMlError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::{
    encode_labels, err, fit_model, into_doubles, into_strings, matrix, metrics, run_fit, str_arg,
    strings, AnofoxMlDoubles, AnofoxMlStrings, Out,
};

/// Methods of [`anofox_ml_select`].
pub(crate) const METHODS: &[&str] = &[
    "learning_curve",
    "validation_curve",
    "permutation_test",
    "nested_cv",
    "bootstrap_validate",
];

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Options {
    /// Folds (default 5; nested_cv: the outer folds).
    cv: Option<usize>,
    /// nested_cv: inner folds (default 3).
    inner_cv: Option<usize>,
    /// Stratify folds / bootstrap samples on the target (default: for
    /// classifiers).
    stratify: Option<bool>,
    metric: Option<String>,
    metric_options: Option<Value>,
    seed: Option<u64>,
    /// learning_curve: training-set fractions (default 0.1, 0.325, 0.55,
    /// 0.775, 1.0).
    train_sizes: Option<Vec<f64>>,
    /// validation_curve / nested_cv: parameter sets, as a list of objects
    /// or an object of value lists (their grid).
    param_grid: Option<Value>,
    /// permutation_test (default 100).
    n_permutations: Option<usize>,
    /// bootstrap_validate (default 200).
    n_bootstraps: Option<usize>,
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

const ON_PROBA: &[&str] = &["roc_auc", "log_loss", "brier_score", "average_precision"];

/// Result rows: one text and `cols` doubles per row.
#[derive(Debug)]
pub(crate) struct Table {
    pub texts: Vec<String>,
    pub values: Vec<f64>,
    pub cols: usize,
}

/// Parameter sets of a `param_grid`: a list of objects, or an object of
/// value lists expanded to their grid (in key order).
fn grid(v: &Value) -> Result<Vec<Map<String, Value>>, String> {
    match v {
        Value::Array(sets) if !sets.is_empty() => sets
            .iter()
            .map(|s| {
                s.as_object()
                    .cloned()
                    .ok_or_else(|| format!("param_grid entries must be objects, got {s}"))
            })
            .collect(),
        Value::Object(dims) if !dims.is_empty() => {
            let mut sets = vec![Map::new()];
            for (name, values) in dims {
                let values = values
                    .as_array()
                    .filter(|a| !a.is_empty())
                    .ok_or_else(|| format!("param_grid '{name}' must be a non-empty list"))?;
                sets = sets
                    .into_iter()
                    .flat_map(|s| {
                        values.iter().map(move |v| {
                            let mut s = s.clone();
                            s.insert(name.clone(), v.clone());
                            s
                        })
                    })
                    .collect();
            }
            Ok(sets)
        }
        _ => Err("param_grid must be a non-empty list of objects or an object of lists".into()),
    }
}

/// How one candidate is fitted and scored.
struct Scoring<'a> {
    kind: &'a str,
    base: Map<String, Value>,
    metric: String,
    metric_options: String,
    proba: bool,
    /// +1 when greater is better, -1 for losses.
    sign: f64,
}

impl Scoring<'_> {
    fn params(&self, extra: &Map<String, Value>) -> String {
        let mut p = self.base.clone();
        p.extend(extra.clone());
        Value::Object(p).to_string()
    }

    /// Fit on `(x, y)` with `params` and predict `x_eval`: predictions, or
    /// the positive-class probability for the probability metrics.
    fn fit_predict(
        &self,
        params: &str,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_eval: &Array2<f64>,
    ) -> MlResult<Array1<f64>> {
        let bad = |e: String| RustMlError::InvalidParameter(e);
        let m = fit_model(self.kind, params, x.view(), Some(y.view()), None, 1).map_err(bad)?;
        if !self.proba {
            return match m.predict(x_eval.view())? {
                Out::A1(p) => Ok(p),
                Out::A2(_) => Err(bad("predict returned several columns".into())),
            };
        }
        let mut classes = y.to_vec();
        classes.sort_by(f64::total_cmp);
        classes.dedup();
        match m.predict_proba(x_eval.view())? {
            Out::A2(p) if p.ncols() == 2 => Ok(p.column(1).to_owned()),
            Out::A2(p) if p.ncols() == 1 && classes.len() == 1 => Ok(Array1::from_elem(
                p.nrows(),
                if classes[0] == 1.0 { 1.0 } else { 0.0 },
            )),
            _ => Err(bad(format!(
                "{} needs a binary classifier here",
                self.metric
            ))),
        }
    }

    /// The metric of predictions (on its own scale).
    fn score(&self, y: &Array1<f64>, p: &Array1<f64>) -> MlResult<f64> {
        let n = p.len();
        let p = p.clone().into_shape_with_order((n, 1)).expect("a column");
        metrics::metric(&self.metric, &self.metric_options, y.clone(), p, None)
            .map_err(RustMlError::InvalidParameter)
    }
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (
        mean,
        (v.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n).sqrt(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn select(
    method: &str,
    kind: &str,
    params: &str,
    options: &str,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    classifier_labels: bool,
    threads: usize,
) -> Result<Table, String> {
    if !METHODS.contains(&method) {
        return Err(format!("unknown method '{method}' (known: {METHODS:?})"));
    }
    let o: Options = if options.trim().is_empty() {
        Options::default()
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    let base: Map<String, Value> = if params.trim().is_empty() {
        Map::new()
    } else {
        match serde_json::from_str(params).map_err(|e| format!("invalid params JSON: {e}"))? {
            Value::Object(m) => m,
            v => return Err(format!("params must be a JSON object, got {v}")),
        }
    };
    let probe = crate::make_model(kind, &Value::Object(base.clone()).to_string())?;
    let classifier = classifier_labels || probe.capabilities().contains(&"predict_proba");
    let metric = o
        .metric
        .clone()
        .unwrap_or_else(|| if classifier { "accuracy" } else { "r2" }.into());
    if !metrics::METRICS.contains(&metric.as_str()) {
        return Err(format!("unknown metric '{metric}'"));
    }
    let s = Scoring {
        kind,
        base,
        proba: ON_PROBA.contains(&metric.as_str()),
        sign: if LOSSES.contains(&metric.as_str()) {
            -1.0
        } else {
            1.0
        },
        metric_options: o.metric_options.clone().unwrap_or(json!({})).to_string(),
        metric,
    };
    let (x, y) = (x.to_owned(), y.to_owned());
    let seed = o.seed.unwrap_or(0);
    let stratify = o.stratify.unwrap_or(classifier);
    let folds = |x: &Array2<f64>, y: &Array1<f64>, k: usize| -> MlResult<Vec<CvSplit>> {
        if stratify {
            ut::stratified_k_fold(x, y, k, seed)
        } else {
            ut::shuffled_k_fold(x.nrows(), k, seed)
        }
    };
    let e = |e: RustMlError| e.to_string();
    let k = o.cv.unwrap_or(5);
    let need_grid = || -> Result<Vec<Map<String, Value>>, String> {
        grid(
            o.param_grid
                .as_ref()
                .ok_or(format!("{method} needs options.param_grid"))?,
        )
    };
    let sr = &s;
    run_fit(threads, || -> Result<Table, String> {
        match method {
            "learning_curve" => {
                let sizes = o
                    .train_sizes
                    .clone()
                    .unwrap_or_else(|| vec![0.1, 0.325, 0.55, 0.775, 1.0]);
                let p = s.params(&Map::new());
                let (abs, train, test) = ut::learning_curve_with_splits(
                    &x,
                    &y,
                    &folds(&x, &y, k).map_err(e)?,
                    seed,
                    &sizes,
                    |a, b, c| s.fit_predict(&p, a, b, c),
                    |a, b| s.score(a, b),
                )
                .map_err(e)?;
                let mut values = Vec::new();
                for (i, n) in abs.iter().enumerate() {
                    let (tr, trs) = mean_std(&train[i]);
                    let (te, tes) = mean_std(&test[i]);
                    values.extend([*n as f64, tr, trs, te, tes]);
                }
                Ok(Table {
                    texts: Vec::new(),
                    values,
                    cols: 5,
                })
            }
            "validation_curve" => {
                let sets = need_grid()?;
                let ps: Vec<String> = sets.iter().map(|c| s.params(c)).collect();
                let configs: Vec<_> = ps
                    .iter()
                    .map(|p| {
                        move |a: &Array2<f64>, b: &Array1<f64>, c: &Array2<f64>| {
                            sr.fit_predict(p, a, b, c)
                        }
                    })
                    .collect();
                let (train, test) = ut::validation_curve_with_splits(
                    &x,
                    &y,
                    &folds(&x, &y, k).map_err(e)?,
                    &configs,
                    |a, b| s.score(a, b),
                )
                .map_err(e)?;
                let mut values = Vec::new();
                for i in 0..sets.len() {
                    let (tr, trs) = mean_std(&train[i]);
                    let (te, tes) = mean_std(&test[i]);
                    values.extend([tr, trs, te, tes]);
                }
                let texts = sets
                    .iter()
                    .map(|c| Value::Object(c.clone()).to_string())
                    .collect();
                Ok(Table {
                    texts,
                    values,
                    cols: 4,
                })
            }
            "permutation_test" => {
                let p = s.params(&Map::new());
                let r = va::permutation_test_score(
                    &x,
                    &y,
                    &folds(&x, &y, k).map_err(e)?,
                    None,
                    o.n_permutations.unwrap_or(100),
                    seed,
                    |a, b, c| s.fit_predict(&p, a, b, c),
                    // Greater is better here: losses are negated.
                    |a, b| s.score(a, b).map(|v| s.sign * v),
                )
                .map_err(e)?;
                let mut values = vec![s.sign * r.score, r.p_value];
                values.extend(r.permutation_scores.iter().map(|v| s.sign * v));
                let cols = values.len();
                Ok(Table {
                    texts: Vec::new(),
                    values,
                    cols,
                })
            }
            "nested_cv" => {
                let sets = need_grid()?;
                let ps: Vec<String> = sets.iter().map(|c| s.params(c)).collect();
                let configs: Vec<_> = ps
                    .iter()
                    .map(|p| {
                        move |a: &Array2<f64>, b: &Array1<f64>, c: &Array2<f64>| {
                            sr.fit_predict(p, a, b, c)
                        }
                    })
                    .collect();
                let inner = o.inner_cv.unwrap_or(3);
                let r = va::nested_cross_validate(
                    &x,
                    &y,
                    &folds(&x, &y, k).map_err(e)?,
                    |a, b| folds(a, b, inner),
                    &configs,
                    |a, b| s.score(a, b).map(|v| s.sign * v),
                )
                .map_err(e)?;
                let mut values = Vec::new();
                for (i, score) in r.outer_scores.iter().enumerate() {
                    values.extend([s.sign * score, s.sign * r.inner_best_scores[i]]);
                }
                let texts = r
                    .best_params_index
                    .iter()
                    .map(|&i| Value::Object(sets[i].clone()).to_string())
                    .collect();
                Ok(Table {
                    texts,
                    values,
                    cols: 2,
                })
            }
            _ => {
                let n_boot = o.n_bootstraps.unwrap_or(200);
                let splits = if stratify {
                    va::bootstrap_splits_stratified(&y, n_boot, seed)
                } else {
                    va::bootstrap_splits(y.len(), n_boot, seed)
                }
                .map_err(e)?;
                let p = s.params(&Map::new());
                // A loss (lower is better): score metrics are negated, and
                // the estimates turned back to the metric's scale.
                let r = va::bootstrap_validate(
                    &x,
                    &y,
                    &splits,
                    seed,
                    |a, b, c| s.fit_predict(&p, a, b, c),
                    |a, b| s.score(a, b).map(|v| -s.sign * v),
                )
                .map_err(e)?;
                let back = |v: f64| -s.sign * v;
                let values = vec![
                    back(r.apparent),
                    back(r.out_of_bag),
                    back(r.point632),
                    back(r.point632_plus),
                    back(r.optimism_corrected),
                    back(r.no_information),
                    r.relative_overfitting,
                    r.n_skipped as f64,
                ];
                Ok(Table {
                    texts: Vec::new(),
                    values,
                    cols: 8,
                })
            }
        }
    })?
}

/// Run model-selection `method` ([`METHODS`]) for `kind` on `n_rows × n_cols`
/// rows: `y` holds doubles, or `y_labels` strings (the other NULL);
/// `params` / `options` are JSON (may be NULL). `texts` receives one string
/// per result row (parameters as JSON, or none), `values` the row-major
/// table and `cols` its width.
///
/// # Safety
/// Strings are NUL-terminated; arrays hold the stated number of values; the
/// outputs are writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_select(
    method: *const c_char,
    kind: *const c_char,
    params: *const c_char,
    options: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const f64,
    y_labels: *const *const c_char,
    threads: usize,
    texts: *mut AnofoxMlStrings,
    values: *mut AnofoxMlDoubles,
    cols: *mut usize,
) -> *mut c_char {
    let run = || -> Result<Table, String> {
        let method = str_arg(method, "method")?;
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
        if !y_labels.is_null() {
            let (_, codes) = encode_labels(&strings(y_labels, n_rows, "y")?);
            select(
                method,
                kind,
                params,
                options,
                x,
                ArrayView1::from(&codes),
                true,
                threads,
            )
        } else if !y.is_null() {
            let y = ArrayView1::from(std::slice::from_raw_parts(y, n_rows));
            select(method, kind, params, options, x, y, false, threads)
        } else {
            Err("y is NULL".into())
        }
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(t)) => {
            *texts = into_strings(&t.texts);
            *values = into_doubles(t.values);
            *cols = t.cols;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(p) => err(crate::panic_message("while validating", p)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Array2<f64>, Array1<f64>) {
        let n = 120;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * 7 + j * 3) % 17) as f64 / 17.0);
        let y = Array1::from_shape_fn(n, |i| 2.0 * x[[i, 0]] - x[[i, 1]] + ((i % 5) as f64) * 0.01);
        (x, y)
    }

    #[test]
    fn learning_curve_and_grid_methods() {
        let (x, y) = data();
        let t = select(
            "learning_curve",
            "OlsRegressor",
            "",
            "",
            x.view(),
            y.view(),
            false,
            1,
        )
        .unwrap();
        assert_eq!(t.cols, 5);
        assert_eq!(t.values.len(), 25);
        assert!(
            t.values[23] > 0.9,
            "test r2 at the full size: {}",
            t.values[23]
        );
        let opts = r#"{"param_grid": {"max_depth": [1, 4]}, "cv": 3}"#;
        let t = select(
            "validation_curve",
            "DecisionTreeRegressor",
            "",
            opts,
            x.view(),
            y.view(),
            false,
            1,
        )
        .unwrap();
        assert_eq!(t.texts, [r#"{"max_depth":1}"#, r#"{"max_depth":4}"#]);
        assert!(t.values[6] > t.values[2], "deeper trees fit better");
        let t = select(
            "nested_cv",
            "DecisionTreeRegressor",
            "",
            opts,
            x.view(),
            y.view(),
            false,
            1,
        )
        .unwrap();
        assert_eq!((t.texts.len(), t.cols), (3, 2));
        let e = select(
            "nested_cv",
            "OlsRegressor",
            "",
            "",
            x.view(),
            y.view(),
            false,
            1,
        )
        .unwrap_err();
        assert!(e.contains("param_grid"), "{e}");
    }

    #[test]
    fn permutation_test_and_bootstrap_on_the_metric_scale() {
        let (x, y) = data();
        let opts = r#"{"n_permutations": 9, "metric": "mse"}"#;
        let t = select(
            "permutation_test",
            "OlsRegressor",
            "",
            opts,
            x.view(),
            y.view(),
            false,
            1,
        )
        .unwrap();
        assert_eq!(t.cols, 11);
        // A real signal: lower mse than every permutation, p = 1 / 10.
        assert!(t.values[2..].iter().all(|&v| v > t.values[0]));
        assert!((t.values[1] - 0.1).abs() < 1e-12);
        let opts = r#"{"n_bootstraps": 20, "metric": "mse"}"#;
        let t = select(
            "bootstrap_validate",
            "OlsRegressor",
            "",
            opts,
            x.view(),
            y.view(),
            false,
            1,
        )
        .unwrap();
        // apparent <= optimism-corrected mse; all on the mse scale (>= 0)
        assert!(t.values[0] >= 0.0 && t.values[4] >= t.values[0]);
    }
}
