//! Hyperparameter search behind `ml_tune_agg`: grid, seeded random,
//! successive halving or TPE ([`crate::tpe`]) search over a JSON search space, every candidate scored by k-fold
//! cross-validation with a metric of the metric aggregates, the best one
//! refitted on all rows.
//!
//! Candidates are generated up front from the seed (TPE: one after the
//! other from the seed and the earlier scores) and each (candidate, fold) is
//! scored independently, so the result does not depend on how many threads
//! score them.

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
    /// `"random"` (default), `"grid"`, `"halving"` (successive halving of
    /// random candidates), `"halving_grid"` or `"tpe"`.
    search: Option<String>,
    /// Halving: keep the best `1 / factor` of the candidates per round and
    /// grow the rows by `factor` (default 3).
    factor: Option<usize>,
    /// Candidates of a random / TPE search (default 20).
    n_trials: Option<usize>,
    /// TPE: random trials before the first model-based one (default 10).
    n_startup_trials: Option<usize>,
    /// TPE: draws from the good trials' estimator per proposal (default 24).
    n_ei_candidates: Option<usize>,
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
pub(crate) enum Dim {
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

pub(crate) fn sample(dim: &Dim, rng: &mut StdRng) -> Value {
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
        "grid" | "halving_grid" => {
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
        "random" | "halving" => {
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
        "tpe" => Ok(Vec::new()),
        s => Err(format!(
            "search must be \"random\", \"grid\", \"halving\", \"halving_grid\" or \"tpe\", got \"{s}\""
        )),
    }
}

/// Mean and std CV score of one candidate, or why it failed.
type CandidateScore = Result<(f64, f64), String>;

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
    let stratify = o.stratify.unwrap_or(classifier);
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
    let with_fixed = |c: &Map<String, Value>| {
        let mut p = fixed.clone();
        p.extend(c.clone());
        p
    };
    let full_params: Vec<Map<String, Value>> = cands.iter().map(with_fixed).collect();

    // Score one (candidate, fold); each fit runs sequentially inside the
    // parallel loop.
    let score_fold = |xs: ArrayView2<f64>,
                      ys: ArrayView1<f64>,
                      p: &Map<String, Value>,
                      (train, test): &(Vec<usize>, Vec<usize>)|
     -> Result<f64, String> {
        let (xtr, ytr) = (xs.select(Axis(0), train), ys.select(Axis(0), train));
        let (xte, yte) = (xs.select(Axis(0), test), ys.select(Axis(0), test));
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
    // Mean / std CV score of the parameter sets `cand` on rows `rows`.
    let round =
        |rows: &[usize], cand: &[&Map<String, Value>]| -> Result<Vec<CandidateScore>, String> {
            let (xs, ys) = (x.select(Axis(0), rows), y.select(Axis(0), rows));
            let folds = if stratify {
                ut::stratified_k_fold(&xs, &ys, k, seed)
            } else {
                ut::shuffled_k_fold(rows.len(), k, seed)
            }
            .map_err(|e| e.to_string())?;
            let jobs: Vec<(usize, usize)> = (0..cand.len())
                .flat_map(|c| (0..folds.len()).map(move |f| (c, f)))
                .collect();
            let scores: Vec<Result<f64, String>> = run_fit(threads, || {
                jobs.par_iter()
                    .map(|&(c, f)| score_fold(xs.view(), ys.view(), cand[c], &folds[f]))
                    .collect()
            })?;
            Ok(scores
                .chunks(folds.len())
                .map(|s| {
                    let s: Vec<f64> = s.iter().cloned().collect::<Result<_, _>>()?;
                    let mean = s.iter().sum::<f64>() / s.len() as f64;
                    let std =
                        (s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
                    Ok((mean, std))
                })
                .collect())
        };
    let better = |a: f64, b: f64| if minimise { a < b } else { a > b };
    let n = x.nrows();
    let record = |t: &mut Value, res: CandidateScore| -> Option<f64> {
        match res {
            Ok((mean, std)) => {
                t["score"] = json!(mean);
                t["score_std"] = json!(std);
                mean.is_finite().then_some(mean)
            }
            Err(e) => {
                t["error"] = json!(e);
                None
            }
        }
    };
    // Refit the best parameters on all rows.
    let finish = |best: Option<(Map<String, Value>, f64)>,
                  trials: Vec<Value>|
     -> Result<TuneResult, String> {
        let (params, best_score) = best.ok_or_else(|| {
            let first = trials
                .iter()
                .find_map(|t| t.get("error"))
                .cloned()
                .unwrap_or_default();
            format!("every trial failed (first error: {first})")
        })?;
        let p = Value::Object(params.clone()).to_string();
        let model = fit_impl(kind, &p, x, Some(y), None, labels.clone(), threads)?;
        Ok(TuneResult {
            best_params: Value::Object(params),
            best_score,
            model,
            trials: Value::Array(trials),
        })
    };
    if o.search.as_deref() == Some("tpe") {
        // One trial at a time, each proposed from the scores before it.
        let mut rng = StdRng::seed_from_u64(seed);
        let rows: Vec<usize> = (0..n).collect();
        let (n_startup, n_ei) = (
            o.n_startup_trials.unwrap_or(10),
            o.n_ei_candidates.unwrap_or(24),
        );
        let mut cands: Vec<Map<String, Value>> = Vec::new();
        let mut scored: Vec<(usize, f64)> = Vec::new();
        let mut trials = Vec::new();
        for c in 0..o.n_trials.unwrap_or(20).max(1) {
            let history: Vec<(&Map<String, Value>, f64)> =
                scored.iter().map(|&(i, s)| (&cands[i], s)).collect();
            let next = crate::tpe::propose(&space, &history, minimise, n_startup, n_ei, &mut rng);
            let res = round(&rows, &[&with_fixed(&next)])?.remove(0);
            let mut t = json!({"trial": c, "params": next});
            if let Some(s) = record(&mut t, res) {
                scored.push((c, s));
            }
            trials.push(t);
            cands.push(next);
        }
        // Ties keep the earlier trial.
        let best = scored
            .iter()
            .copied()
            .reduce(|b, s| if better(s.1, b.1) { s } else { b });
        return finish(best.map(|(c, s)| (with_fixed(&cands[c]), s)), trials);
    }
    let halving = matches!(o.search.as_deref(), Some("halving") | Some("halving_grid"));
    let factor = o.factor.unwrap_or(3).max(2);
    // Rounds: (rows, candidates). Halving starts on n / factor^(rounds-1)
    // rows of a seeded permutation and keeps the best 1/factor each round.
    let n_rounds = if halving {
        let mut r = 1;
        let mut c = full_params.len();
        while c > 1 && n / factor.pow(r as u32) >= 2 * k {
            c = c.div_ceil(factor);
            r += 1;
        }
        r
    } else {
        1
    };
    let mut perm: Vec<usize> = (0..n).collect();
    {
        use rand::seq::SliceRandom;
        perm.shuffle(&mut StdRng::seed_from_u64(seed));
    }
    let mut alive: Vec<usize> = (0..full_params.len()).collect();
    let mut trials = Vec::new();
    let mut best: Option<(usize, f64)> = None;
    for r in 0..n_rounds {
        let n_rows = if r + 1 == n_rounds {
            n
        } else {
            n / factor.pow((n_rounds - 1 - r) as u32)
        };
        let mut rows: Vec<usize> = perm[..n_rows].to_vec();
        rows.sort_unstable();
        let alive_params: Vec<&Map<String, Value>> =
            alive.iter().map(|&c| &full_params[c]).collect();
        let res = round(&rows, &alive_params)?;
        let mut scored: Vec<(usize, f64)> = Vec::new();
        for (&c, res) in alive.iter().zip(res) {
            let mut t = json!({"trial": c, "params": cands[c]});
            if halving {
                t["round"] = json!(r);
                t["n_rows"] = json!(n_rows);
            }
            if let Some(s) = record(&mut t, res) {
                scored.push((c, s));
            }
            trials.push(t);
        }
        // Stable: ties keep the earlier candidate.
        scored.sort_by(|a, b| {
            if better(a.1, b.1) {
                std::cmp::Ordering::Less
            } else if better(b.1, a.1) {
                std::cmp::Ordering::Greater
            } else {
                a.0.cmp(&b.0)
            }
        });
        best = scored.first().copied();
        let keep = if r + 1 == n_rounds {
            1
        } else {
            alive.len().div_ceil(factor).max(1)
        };
        alive = scored.iter().take(keep).map(|s| s.0).collect();
        if alive.is_empty() {
            break;
        }
    }
    finish(best.map(|(c, s)| (full_params[c].clone(), s)), trials)
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

    #[test]
    fn halving_narrows_down_to_the_best_candidate() {
        let (x, y) = data();
        let r = tune(
            "DecisionTreeRegressor",
            r#"{"max_depth": [0, 1, 2, 3, 4, 5, 6, 7, 8]}"#,
            r#"{"search": "halving_grid", "cv": 3, "factor": 3}"#,
            x.view(),
            y.view(),
            None,
            4,
        )
        .unwrap();
        let t = r.trials.as_array().unwrap();
        // 9 candidates on n/9 rows, 3 on n/3, 1 on all rows.
        let per_round = |k: u64| t.iter().filter(|t| t["round"] == k).count();
        assert_eq!(
            (per_round(0), per_round(1), per_round(2)),
            (9, 3, 1),
            "{}",
            r.trials
        );
        assert_eq!(t.last().unwrap()["n_rows"], 200);
        assert_ne!(r.best_params["max_depth"], 0, "{}", r.trials);
    }

    #[test]
    fn tpe_is_seeded_thread_independent_and_improves_on_the_startup_trials() {
        // Noisy sine: few neighbours fit the noise, many flatten the curve.
        let n = 300;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64 * 6.0);
        let y = Array1::from_shape_fn(n, |i| {
            (i as f64 / n as f64 * 6.0).sin() + (((i * 7919) % 101) as f64 / 101.0 - 0.5)
        });
        let space = r#"{"n_neighbors": {"int": [1, 150], "log": true}}"#;
        let opts = r#"{"search": "tpe", "n_trials": 30, "n_startup_trials": 10, "seed": 1, "metric": "rmse", "cv": 4}"#;
        let run = |threads| {
            tune(
                "KnnRegressor",
                space,
                opts,
                x.view(),
                y.view(),
                None,
                threads,
            )
            .unwrap()
        };
        let (a, b) = (run(1), run(8));
        assert_eq!(a.trials, b.trials);
        assert_eq!(a.model, b.model);
        let t = a.trials.as_array().unwrap();
        assert_eq!(t.len(), 30);
        // The model-based trials do better on average than the random ones.
        let score = |t: &Value| t["score"].as_f64().unwrap();
        let mean = |t: &[Value]| t.iter().map(score).sum::<f64>() / t.len() as f64;
        assert!(mean(&t[20..]) < mean(&t[..10]), "{}", a.trials);
        let k = a.best_params["n_neighbors"].as_i64().unwrap();
        assert!((5..=60).contains(&k), "{}", a.trials);
    }
}
