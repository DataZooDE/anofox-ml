//! Model-free evaluation metrics behind [`anofox_ml_metric`], dispatched by
//! name onto `anofox-ml-metrics`.
//!
//! `y_true` holds `n` values; `y_pred` holds `n × n_cols` doubles, row-major:
//! one prediction / score per row (`n_cols = 1`), a probability row for
//! multiclass metrics, the feature row for `silhouette`, or
//! `[lower, upper]` for the interval metrics.

use std::ffi::c_char;

use anofox_ml_metrics as m;
use m::{Average, ClassificationOptions, MultiClass, ZeroDivision};
use ndarray::{Array1, Array2};
use serde::Deserialize;

use crate::{
    encode_labels, err, into_doubles, into_strings, str_arg, strings, AnofoxMlDoubles,
    AnofoxMlStrings,
};

/// Options shared by the metrics (all optional, sklearn defaults).
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Options {
    /// `binary` (default), `macro`, `micro` or `weighted`.
    average: Option<String>,
    /// Positive label for `average = binary` (default 1).
    pos_label: Option<f64>,
    /// `0` (default), `1` or `"nan"`.
    zero_division: Option<serde_json::Value>,
    /// F-beta weight (default 1).
    beta: Option<f64>,
    /// Multiclass ROC AUC: `ovr` (default) or `ovo`.
    multi_class: Option<String>,
    /// Labels: the column order of a probability matrix, or the classes to
    /// average over.
    labels: Option<Vec<f64>>,
    /// Miscoverage level of `interval_score`.
    alpha: Option<f64>,
}

impl Options {
    fn average(&self, default: Average) -> Result<Average, String> {
        Ok(match self.average.as_deref() {
            None => match default {
                Average::Binary { .. } => Average::Binary {
                    pos_label: self.pos_label.unwrap_or(1.0),
                },
                other => other,
            },
            Some("binary") => Average::Binary {
                pos_label: self.pos_label.unwrap_or(1.0),
            },
            Some("macro") => Average::Macro,
            Some("micro") => Average::Micro,
            Some("weighted") => Average::Weighted,
            Some(a) => {
                return Err(format!(
                    "unknown average '{a}' (binary, macro, micro, weighted)"
                ))
            }
        })
    }

    fn zero_division(&self) -> Result<ZeroDivision, String> {
        match &self.zero_division {
            None => Ok(ZeroDivision::Zero),
            Some(v) if v == 0 => Ok(ZeroDivision::Zero),
            Some(v) if v == 1 => Ok(ZeroDivision::One),
            Some(serde_json::Value::String(s)) if s.eq_ignore_ascii_case("nan") => {
                Ok(ZeroDivision::NaN)
            }
            Some(v) => Err(format!("zero_division must be 0, 1 or \"nan\", got {v}")),
        }
    }
}

/// Metric names accepted by [`anofox_ml_metric`].
pub(crate) const METRICS: &[&str] = &[
    "r2",
    "mse",
    "rmse",
    "mae",
    "mape",
    "median_absolute_error",
    "max_error",
    "explained_variance",
    "mean_squared_log_error",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "fbeta",
    "matthews_corrcoef",
    "cohen_kappa",
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
    "adjusted_rand",
    "normalized_mutual_info",
    "silhouette",
    "coverage",
    "interval_score",
    "conformal_margin",
    "mean_pinball_loss",
];

pub(crate) fn metric(
    name: &str,
    options: &str,
    y_true: Array1<f64>,
    y_pred: Array2<f64>,
    w: Option<Array1<f64>>,
) -> Result<f64, String> {
    let o: Options = if options.trim().is_empty() {
        Options::default()
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    let cols = y_pred.ncols();
    let need_cols = |want: usize| -> Result<(), String> {
        if cols == want {
            Ok(())
        } else {
            Err(format!("{name} needs {want} value(s) per row, got {cols}"))
        }
    };
    let no_weight = |w: &Option<Array1<f64>>| -> Result<(), String> {
        match w {
            None => Ok(()),
            Some(_) => Err(format!("{name} does not accept a weight")),
        }
    };
    let col = |j: usize| y_pred.column(j).to_owned();
    let labels = o.labels.as_deref();
    let opts = || -> Result<ClassificationOptions<'_, f64>, String> {
        let mut c = ClassificationOptions::new().with_zero_division(o.zero_division()?);
        if let Some(l) = labels {
            c = c.with_labels(l);
        }
        if let Some(w) = &w {
            c = c.with_sample_weight(w);
        }
        Ok(c)
    };
    let r = match name {
        "r2"
        | "mse"
        | "rmse"
        | "mae"
        | "mape"
        | "median_absolute_error"
        | "max_error"
        | "explained_variance"
        | "mean_squared_log_error" => {
            need_cols(1)?;
            no_weight(&w)?;
            let p = col(0);
            match name {
                "r2" => m::r2_score(&y_true, &p),
                "mse" => m::mse(&y_true, &p),
                "rmse" => m::mse(&y_true, &p).map(f64::sqrt),
                "mae" => m::mae(&y_true, &p),
                "mape" => m::mean_absolute_percentage_error(&y_true, &p),
                "median_absolute_error" => m::median_absolute_error(&y_true, &p),
                "max_error" => m::max_error(&y_true, &p),
                "explained_variance" => m::explained_variance_score(&y_true, &p),
                _ => m::mean_squared_log_error(&y_true, &p),
            }
        }
        "accuracy" => {
            need_cols(1)?;
            m::accuracy_score_with(&y_true, &col(0), w.as_ref())
        }
        "balanced_accuracy"
        | "matthews_corrcoef"
        | "cohen_kappa"
        | "adjusted_rand"
        | "normalized_mutual_info"
        | "average_precision" => {
            need_cols(1)?;
            no_weight(&w)?;
            let p = col(0);
            match name {
                "balanced_accuracy" => m::balanced_accuracy_score(&y_true, &p),
                "matthews_corrcoef" => m::matthews_corrcoef(&y_true, &p),
                "cohen_kappa" => m::cohen_kappa_score(&y_true, &p),
                "adjusted_rand" => m::adjusted_rand_score(&y_true, &p),
                "normalized_mutual_info" => m::normalized_mutual_info_score(&y_true, &p),
                _ => m::average_precision_score(&y_true, &p),
            }
        }
        "precision" | "recall" | "f1" | "fbeta" => {
            need_cols(1)?;
            let p = col(0);
            let avg = o.average(Average::binary())?;
            let c = opts()?;
            match name {
                "precision" => m::precision_score_with(&y_true, &p, avg, &c),
                "recall" => m::recall_score_with(&y_true, &p, avg, &c),
                "f1" => m::f1_score_with(&y_true, &p, avg, &c),
                _ => m::fbeta_score_with(&y_true, &p, o.beta.unwrap_or(1.0), avg, &c),
            }
        }
        "roc_auc" if cols == 1 => m::roc_auc_score_with(&y_true, &col(0), w.as_ref()),
        "roc_auc" => {
            let mc = match o.multi_class.as_deref() {
                None | Some("ovr") => MultiClass::Ovr,
                Some("ovo") => MultiClass::Ovo,
                Some(v) => return Err(format!("unknown multi_class '{v}' (ovr, ovo)")),
            };
            let avg = o.average(Average::Macro)?;
            m::roc_auc_score_multiclass(&y_true, &y_pred, mc, avg, labels, w.as_ref())
        }
        "log_loss" if cols == 1 => m::log_loss_with(&y_true, &col(0), w.as_ref()),
        "log_loss" => m::log_loss_multiclass(&y_true, &y_pred, labels, w.as_ref()),
        "brier_score" => {
            need_cols(1)?;
            m::brier_score_loss_with(&y_true, &col(0), w.as_ref())
        }
        "silhouette" => {
            no_weight(&w)?;
            m::silhouette_score(&y_pred, &y_true)
        }
        "coverage" | "interval_score" => {
            need_cols(2)?;
            no_weight(&w)?;
            let (lo, hi) = (col(0), col(1));
            if name == "coverage" {
                m::coverage_score(&y_true, &lo, &hi)
            } else {
                let alpha = o.alpha.ok_or("interval_score needs alpha")?;
                m::interval_score(&y_true, &lo, &hi, alpha)
            }
        }
        "mean_pinball_loss" => {
            need_cols(1)?;
            no_weight(&w)?;
            let alpha = o.alpha.ok_or("mean_pinball_loss needs alpha")?;
            m::intervals::mean_pinball_loss(&y_true, &col(0), alpha)
        }
        "conformal_margin" => {
            need_cols(1)?;
            no_weight(&w)?;
            let alpha = o.alpha.ok_or("conformal_margin needs alpha")?;
            let scores: Vec<f64> = y_true
                .iter()
                .zip(col(0).iter())
                .map(|(t, p)| (t - p).abs())
                .collect();
            anofox_ml_conformal::conformal_quantile(&scores, alpha)
                .map(|q| q.unwrap_or(f64::INFINITY))
        }
        _ => return Err(format!("unknown metric '{name}' (known: {METRICS:?})")),
    };
    r.map_err(|e| e.to_string())
}

/// Compute metric `name` with JSON `options` (may be NULL) on `n` rows:
/// `y_true` holds `n` doubles, `y_pred` holds `n * n_cols`, `w` holds `n`
/// weights or is NULL. `out` receives the value.
///
/// # Safety
/// `name` / `options` are NUL-terminated strings (`options` may be NULL);
/// the arrays hold the stated number of doubles; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_metric(
    name: *const c_char,
    options: *const c_char,
    y_true: *const f64,
    y_pred: *const f64,
    n: usize,
    n_cols: usize,
    w: *const f64,
    out: *mut f64,
) -> *mut c_char {
    let run = || -> Result<f64, String> {
        let name = str_arg(name, "metric name")?;
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        if n == 0 || n_cols == 0 {
            return Err("empty input".into());
        }
        if y_true.is_null() || y_pred.is_null() {
            return Err("input is NULL".into());
        }
        let yt = Array1::from(std::slice::from_raw_parts(y_true, n).to_vec());
        let yp = Array2::from_shape_vec(
            (n, n_cols),
            std::slice::from_raw_parts(y_pred, n * n_cols).to_vec(),
        )
        .map_err(|e| e.to_string())?;
        let w = (!w.is_null()).then(|| Array1::from(std::slice::from_raw_parts(w, n).to_vec()));
        metric(name, options, yt, yp, w)
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(v)) => {
            *out = v;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(p) => err(crate::panic_message("while computing a metric", p)),
    }
}

/// Metrics that compare two label vectors, which also accept string labels.
const LABEL_METRICS: &[&str] = &[
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "fbeta",
    "matthews_corrcoef",
    "cohen_kappa",
    "adjusted_rand",
    "normalized_mutual_info",
];

/// [`metric`] on string labels: both sides are coded by the sorted union of
/// their labels, and string `pos_label` / `labels` options are translated.
pub(crate) fn label_metric(
    name: &str,
    options: &str,
    y_true: &[&str],
    y_pred: &[&str],
    w: Option<Array1<f64>>,
) -> Result<f64, String> {
    if !LABEL_METRICS.contains(&name) {
        return Err(format!("{name} needs numeric values, not labels"));
    }
    let all: Vec<&str> = y_true.iter().chain(y_pred).copied().collect();
    let (labels, codes) = encode_labels(&all);
    let o = coded_options(options, &labels)?;
    let n = y_true.len();
    let yt = Array1::from(codes[..n].to_vec());
    let yp = Array2::from_shape_vec((n, 1), codes[n..].to_vec()).map_err(|e| e.to_string())?;
    metric(name, &o, yt, yp, w)
}

/// `options` with string `pos_label` / `labels` replaced by their codes in
/// the sorted `labels`.
fn coded_options(options: &str, labels: &[String]) -> Result<String, String> {
    let code = |v: &serde_json::Value| -> Result<serde_json::Value, String> {
        let s = v
            .as_str()
            .ok_or_else(|| format!("with string labels, {v} must be a label string"))?;
        labels
            .binary_search_by(|l| l.as_str().cmp(s))
            .map(|i| serde_json::json!(i))
            .map_err(|_| format!("label '{s}' does not occur"))
    };
    let mut o: serde_json::Value = if options.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    if let Some(obj) = o.as_object_mut() {
        if let Some(p) = obj.get("pos_label") {
            let c = code(p)?;
            obj.insert("pos_label".into(), c);
        }
        if let Some(serde_json::Value::Array(ls)) = obj.get("labels") {
            let c = ls.iter().map(code).collect::<Result<Vec<_>, _>>()?;
            obj.insert("labels".into(), serde_json::Value::Array(c));
        }
    }
    Ok(o.to_string())
}

/// [`anofox_ml_metric`] for string labels: `y_true` and `y_pred` hold `n`
/// NUL-terminated strings each.
///
/// # Safety
/// As [`anofox_ml_metric`], with string arrays.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_metric_labels(
    name: *const c_char,
    options: *const c_char,
    y_true: *const *const c_char,
    y_pred: *const *const c_char,
    n: usize,
    w: *const f64,
    out: *mut f64,
) -> *mut c_char {
    let run = || -> Result<f64, String> {
        let name = str_arg(name, "metric name")?;
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        if n == 0 {
            return Err("empty input".into());
        }
        let yt = strings(y_true, n, "y_true")?;
        let yp = strings(y_pred, n, "y_pred")?;
        let w = (!w.is_null()).then(|| Array1::from(std::slice::from_raw_parts(w, n).to_vec()));
        label_metric(name, options, &yt, &yp, w)
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(v)) => {
            *out = v;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(p) => err(crate::panic_message("while computing a metric", p)),
    }
}

/// Table-valued metrics behind [`anofox_ml_metric_table`].
pub(crate) const TABLE_METRICS: &[&str] = &[
    "confusion_matrix",
    "classification_report",
    "roc_curve",
    "precision_recall_curve",
];

/// A table-valued metric: class labels (the confusion matrix's rows and
/// columns, the report's rows; empty for the curves) and an
/// `n_rows × n_cols` row-major table.
///
/// - `confusion_matrix`: `k × k` counts, true classes in rows.
/// - `classification_report`: one row per class, `[precision, recall,
///   fscore, support]` (`beta` option, default 1).
/// - `roc_curve`: rows `[threshold, fpr, tpr]`, thresholds descending.
/// - `precision_recall_curve`: rows `[threshold, precision, recall]`,
///   thresholds ascending, then the sentinel `[NaN, 1, 0]`.
pub(crate) struct Table {
    pub labels: Vec<f64>,
    pub values: Vec<f64>,
    pub cols: usize,
}

pub(crate) fn table_metric(
    name: &str,
    options: &str,
    y_true: Array1<f64>,
    y_pred: Array1<f64>,
    w: Option<Array1<f64>>,
) -> Result<Table, String> {
    let o: Options = if options.trim().is_empty() {
        Options::default()
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    let opts = || -> Result<ClassificationOptions<'_, f64>, String> {
        let mut c = ClassificationOptions::new().with_zero_division(o.zero_division()?);
        if let Some(l) = o.labels.as_deref() {
            c = c.with_labels(l);
        }
        if let Some(w) = &w {
            c = c.with_sample_weight(w);
        }
        Ok(c)
    };
    let classes = || -> Vec<f64> {
        if let Some(l) = &o.labels {
            return l.clone();
        }
        let mut c: Vec<f64> = y_true.iter().chain(y_pred.iter()).copied().collect();
        c.sort_by(f64::total_cmp);
        c.dedup();
        c
    };
    let curve_only = |w: &Option<Array1<f64>>| -> Result<(), String> {
        match w {
            None => Ok(()),
            Some(_) => Err(format!("{name} does not accept a weight")),
        }
    };
    let rows = |cols: Vec<Array1<f64>>| -> Vec<f64> {
        let n = cols[0].len();
        (0..n)
            .flat_map(|r| cols.iter().map(move |c| c[r]))
            .collect()
    };
    let e = |e: anofox_ml_core::RustMlError| e.to_string();
    match name {
        "confusion_matrix" => {
            let cm = m::confusion_matrix_with(&y_true, &y_pred, &opts()?).map_err(e)?;
            Ok(Table {
                labels: classes(),
                cols: cm.ncols(),
                values: cm.as_standard_layout().iter().copied().collect(),
            })
        }
        "classification_report" => {
            let s = m::precision_recall_fscore_support_per_class(
                &y_true,
                &y_pred,
                o.beta.unwrap_or(1.0),
                &opts()?,
            )
            .map_err(e)?;
            Ok(Table {
                values: rows(vec![s.precision, s.recall, s.fscore, s.support]),
                labels: s.labels,
                cols: 4,
            })
        }
        "roc_curve" => {
            curve_only(&w)?;
            let (fpr, tpr, t) = m::roc_curve(&y_true, &y_pred).map_err(e)?;
            Ok(Table {
                labels: Vec::new(),
                values: rows(vec![t, fpr, tpr]),
                cols: 3,
            })
        }
        "precision_recall_curve" => {
            curve_only(&w)?;
            let (p, r, t) = m::precision_recall_curve(&y_true, &y_pred).map_err(e)?;
            let mut t = t.to_vec();
            t.push(f64::NAN);
            Ok(Table {
                labels: Vec::new(),
                values: rows(vec![Array1::from(t), p, r]),
                cols: 3,
            })
        }
        _ => Err(format!(
            "unknown metric '{name}' (known: {TABLE_METRICS:?})"
        )),
    }
}

/// Run a table-valued metric and hand its labels (as text) and values out.
unsafe fn table_out(
    t: Result<(Table, Vec<String>), String>,
    labels: *mut AnofoxMlStrings,
    values: *mut AnofoxMlDoubles,
    cols: *mut usize,
) -> *mut c_char {
    match t {
        Ok((t, names)) => {
            *labels = into_strings(&names);
            *values = into_doubles(t.values);
            *cols = t.cols;
            std::ptr::null_mut()
        }
        Err(e) => err(e),
    }
}

/// Compute the table-valued metric `name` ([`TABLE_METRICS`]) on `n` rows
/// of numeric `y_true` / `y_pred` (scores for the curves). `labels`
/// receives the class labels as text, `values` the table, `cols` its width.
///
/// # Safety
/// As [`anofox_ml_metric`] with `n_cols = 1`; the outputs are writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_metric_table(
    name: *const c_char,
    options: *const c_char,
    y_true: *const f64,
    y_pred: *const f64,
    n: usize,
    w: *const f64,
    labels: *mut AnofoxMlStrings,
    values: *mut AnofoxMlDoubles,
    cols: *mut usize,
) -> *mut c_char {
    let run = || -> Result<(Table, Vec<String>), String> {
        let name = str_arg(name, "metric name")?;
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        if n == 0 {
            return Err("empty input".into());
        }
        if y_true.is_null() || y_pred.is_null() {
            return Err("input is NULL".into());
        }
        let yt = Array1::from(std::slice::from_raw_parts(y_true, n).to_vec());
        let yp = Array1::from(std::slice::from_raw_parts(y_pred, n).to_vec());
        let w = (!w.is_null()).then(|| Array1::from(std::slice::from_raw_parts(w, n).to_vec()));
        let t = table_metric(name, options, yt, yp, w)?;
        let names = t.labels.iter().map(|&v| number_label(v)).collect();
        Ok((t, names))
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(t) => table_out(t, labels, values, cols),
        Err(p) => err(crate::panic_message("while computing a metric", p)),
    }
}

/// [`anofox_ml_metric_table`] for string class labels
/// (`confusion_matrix`, `classification_report`).
///
/// # Safety
/// As [`anofox_ml_metric_labels`]; the outputs are writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_metric_table_labels(
    name: *const c_char,
    options: *const c_char,
    y_true: *const *const c_char,
    y_pred: *const *const c_char,
    n: usize,
    w: *const f64,
    labels: *mut AnofoxMlStrings,
    values: *mut AnofoxMlDoubles,
    cols: *mut usize,
) -> *mut c_char {
    let run = || -> Result<(Table, Vec<String>), String> {
        let name = str_arg(name, "metric name")?;
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        if n == 0 {
            return Err("empty input".into());
        }
        if name != "confusion_matrix" && name != "classification_report" {
            return Err(format!("{name} needs numeric labels and scores"));
        }
        let yt = strings(y_true, n, "y_true")?;
        let yp = strings(y_pred, n, "y_pred")?;
        let all: Vec<&str> = yt.iter().chain(&yp).copied().collect();
        let (names, codes) = encode_labels(&all);
        let o = coded_options(options, &names)?;
        let w = (!w.is_null()).then(|| Array1::from(std::slice::from_raw_parts(w, n).to_vec()));
        let t = table_metric(
            name,
            &o,
            Array1::from(codes[..n].to_vec()),
            Array1::from(codes[n..].to_vec()),
            w,
        )?;
        let text = t
            .labels
            .iter()
            .map(|&c| names[c as usize].clone())
            .collect();
        Ok((t, text))
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(t) => table_out(t, labels, values, cols),
        Err(p) => err(crate::panic_message("while computing a metric", p)),
    }
}

/// A numeric class as text: `1`, not `1.0`, for integral values.
fn number_label(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        v.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn run(name: &str, opts: &str, yt: Array1<f64>, yp: Array2<f64>) -> f64 {
        metric(name, opts, yt, yp, None).unwrap()
    }

    fn col(v: Array1<f64>) -> Array2<f64> {
        let n = v.len();
        v.into_shape_with_order((n, 1)).unwrap()
    }

    #[test]
    fn regression_metrics_match_hand_values() {
        let yt = array![3.0, -0.5, 2.0, 7.0];
        let yp = col(array![2.5, 0.0, 2.0, 8.0]);
        // sklearn: r2_score = 0.9486081370449679, mse = 0.375, mae = 0.5
        assert!((run("r2", "", yt.clone(), yp.clone()) - 0.948_608_137_044_967_9).abs() < 1e-12);
        assert_eq!(run("mse", "", yt.clone(), yp.clone()), 0.375);
        assert_eq!(run("rmse", "", yt.clone(), yp.clone()), 0.375_f64.sqrt());
        assert_eq!(run("mae", "", yt, yp), 0.5);
    }

    #[test]
    fn classification_options_are_applied() {
        let yt = array![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let yp = col(array![0.0, 2.0, 1.0, 0.0, 0.0, 1.0]);
        // sklearn: f1_score(average='macro') = 0.26666666666666666
        let f1 = run("f1", r#"{"average": "macro"}"#, yt.clone(), yp.clone());
        assert!((f1 - 0.266_666_666_666_666_66).abs() < 1e-12);
        let e = metric("f1", "", yt.clone(), yp.clone(), None).unwrap_err();
        assert!(e.contains("multiclass"), "{e}");
        let e = metric("f1", r#"{"avg": "macro"}"#, yt, yp, None).unwrap_err();
        assert!(e.contains("unknown field"), "{e}");
    }

    #[test]
    fn probability_matrix_and_intervals() {
        let yt = array![0.0, 1.0, 2.0];
        let p = array![[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]];
        // -(ln .8 + ln .7 + ln .7) / 3
        let want = -(0.8_f64.ln() + 2.0 * 0.7_f64.ln()) / 3.0;
        assert!((run("log_loss", "", yt.clone(), p.clone()) - want).abs() < 1e-12);
        assert_eq!(run("roc_auc", "", yt, p), 1.0);
        let iv = array![[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]];
        assert_eq!(run("coverage", "", array![1.0, 1.0, 3.0, 1.0], iv), 0.75);
        let e = metric("mse", "", array![1.0], array![[1.0, 2.0]], None).unwrap_err();
        assert!(e.contains("needs 1 value"), "{e}");
        let e = metric("mse", "", array![1.0], col(array![1.0]), Some(array![1.0])).unwrap_err();
        assert!(e.contains("does not accept a weight"), "{e}");
    }

    #[test]
    fn string_labels_are_coded_consistently() {
        let yt = ["cat", "dog", "dog", "bird"];
        let yp = ["cat", "dog", "cat", "bird"];
        assert_eq!(label_metric("accuracy", "", &yt, &yp, None).unwrap(), 0.75);
        // f1 of 'dog' as the positive label: precision 1, recall 1/2.
        let f1 = label_metric(
            "f1",
            r#"{"average": "macro", "labels": ["dog"]}"#,
            &yt,
            &yp,
            None,
        )
        .unwrap();
        assert!((f1 - 2.0 / 3.0).abs() < 1e-12, "{f1}");
        let e = label_metric("f1", r#"{"pos_label": "fish"}"#, &yt, &yp, None).unwrap_err();
        assert!(e.contains("does not occur"), "{e}");
        let e = label_metric("r2", "", &yt, &yp, None).unwrap_err();
        assert!(e.contains("numeric"), "{e}");
    }

    #[test]
    fn table_metrics_match_sklearn() {
        // sklearn: confusion_matrix([2, 0, 2, 2, 0, 1], [0, 0, 2, 2, 0, 2])
        let yt = array![2.0, 0.0, 2.0, 2.0, 0.0, 1.0];
        let yp = array![0.0, 0.0, 2.0, 2.0, 0.0, 2.0];
        let t = table_metric("confusion_matrix", "", yt.clone(), yp.clone(), None).unwrap();
        assert_eq!(t.labels, [0.0, 1.0, 2.0]);
        assert_eq!(t.values, [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0]);
        // classification_report row of class 2: precision 2/3, recall 2/3, support 3
        let t = table_metric("classification_report", "", yt, yp, None).unwrap();
        assert_eq!(t.cols, 4);
        assert!((t.values[8] - 2.0 / 3.0).abs() < 1e-12 && t.values[11] == 3.0);
        // sklearn: roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) has 5 points
        // with drop_intermediate=False; precision_recall_curve ends at (1, 0)
        let (yt, s) = (array![0.0, 0.0, 1.0, 1.0], array![0.1, 0.4, 0.35, 0.8]);
        let t = table_metric("roc_curve", "", yt.clone(), s.clone(), None).unwrap();
        assert_eq!(
            (t.values.len() / 3, t.values[1], t.values[2]),
            (5, 0.0, 0.0)
        );
        let t = table_metric("precision_recall_curve", "", yt, s, None).unwrap();
        let last = &t.values[t.values.len() - 3..];
        assert!(last[0].is_nan() && last[1] == 1.0 && last[2] == 0.0);
    }
}
