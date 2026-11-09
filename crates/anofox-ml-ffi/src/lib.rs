//! C ABI over `anofox-ml-registry` for the `anofox_ml` DuckDB extension.
//!
//! Every function returns `NULL` on success or an error message (free it
//! with [`anofox_ml_free_string`]). Output buffers are allocated here and
//! released with the matching `anofox_ml_free_*` function.
//!
//! A model BLOB is `AMLX`, a layout version byte, a `u32` little-endian
//! length plus that many bytes of JSON metadata ([`Meta`]: kind, number of
//! features, classes), then the registry's model bytes (anofox-ml's binary
//! persistence format). Feature matrices and 2-D outputs are row-major
//! `n_rows × n_cols` doubles.

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::ffi::{c_char, CStr, CString};
use std::hash::{Hash, Hasher};

mod conformal;
mod cv;
mod encode;
mod impute;
mod inspect;
mod metrics;
mod select;
mod stats;
mod text;
mod tune;

use anofox_ml_registry::{DynModel, Out};
use ndarray::{ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"AMLX";
const LAYOUT: u8 = 1;

/// Metadata stored in front of the model bytes, readable without decoding
/// the model.
#[derive(Serialize, Deserialize)]
struct Meta {
    kind: String,
    n_features: usize,
    /// Sorted distinct training targets of a model with `predict_proba`:
    /// the column order of its probabilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    classes: Option<Vec<f64>>,
    /// String class labels of a model fitted on a `VARCHAR` target: the
    /// model sees label `i` (sorted order) as the class `i`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    labels: Option<Vec<String>>,
    /// Calibration scores of a conformalized model (see [`conformal`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    conformity: Option<conformal::Conformity>,
    /// Categories of a `VARCHAR` encoder (no registry model behind it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    encoder: Option<encode::EncoderFit>,
}

/// Cached decoded model: (BLOB hash, BLOB length, metadata, model).
type CachedModel = Option<(u64, usize, Meta, Box<dyn DynModel>)>;

/// Prediction methods callable through [`anofox_ml_apply`].
const METHODS: [&str; 8] = [
    "predict",
    "predict_proba",
    "predict_log_proba",
    "decision_function",
    "transform",
    "inverse_transform",
    "score_samples",
    "staged_predict",
];

#[repr(C)]
pub struct AnofoxMlBytes {
    pub data: *mut u8,
    pub len: usize,
}

#[repr(C)]
pub struct AnofoxMlDoubles {
    pub data: *mut f64,
    pub len: usize,
}

/// What [`anofox_ml_model_info`] reports about a model BLOB.
#[repr(C)]
pub struct AnofoxMlModelInfo {
    pub kind: *mut c_char,
    pub n_features: usize,
    /// NULL when the model has no classes.
    pub classes: AnofoxMlDoubles,
    /// Methods the model supports, separated by `\n`.
    pub capabilities: *mut c_char,
    /// Estimator parameters as JSON.
    pub params: *mut c_char,
    pub format_version: u32,
    /// anofox-ml version that wrote the model.
    pub crate_version: *mut c_char,
    /// String class labels (empty unless fitted on string labels).
    pub labels: AnofoxMlStrings,
    /// Training statistics from the fitted state, NaN where the model has
    /// none: iterations run, KMeans inertia, out-of-bag score.
    pub n_iter: f64,
    pub inertia: f64,
    pub oob_score: f64,
}

#[repr(C)]
pub struct AnofoxMlStrings {
    pub data: *mut *mut c_char,
    pub len: usize,
}

fn into_strings<S: AsRef<str>>(v: &[S]) -> AnofoxMlStrings {
    let ptrs: Vec<*mut c_char> = v
        .iter()
        .map(|s| cstring(s.as_ref().replace('\0', " ")))
        .collect();
    let len = ptrs.len();
    AnofoxMlStrings {
        data: Box::into_raw(ptrs.into_boxed_slice()).cast(),
        len,
    }
}

/// Read `n` NUL-terminated strings.
unsafe fn strings<'a>(
    p: *const *const c_char,
    n: usize,
    what: &str,
) -> Result<Vec<&'a str>, String> {
    if p.is_null() {
        return Err(format!("{what} is NULL"));
    }
    std::slice::from_raw_parts(p, n)
        .iter()
        .map(|&s| str_arg(s, what))
        .collect()
}

/// Semi-supervised kinds: they fit on labelled and unlabelled rows, with
/// `y = -1` marking an unlabelled row (`-1` is never a class).
pub(crate) fn accepts_unlabelled(kind: &str) -> bool {
    matches!(
        kind,
        "LabelPropagation" | "LabelSpreading" | "SelfTrainingClassifier"
    )
}

/// Whether `kind` fits on unlabelled rows (see [`accepts_unlabelled`]);
/// `ml_fit_agg` then passes rows with a NULL target as unlabelled.
///
/// # Safety
/// `kind` is NULL or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_accepts_unlabelled(kind: *const c_char) -> bool {
    str_arg(kind, "kind").is_ok_and(accepts_unlabelled)
}

/// [`encode_labels`] where a NULL entry is an unlabelled row (code `-1`).
unsafe fn labels_with_unlabelled(
    y: *const *const c_char,
    n: usize,
) -> Result<(Vec<String>, Vec<f64>), String> {
    if y.is_null() {
        return Err("y is NULL".into());
    }
    let values = std::slice::from_raw_parts(y, n)
        .iter()
        .map(|&p| (!p.is_null()).then(|| str_arg(p, "y")).transpose())
        .collect::<Result<Vec<Option<&str>>, String>>()?;
    let known: Vec<&str> = values.iter().flatten().copied().collect();
    let (labels, _) = encode_labels(&known);
    let codes = values
        .iter()
        .map(|v| match v {
            Some(v) => labels
                .binary_search_by(|l| l.as_str().cmp(v))
                .expect("present") as f64,
            None => -1.0,
        })
        .collect();
    Ok((labels, codes))
}

/// Sorted distinct labels and the code (index) of every value.
pub(crate) fn encode_labels(values: &[&str]) -> (Vec<String>, Vec<f64>) {
    let mut labels: Vec<String> = values.iter().map(|s| s.to_string()).collect();
    labels.sort_unstable();
    labels.dedup();
    let codes = values
        .iter()
        .map(|v| {
            labels
                .binary_search_by(|l| l.as_str().cmp(v))
                .expect("present") as f64
        })
        .collect();
    (labels, codes)
}

/// The process-wide pool fits run in, sized to the thread budget of the
/// query (DuckDB's `threads`, or `anofox_ml_threads`). A DuckDB worker that
/// finalizes a group blocks in `install` while the pool runs the fit, so
/// many groups share the same threads and a single fit can use them all.
static SHARED: std::sync::Mutex<Option<(usize, std::sync::Arc<rayon::ThreadPool>)>> =
    std::sync::Mutex::new(None);

fn shared_pool(threads: usize) -> Result<std::sync::Arc<rayon::ThreadPool>, String> {
    let mut guard = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((n, pool)) = guard.as_ref() {
        if *n == threads {
            return Ok(pool.clone());
        }
    }
    let pool = std::sync::Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("anofox-ml-{i}"))
            .build()
            .map_err(|e| e.to_string())?,
    );
    *guard = Some((threads, pool.clone()));
    Ok(pool)
}

/// Run a fit with up to `threads` threads.
fn run_fit<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> Result<T, String> {
    if threads <= 1 {
        Ok(POOL.with(|p| p.install(f)))
    } else {
        Ok(shared_pool(threads)?.install(f))
    }
}

thread_local! {
    /// One rayon thread per DuckDB worker, for predictions (DuckDB already
    /// runs chunks in parallel) and sequential fits.
    static POOL: rayon::ThreadPool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("one-thread pool");
    /// Last decoded model of this thread, keyed by a hash of its BLOB, so
    /// predicting row by row does not decode the model every time.
    static CACHE: RefCell<CachedModel> = const { RefCell::new(None) };
}

fn err(msg: impl std::fmt::Display) -> *mut c_char {
    CString::new(msg.to_string().replace('\0', " "))
        .expect("no interior NUL")
        .into_raw()
}

unsafe fn str_arg<'a>(p: *const c_char, what: &str) -> Result<&'a str, String> {
    if p.is_null() {
        return Err(format!("{what} is NULL"));
    }
    CStr::from_ptr(p)
        .to_str()
        .map_err(|_| format!("{what} is not valid UTF-8"))
}

unsafe fn matrix<'a>(
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
) -> Result<ArrayView2<'a, f64>, String> {
    if n_rows == 0 || n_cols == 0 {
        return Err("empty feature matrix".into());
    }
    if x.is_null() {
        return Err("feature matrix is NULL".into());
    }
    ArrayView2::from_shape(
        (n_rows, n_cols),
        std::slice::from_raw_parts(x, n_rows * n_cols),
    )
    .map_err(|e| e.to_string())
}

fn encode(meta: &Meta, model: &[u8]) -> Vec<u8> {
    let meta = serde_json::to_vec(meta).expect("metadata serializes");
    let mut out = Vec::with_capacity(9 + meta.len() + model.len());
    out.extend_from_slice(MAGIC);
    out.push(LAYOUT);
    out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
    out.extend_from_slice(&meta);
    out.extend_from_slice(model);
    out
}

fn decode(blob: &[u8]) -> Result<(Meta, &[u8]), String> {
    let rest = blob
        .strip_prefix(MAGIC.as_slice())
        .ok_or("not an anofox_ml model BLOB")?;
    match rest.first() {
        Some(&LAYOUT) => {}
        Some(v) => return Err(format!("unsupported model BLOB layout {v}")),
        None => return Err("truncated model BLOB".into()),
    }
    let n = rest
        .get(1..5)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize)
        .ok_or("truncated model BLOB")?;
    let meta = rest.get(5..5 + n).ok_or("truncated model BLOB")?;
    let meta: Meta =
        serde_json::from_slice(meta).map_err(|e| format!("invalid model BLOB metadata: {e}"))?;
    Ok((meta, &rest[5 + n..]))
}

/// Run `f` on the decoded model of `blob`, decoding it only when it differs
/// from the last model this thread used.
fn with_model<T>(
    blob: &[u8],
    f: impl FnOnce(&Meta, &dyn DynModel) -> Result<T, String>,
) -> Result<T, String> {
    let mut h = DefaultHasher::new();
    blob.hash(&mut h);
    let key = h.finish();
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let hit = matches!(&*cache, Some((k, n, _, _)) if *k == key && *n == blob.len());
        if !hit {
            let (meta, bytes) = decode(blob)?;
            if meta.encoder.is_some() {
                return Err(format!(
                    "{} encodes VARCHAR features: use ml_transform(model, [col, ...]) with VARCHAR values",
                    meta.kind
                ));
            }
            let model = anofox_ml_registry::load(&meta.kind, bytes).map_err(|e| e.to_string())?;
            *cache = Some((key, blob.len(), meta, model));
        }
        let (_, _, meta, model) = cache.as_ref().expect("filled above");
        f(meta, model.as_ref())
    })
}

fn into_doubles(v: Vec<f64>) -> AnofoxMlDoubles {
    let mut b = v.into_boxed_slice();
    let out = AnofoxMlDoubles {
        data: b.as_mut_ptr(),
        len: b.len(),
    };
    std::mem::forget(b);
    out
}

fn fit_model(
    kind: &str,
    params: &str,
    x: ArrayView2<f64>,
    y: Option<ArrayView1<f64>>,
    w: Option<ArrayView1<f64>>,
    threads: usize,
) -> Result<Box<dyn DynModel>, String> {
    let params: serde_json::Value = if params.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(params).map_err(|e| format!("invalid params JSON: {e}"))?
    };
    let mut model = anofox_ml_registry::make(kind, &params).map_err(|e| e.to_string())?;
    run_fit(threads, || model.fit(x, y, w))?.map_err(|e| e.to_string())?;
    Ok(model)
}

/// Run `method` of a fitted model on `x` (`labels` and `embedding`, which
/// describe the training rows, ignore `x`); returns the
/// row-major output and its column count.
fn run_method(
    model: &dyn DynModel,
    method: &str,
    x: ArrayView2<f64>,
) -> Result<(Vec<f64>, usize), String> {
    let out = POOL.with(|p| {
        p.install(|| match method {
            "predict" => model.predict(x),
            "predict_proba" => model.predict_proba(x),
            // Native where the estimator has it (finite where predict_proba
            // underflows); otherwise ln of the probabilities (0 gives -inf).
            "predict_log_proba" if model.capabilities().contains(&"predict_log_proba") => {
                model.predict_log_proba(x)
            }
            "predict_log_proba" => model.predict_proba(x).map(|o| match o {
                Out::A1(a) => Out::A1(a.mapv(f64::ln)),
                Out::A2(a) => Out::A2(a.mapv(f64::ln)),
            }),
            "decision_function" => model.decision_function(x),
            "transform" => model.transform(x),
            "inverse_transform" => model.inverse_transform(x),
            "score_samples" => model.score_samples(x),
            "staged_predict" => model.staged_predict(x),
            "labels" => model.labels(),
            "embedding" => model.embedding(),
            _ => Err(anofox_ml_core::RustMlError::InvalidParameter(format!(
                "unknown method '{method}'"
            ))),
        })
    });
    Ok(flatten(out.map_err(|e| e.to_string())?))
}

/// Row-major values of a registry output and its column count.
fn flatten(out: Out) -> (Vec<f64>, usize) {
    match out {
        Out::A1(a) => (a.to_vec(), 1),
        Out::A2(a) => {
            let cols = a.ncols();
            (a.as_standard_layout().iter().copied().collect(), cols)
        }
    }
}

fn fit_impl(
    kind: &str,
    params: &str,
    x: ArrayView2<f64>,
    y: Option<ArrayView1<f64>>,
    w: Option<ArrayView1<f64>>,
    labels: Option<Vec<String>>,
    threads: usize,
) -> Result<Vec<u8>, String> {
    let model = fit_model(kind, params, x, y, w, threads)?;
    encode_fitted(kind, model.as_ref(), x.ncols(), y, labels)
}

/// The BLOB of a fitted model: metadata (classes from the targets `y` of a
/// model with `predict_proba`, string labels) and the registry bytes.
fn encode_fitted(
    kind: &str,
    model: &dyn DynModel,
    n_features: usize,
    y: Option<ArrayView1<f64>>,
    labels: Option<Vec<String>>,
) -> Result<Vec<u8>, String> {
    let bytes = model.to_bytes().map_err(|e| e.to_string())?;
    let classes = match y {
        Some(y) if model.capabilities().contains(&"predict_proba") => {
            let mut c = y.to_vec();
            if accepts_unlabelled(kind) {
                c.retain(|&v| v != -1.0);
            }
            c.sort_by(f64::total_cmp);
            c.dedup();
            Some(c)
        }
        _ => None,
    };
    let meta = Meta {
        kind: kind.to_string(),
        n_features,
        classes,
        labels,
        conformity: None,
        encoder: None,
    };
    Ok(encode(&meta, &bytes))
}

/// Fit on the rows where `validation` is false, with the others as the
/// validation set of the model's early stopping (`fit_with_validation`).
#[allow(clippy::too_many_arguments)]
fn fit_validated_impl(
    kind: &str,
    params: &str,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    w: Option<ArrayView1<f64>>,
    validation: &[bool],
    labels: Option<Vec<String>>,
    threads: usize,
) -> Result<Vec<u8>, String> {
    let params_json: serde_json::Value = if params.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(params).map_err(|e| format!("invalid params JSON: {e}"))?
    };
    let mut model = anofox_ml_registry::make(kind, &params_json).map_err(|e| e.to_string())?;
    if !model.capabilities().contains(&"fit_with_validation") {
        return Err(format!(
            "{kind} does not support fit_with_validation (an explicit validation set); use ml_fit_agg"
        ));
    }
    let rows =
        |val: bool| -> Vec<usize> { (0..x.nrows()).filter(|&i| validation[i] == val).collect() };
    let (train, val) = (rows(false), rows(true));
    if train.is_empty() || val.is_empty() {
        return Err("fit_with_validation needs both training and validation rows".into());
    }
    let take = |idx: &[usize]| {
        (
            x.select(ndarray::Axis(0), idx),
            y.select(ndarray::Axis(0), idx),
            w.map(|w| w.select(ndarray::Axis(0), idx)),
        )
    };
    let ((xt, yt, wt), (xv, yv, wv)) = (take(&train), take(&val));
    run_fit(threads, || {
        model.fit_with_validation(
            xt.view(),
            Some(yt.view()),
            wt.as_ref().map(|w| w.view()),
            xv.view(),
            Some(yv.view()),
            wv.as_ref().map(|w| w.view()),
        )
    })?
    .map_err(|e| e.to_string())?;
    encode_fitted(kind, model.as_ref(), x.ncols(), Some(y), labels)
}

/// Fit `kind` with an explicit validation set: rows with `validation[i] != 0`
/// are the validation set of the model's early stopping, the others the
/// training set. The target is `y` (`n_rows` doubles), or the string
/// labels `y_labels` when `y` is NULL. `w` may be NULL.
///
/// # Safety
/// As [`anofox_ml_fit`]; `validation` holds `n_rows` bytes, `y_labels`
/// (when used) `n_rows` NUL-terminated strings.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_fit_validation(
    kind: *const c_char,
    params: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const f64,
    y_labels: *const *const c_char,
    w: *const f64,
    validation: *const u8,
    threads: usize,
    out: *mut AnofoxMlBytes,
) -> *mut c_char {
    let run = || -> Result<Vec<u8>, String> {
        let kind = str_arg(kind, "kind")?;
        let params = if params.is_null() {
            ""
        } else {
            str_arg(params, "params")?
        };
        let x = matrix(x, n_rows, n_cols)?;
        if validation.is_null() {
            return Err("validation flags are NULL".into());
        }
        let validation: Vec<bool> = std::slice::from_raw_parts(validation, n_rows)
            .iter()
            .map(|&v| v != 0)
            .collect();
        let w = (!w.is_null()).then(|| ArrayView1::from(std::slice::from_raw_parts(w, n_rows)));
        if y.is_null() {
            let (labels, codes) = encode_labels(&strings(y_labels, n_rows, "y")?);
            fit_validated_impl(
                kind,
                params,
                x,
                ArrayView1::from(&codes),
                w,
                &validation,
                Some(labels),
                threads,
            )
        } else {
            let y = ArrayView1::from(std::slice::from_raw_parts(y, n_rows));
            fit_validated_impl(kind, params, x, y, w, &validation, None, threads)
        }
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
        Err(_) => err("internal error (panic) while fitting"),
    }
}

/// The `k` nearest training rows of every row of `x` (the kNN kinds;
/// `k = 0`: the model's `n_neighbors`): `dist` and `idx` receive
/// `n_rows * k_out` values each, row-major, nearest first; `idx` holds
/// 0-based training-row positions.
///
/// # Safety
/// As [`anofox_ml_predict`]; `dist`, `idx` and `k_out` are writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_kneighbors(
    model: *const u8,
    model_len: usize,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    k: usize,
    dist: *mut AnofoxMlDoubles,
    idx: *mut AnofoxMlDoubles,
    k_out: *mut usize,
) -> *mut c_char {
    let run = || -> Result<(Vec<f64>, Vec<f64>, usize), String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        let x = matrix(x, n_rows, n_cols)?;
        with_model(blob, |meta, model| {
            require_method(&meta.kind, model, "kneighbors")?;
            if x.ncols() != meta.n_features {
                return Err(format!(
                    "{} was fitted on {} features, got {}",
                    meta.kind,
                    meta.n_features,
                    x.ncols()
                ));
            }
            let (d, i) = POOL
                .with(|p| p.install(|| model.kneighbors(x, (k > 0).then_some(k))))
                .map_err(|e| e.to_string())?;
            let cols = d.ncols();
            Ok((
                d.as_standard_layout().iter().copied().collect(),
                i.as_standard_layout().iter().map(|&v| v as f64).collect(),
                cols,
            ))
        })
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok((d, i, cols))) => {
            *dist = into_doubles(d);
            *idx = into_doubles(i);
            *k_out = cols;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while searching neighbours"),
    }
}

/// Continue training a fitted model BLOB on new rows, after merging the JSON
/// `params`: `warm_start` (more stages / epochs, e.g. a larger
/// `n_estimators`) or `partial_fit` (one more batch; no targets for the
/// unsupervised kinds). Targets are numbers, or the model's string labels in
/// `labels`.
fn continue_training(
    method: &str,
    blob: &[u8],
    params: &str,
    x: ArrayView2<f64>,
    y: Option<ArrayView1<f64>>,
    labels: Option<&[&str]>,
    threads: usize,
) -> Result<Vec<u8>, String> {
    let (mut meta, bytes) = decode(blob)?;
    if meta.encoder.is_some() {
        return Err(format!(
            "{} is an encoder and cannot be trained further",
            meta.kind
        ));
    }
    let mut model = anofox_ml_registry::load(&meta.kind, bytes).map_err(|e| e.to_string())?;
    require_method(&meta.kind, model.as_ref(), method)?;
    if x.ncols() != meta.n_features {
        return Err(format!(
            "{} was fitted on {} features, got {}",
            meta.kind,
            meta.n_features,
            x.ncols()
        ));
    }
    // Targets in the model's coding: label indices, and only known classes.
    let codes: Option<Vec<f64>> = match (&meta.labels, labels, y) {
        (Some(known), Some(new), _) => new
            .iter()
            .map(|l| {
                known
                    .iter()
                    .position(|k| k == l)
                    .map(|i| i as f64)
                    .ok_or_else(|| format!("label '{l}' is not a label of the model ({known:?})"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)?,
        (Some(_), None, _) => {
            return Err("the model was fitted on VARCHAR labels; pass VARCHAR targets".into())
        }
        (None, Some(_), _) => {
            return Err("the model was fitted on numeric targets; pass DOUBLE targets".into())
        }
        (None, None, Some(y)) => Some(y.to_vec()),
        (None, None, None) if method == "partial_fit" => None,
        (None, None, None) => return Err(format!("{method} needs targets")),
    };
    if let (Some(classes), Some(codes)) = (&meta.classes, &codes) {
        if let Some(v) = codes.iter().find(|v| !classes.contains(v)) {
            return Err(format!("target {v} is not a class of the model"));
        }
    }
    if !params.trim().is_empty() {
        let p: serde_json::Value =
            serde_json::from_str(params).map_err(|e| format!("invalid params JSON: {e}"))?;
        model.set_params(&p).map_err(|e| e.to_string())?;
    }
    // The model's methods take x and y with one lifetime: both owned here.
    let x = x.to_owned();
    let y = codes.as_ref().map(ArrayView1::from);
    run_fit(threads, || match method {
        "warm_start" => model.warm_start(x.view(), y, None),
        _ => model.partial_fit(x.view(), y, None, None),
    })?
    .map_err(|e| e.to_string())?;
    // Calibration scores belong to the old model.
    meta.conformity = None;
    Ok(encode(&meta, &model.to_bytes().map_err(|e| e.to_string())?))
}

/// Continue training `model` on `n_rows` new rows (`warm_start`; see [`continue_training`]):
/// targets `y` (doubles), or the string labels `y_labels` when `y` is NULL.
///
/// # Safety
/// `model` holds `model_len` bytes, `x` `n_rows * n_cols` doubles, `y`
/// `n_rows` doubles or `y_labels` `n_rows` NUL-terminated strings; `params`
/// is NULL or NUL-terminated; `out` is writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_warm_start(
    model: *const u8,
    model_len: usize,
    params: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const f64,
    y_labels: *const *const c_char,
    threads: usize,
    out: *mut AnofoxMlBytes,
) -> *mut c_char {
    let run = || -> Result<Vec<u8>, String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        let params = if params.is_null() {
            ""
        } else {
            str_arg(params, "params")?
        };
        let x = matrix(x, n_rows, n_cols)?;
        if y.is_null() {
            let labels = strings(y_labels, n_rows, "y")?;
            continue_training("warm_start", blob, params, x, None, Some(&labels), threads)
        } else {
            let y = ArrayView1::from(std::slice::from_raw_parts(y, n_rows));
            continue_training("warm_start", blob, params, x, Some(y), None, threads)
        }
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
        Err(_) => err("internal error (panic) while warm-starting"),
    }
}

/// Fold `n_rows` new rows into `model` as one more batch (`partial_fit`; see
/// [`continue_training`]): targets `y` (doubles), the string labels
/// `y_labels` when `y` is NULL, or none (both NULL) for the unsupervised kinds.
///
/// # Safety
/// `model` holds `model_len` bytes, `x` `n_rows * n_cols` doubles, `y`
/// `n_rows` doubles or `y_labels` `n_rows` NUL-terminated strings; `params`
/// is NULL or NUL-terminated; `out` is writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_partial_fit(
    model: *const u8,
    model_len: usize,
    params: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const f64,
    y_labels: *const *const c_char,
    threads: usize,
    out: *mut AnofoxMlBytes,
) -> *mut c_char {
    let run = || -> Result<Vec<u8>, String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        let params = if params.is_null() {
            ""
        } else {
            str_arg(params, "params")?
        };
        let x = matrix(x, n_rows, n_cols)?;
        if !y.is_null() {
            let y = ArrayView1::from(std::slice::from_raw_parts(y, n_rows));
            continue_training("partial_fit", blob, params, x, Some(y), None, threads)
        } else if !y_labels.is_null() {
            let labels = strings(y_labels, n_rows, "y")?;
            continue_training("partial_fit", blob, params, x, None, Some(&labels), threads)
        } else {
            continue_training("partial_fit", blob, params, x, None, None, threads)
        }
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
        Err(_) => err("internal error (panic) while fitting a batch"),
    }
}

/// Fit, then run the first of the comma-separated `methods` the model
/// supports on the training data.
fn fit_apply_impl(
    kind: &str,
    params: &str,
    x: ArrayView2<f64>,
    y: Option<ArrayView1<f64>>,
    w: Option<ArrayView1<f64>>,
    methods: &str,
    threads: usize,
) -> Result<(Vec<f64>, usize), String> {
    let model = fit_model(kind, params, x, y, w, threads)?;
    let caps = model.capabilities();
    let wanted: Vec<&str> = methods.split(',').map(str::trim).collect();
    let method = wanted
        .iter()
        .find(|m| caps.contains(m))
        .ok_or_else(|| format!("{kind} supports none of {wanted:?}"))?;
    let (v, cols) = run_method(model.as_ref(), method, x)?;
    if v.len() != x.nrows() * cols {
        return Err(format!(
            "{kind}.{method} returned {} values for {} rows",
            v.len(),
            x.nrows()
        ));
    }
    Ok((v, cols))
}

/// Fit `kind` (with `y`, `w` when non-NULL) and run the first supported of
/// the comma-separated `methods` (e.g. `labels,predict`) on the training
/// rows; `out` receives `n_rows * out_cols` doubles, row-major. The model is
/// not kept.
///
/// # Safety
/// As [`anofox_ml_fit`]; `methods` is a NUL-terminated string and
/// `out` / `out_cols` are writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_fit_apply(
    kind: *const c_char,
    params: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const f64,
    w: *const f64,
    methods: *const c_char,
    threads: usize,
    out: *mut AnofoxMlDoubles,
    out_cols: *mut usize,
) -> *mut c_char {
    let run = || -> Result<(Vec<f64>, usize), String> {
        let kind = str_arg(kind, "kind")?;
        let params = if params.is_null() {
            ""
        } else {
            str_arg(params, "params")?
        };
        let methods = str_arg(methods, "methods")?;
        let x = matrix(x, n_rows, n_cols)?;
        let y = (!y.is_null()).then(|| ArrayView1::from(std::slice::from_raw_parts(y, n_rows)));
        let w = (!w.is_null()).then(|| ArrayView1::from(std::slice::from_raw_parts(w, n_rows)));
        fit_apply_impl(kind, params, x, y, w, methods, threads)
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok((v, cols))) => {
            *out = into_doubles(v);
            *out_cols = cols;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while fitting"),
    }
}

/// Fit `kind` with JSON `params` on `x` (and `y`, `w` when non-NULL) with up
/// to `threads` threads (0 or 1 = sequential) and return the model BLOB in
/// `out`.
///
/// # Safety
/// `kind` / `params` are NUL-terminated strings (`params` may be NULL);
/// `x` holds `n_rows * n_cols` doubles, `y` / `w` hold `n_rows` doubles or
/// are NULL; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_fit(
    kind: *const c_char,
    params: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const f64,
    w: *const f64,
    threads: usize,
    out: *mut AnofoxMlBytes,
) -> *mut c_char {
    let run = || -> Result<Vec<u8>, String> {
        let kind = str_arg(kind, "kind")?;
        let params = if params.is_null() {
            ""
        } else {
            str_arg(params, "params")?
        };
        let x = matrix(x, n_rows, n_cols)?;
        let y = (!y.is_null()).then(|| ArrayView1::from(std::slice::from_raw_parts(y, n_rows)));
        let w = (!w.is_null()).then(|| ArrayView1::from(std::slice::from_raw_parts(w, n_rows)));
        fit_impl(kind, params, x, y, w, None, threads)
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
        Err(_) => err("internal error (panic) while fitting"),
    }
}

/// [`anofox_ml_fit`] for a classifier with string labels `y` (`n_rows`
/// NUL-terminated strings): the labels are stored in the BLOB and the model
/// is fitted on their sorted index.
///
/// # Safety
/// As [`anofox_ml_fit`]; `y` holds `n_rows` NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_fit_labels(
    kind: *const c_char,
    params: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const *const c_char,
    w: *const f64,
    threads: usize,
    out: *mut AnofoxMlBytes,
) -> *mut c_char {
    let run = || -> Result<Vec<u8>, String> {
        let kind = str_arg(kind, "kind")?;
        let params = if params.is_null() {
            ""
        } else {
            str_arg(params, "params")?
        };
        let x = matrix(x, n_rows, n_cols)?;
        let (labels, codes) = if accepts_unlabelled(kind) {
            labels_with_unlabelled(y, n_rows)?
        } else {
            encode_labels(&strings(y, n_rows, "y")?)
        };
        let w = (!w.is_null()).then(|| ArrayView1::from(std::slice::from_raw_parts(w, n_rows)));
        fit_impl(
            kind,
            params,
            x,
            Some(ArrayView1::from(&codes)),
            w,
            Some(labels),
            threads,
        )
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
        Err(_) => err("internal error (panic) while fitting"),
    }
}

/// A class value as text: its label for a model fitted on string labels,
/// otherwise the number (`1`, not `1.0`, for integral values).
fn label_of(meta: &Meta, v: f64) -> Result<String, String> {
    match &meta.labels {
        Some(labels) => {
            let i = v.round();
            if v.is_finite() && i >= 0.0 && (i as usize) < labels.len() {
                Ok(labels[i as usize].clone())
            } else {
                Err(format!(
                    "{} predicted {v}, which is not a label index",
                    meta.kind
                ))
            }
        }
        None if v.fract() == 0.0 && v.abs() < 1e15 => Ok(format!("{}", v as i64)),
        None => Ok(v.to_string()),
    }
}

/// Predict `n_rows` rows of `x` and return the class labels as strings.
///
/// # Safety
/// As [`anofox_ml_predict`]; release `out` with [`anofox_ml_free_strings`].
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_predict_labels(
    model: *const u8,
    model_len: usize,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    out: *mut AnofoxMlStrings,
) -> *mut c_char {
    let run = || -> Result<Vec<String>, String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        let x = matrix(x, n_rows, n_cols)?;
        let (v, cols) = apply_impl(blob, "predict", x)?;
        if cols != 1 {
            return Err(format!("predict returned {cols} columns"));
        }
        let (meta, _) = decode(blob)?;
        v.iter().map(|&p| label_of(&meta, p)).collect()
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(v)) => {
            *out = into_strings(&v);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while predicting"),
    }
}

/// # Safety
/// `s` was returned by this library.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_free_strings(s: AnofoxMlStrings) {
    if !s.data.is_null() {
        let v = Box::from_raw(std::ptr::slice_from_raw_parts_mut(s.data, s.len));
        for p in v.iter() {
            anofox_ml_free_string(*p);
        }
    }
}

/// Apply `method` to `x`; returns the row-major output and its column count.
/// Methods of every registered kind (static per kind), in registry order.
fn kind_capabilities() -> &'static [(&'static str, Vec<&'static str>)] {
    static CAPS: std::sync::OnceLock<Vec<(&'static str, Vec<&'static str>)>> =
        std::sync::OnceLock::new();
    CAPS.get_or_init(|| {
        anofox_ml_registry::kinds()
            .into_iter()
            .filter_map(|k| {
                let m = anofox_ml_registry::make(k, &serde_json::json!({})).ok()?;
                Some((k, m.capabilities()))
            })
            .collect()
    })
}

/// `Ok` when `model` has `method` (`predict_log_proba` needs
/// `predict_proba`); otherwise an error naming what it supports and the
/// kinds that have the method.
fn require_method(kind: &str, model: &dyn DynModel, method: &str) -> Result<(), String> {
    let cap = match method {
        "predict_log_proba" => "predict_proba",
        m => m,
    };
    let caps = model.capabilities();
    if caps.contains(&cap) {
        return Ok(());
    }
    let with: Vec<&str> = kind_capabilities()
        .iter()
        .filter(|(_, c)| c.contains(&cap))
        .map(|(k, _)| *k)
        .collect();
    let examples = with.iter().take(5).copied().collect::<Vec<_>>().join(", ");
    Err(format!(
        "{kind} does not support {cap} (it supports: {}). {} kinds do, e.g. {examples}; list them with \
         SELECT kind FROM ml_estimators() WHERE list_contains(capabilities, '{cap}')",
        caps.join(", "),
        with.len()
    ))
}

fn apply_impl(blob: &[u8], method: &str, x: ArrayView2<f64>) -> Result<(Vec<f64>, usize), String> {
    if !METHODS.contains(&method) {
        return Err(format!("unknown method '{method}' (known: {METHODS:?})"));
    }
    with_model(blob, |meta, model| {
        require_method(&meta.kind, model, method)?;
        if x.ncols() != meta.n_features {
            return Err(format!(
                "{} was fitted on {} features, got {}",
                meta.kind,
                meta.n_features,
                x.ncols()
            ));
        }
        run_method(model, method, x)
    })
}

/// Apply `method` (`predict`, `predict_proba`, `decision_function`,
/// `transform`, `inverse_transform` or `score_samples`) of a model BLOB to
/// `n_rows` rows of `x`. `out` receives `n_rows * out_cols` doubles, row-major.
///
/// # Safety
/// `model` holds `model_len` bytes, `method` is a NUL-terminated string,
/// `x` holds `n_rows * n_cols` doubles, `out` / `out_cols` are writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_apply(
    model: *const u8,
    model_len: usize,
    method: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    out: *mut AnofoxMlDoubles,
    out_cols: *mut usize,
) -> *mut c_char {
    let run = || -> Result<(Vec<f64>, usize), String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let method = str_arg(method, "method")?;
        let blob = std::slice::from_raw_parts(model, model_len);
        apply_impl(blob, method, matrix(x, n_rows, n_cols)?)
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok((v, cols))) => {
            *out = into_doubles(v);
            *out_cols = cols;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while predicting"),
    }
}

/// Predict with a model BLOB on `n_rows` rows of `x`; `out` receives
/// `n_rows` doubles.
///
/// # Safety
/// `model` holds `model_len` bytes, `x` holds `n_rows * n_cols` doubles,
/// `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_predict(
    model: *const u8,
    model_len: usize,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    out: *mut AnofoxMlDoubles,
) -> *mut c_char {
    let run = || -> Result<Vec<f64>, String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        match apply_impl(blob, "predict", matrix(x, n_rows, n_cols)?)? {
            (v, 1) => Ok(v),
            (_, cols) => Err(format!("predict returned {cols} columns")),
        }
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

/// JSON array of `{"kind": ..., "default_params": ...}` for every
/// registered estimator.
///
/// # Safety
/// `out` is writable; free the string with [`anofox_ml_free_string`].
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_estimators_json(out: *mut *mut c_char) -> *mut c_char {
    let list: Vec<serde_json::Value> = anofox_ml_registry::kinds()
        .into_iter()
        .map(|k| {
            let params = anofox_ml_registry::default_params(k).unwrap_or(serde_json::Value::Null);
            serde_json::json!({ "kind": k, "default_params": params })
        })
        .collect();
    *out = err(serde_json::Value::Array(list));
    std::ptr::null_mut()
}

/// Methods of `kind` (as `ml_model_info(model).capabilities` of a fitted
/// model), separated by `\n`.
///
/// # Safety
/// `kind` is a NUL-terminated string; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_kind_capabilities(
    kind: *const c_char,
    out: *mut *mut c_char,
) -> *mut c_char {
    match str_arg(kind, "kind").and_then(|k| {
        kind_capabilities()
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, c)| c.join("\n"))
            .ok_or_else(|| format!("unknown kind '{k}'"))
    }) {
        Ok(c) => {
            *out = err(c);
            std::ptr::null_mut()
        }
        Err(e) => err(e),
    }
}

/// Registered estimator kinds, separated by `\n`.
///
/// # Safety
/// `out` is writable; free the string with [`anofox_ml_free_string`].
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_kinds(out: *mut *mut c_char) -> *mut c_char {
    *out = err(anofox_ml_registry::kinds().join("\n"));
    std::ptr::null_mut()
}

/// Default parameters of `kind` as JSON.
///
/// # Safety
/// `kind` is a NUL-terminated string; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_default_params(
    kind: *const c_char,
    out: *mut *mut c_char,
) -> *mut c_char {
    let run = || -> Result<String, String> {
        let kind = str_arg(kind, "kind")?;
        anofox_ml_registry::default_params(kind)
            .map(|v| v.to_string())
            .map_err(|e| e.to_string())
    };
    match run() {
        Ok(json) => {
            *out = err(json);
            std::ptr::null_mut()
        }
        Err(e) => err(e),
    }
}

/// The model kind stored in a BLOB.
///
/// # Safety
/// `model` holds `model_len` bytes; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_model_kind(
    model: *const u8,
    model_len: usize,
    out: *mut *mut c_char,
) -> *mut c_char {
    if model.is_null() {
        return err("model is NULL");
    }
    match decode(std::slice::from_raw_parts(model, model_len)) {
        Ok((meta, _)) => {
            *out = err(meta.kind);
            std::ptr::null_mut()
        }
        Err(e) => err(e),
    }
}

/// The fitted state of a model BLOB as JSON (coefficients, importances,
/// centers, ...; the layout is that of the estimator's fitted type).
///
/// # Safety
/// `model` holds `model_len` bytes; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_model_fitted(
    model: *const u8,
    model_len: usize,
    out: *mut *mut c_char,
) -> *mut c_char {
    let run = || -> Result<String, String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        if let Ok(e) = encode::encoder_of(blob) {
            return Ok(serde_json::json!({ "categories": e.categories }).to_string());
        }
        with_model(blob, |_, model| {
            model.fitted_json().map_err(|e| e.to_string())
        })
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(json)) => {
            *out = err(json);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while reading the model"),
    }
}

/// The first fitted array found under one of `keys` (tried in order, each
/// over the whole fitted state, depth first): `(rows, cols, data)`. ndarray
/// values (`{"v", "dim", "data"}`), plain number lists and scalars count.
fn fitted_array(json: &serde_json::Value, keys: &[&str]) -> Option<(usize, usize, Vec<f64>)> {
    fn as_array(v: &serde_json::Value) -> Option<(usize, usize, Vec<f64>)> {
        match v {
            serde_json::Value::Number(n) => Some((1, 1, vec![n.as_f64()?])),
            serde_json::Value::Array(a) => {
                let d: Option<Vec<f64>> = a.iter().map(serde_json::Value::as_f64).collect();
                d.map(|d| (1, d.len(), d))
            }
            serde_json::Value::Object(o) => {
                let dim: Vec<usize> = o
                    .get("dim")?
                    .as_array()?
                    .iter()
                    .filter_map(|d| d.as_u64())
                    .map(|d| d as usize)
                    .collect();
                let data: Vec<f64> = o
                    .get("data")?
                    .as_array()?
                    .iter()
                    .filter_map(serde_json::Value::as_f64)
                    .collect();
                match dim.as_slice() {
                    [n] => Some((1, *n, data)),
                    [r, c] => Some((*r, *c, data)),
                    _ => None,
                }
            }
            _ => None,
        }
    }
    fn find<'a>(v: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
        match v {
            serde_json::Value::Object(o) => {
                o.get(key).or_else(|| o.values().find_map(|c| find(c, key)))
            }
            serde_json::Value::Array(a) => a.iter().find_map(|c| find(c, key)),
            _ => None,
        }
    }
    keys.iter().find_map(|k| find(json, k).and_then(as_array))
}

/// A fitted array of a model BLOB under the first of the comma-separated
/// `keys` it has (see [`fitted_array`]); `out` receives `rows * cols`
/// doubles, row-major. The key `feature_importances` calls the model's
/// `feature_importances` method instead.
///
/// # Safety
/// `model` holds `model_len` bytes, `keys` is NUL-terminated; the outputs
/// are writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_model_array(
    model: *const u8,
    model_len: usize,
    keys: *const c_char,
    out: *mut AnofoxMlDoubles,
    rows: *mut usize,
    cols: *mut usize,
) -> *mut c_char {
    let run = || -> Result<(usize, usize, Vec<f64>), String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let keys: Vec<&str> = str_arg(keys, "keys")?.split(',').collect();
        let blob = std::slice::from_raw_parts(model, model_len);
        with_model(blob, |meta, model| {
            if keys == ["feature_importances"] {
                require_method(&meta.kind, model, "feature_importances")?;
                let (v, cols) = flatten(model.feature_importances().map_err(|e| e.to_string())?);
                return Ok((v.len() / cols.max(1), cols, v));
            }
            let json: serde_json::Value =
                serde_json::from_str(&model.fitted_json().map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            fitted_array(&json, &keys)
                .ok_or_else(|| format!("{} has no {}", meta.kind, keys.join(" / ")))
        })
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok((r, c, v))) => {
            *out = into_doubles(v);
            *rows = r;
            *cols = c;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while reading the model"),
    }
}

fn cstring(s: impl Into<Vec<u8>>) -> *mut c_char {
    CString::new(s).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// Describe a model BLOB: kind, feature count, classes, supported methods,
/// parameters and the persistence header.
///
/// # Safety
/// `model` holds `model_len` bytes; `out` is writable. Release it with
/// [`anofox_ml_free_model_info`].
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_model_info(
    model: *const u8,
    model_len: usize,
    out: *mut AnofoxMlModelInfo,
) -> *mut c_char {
    let run = || -> Result<AnofoxMlModelInfo, String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        let (meta, bytes) = decode(blob)?;
        if let Some(e) = &meta.encoder {
            return Ok(AnofoxMlModelInfo {
                kind: cstring(meta.kind.as_str()),
                n_features: meta.n_features,
                classes: AnofoxMlDoubles {
                    data: std::ptr::null_mut(),
                    len: 0,
                },
                capabilities: cstring("transform"),
                params: cstring(e.params_json().to_string()),
                format_version: 0,
                crate_version: cstring(env!("CARGO_PKG_VERSION")),
                labels: into_strings::<String>(&[]),
                n_iter: f64::NAN,
                inertia: f64::NAN,
                oob_score: f64::NAN,
            });
        }
        let header =
            anofox_ml_core::persistence::binary_header(bytes).map_err(|e| e.to_string())?;
        with_model(blob, |meta, model| {
            let params = model.params().map_err(|e| e.to_string())?;
            let fitted = model
                .fitted_json()
                .ok()
                .and_then(|j| serde_json::from_str::<serde_json::Value>(&j).ok());
            let stat = |keys: &[&str]| {
                fitted
                    .as_ref()
                    .and_then(|j| fitted_array(j, keys))
                    .filter(|(r, c, _)| r * c == 1)
                    .map_or(f64::NAN, |(_, _, v)| v[0])
            };
            Ok(AnofoxMlModelInfo {
                kind: cstring(meta.kind.as_str()),
                n_features: meta.n_features,
                classes: meta.classes.clone().map_or(
                    AnofoxMlDoubles {
                        data: std::ptr::null_mut(),
                        len: 0,
                    },
                    into_doubles,
                ),
                capabilities: cstring(model.capabilities().join("\n")),
                params: cstring(params.to_string()),
                format_version: header.as_ref().map_or(0, |h| h.format_version),
                crate_version: cstring(header.map(|h| h.crate_version).unwrap_or_default()),
                labels: into_strings(meta.labels.as_deref().unwrap_or_default()),
                n_iter: stat(&["n_iter", "n_iter_"]),
                inertia: stat(&["inertia"]),
                oob_score: stat(&["oob_score_value", "oob_score_"]),
            })
        })
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(info)) => {
            *out = info;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while reading the model"),
    }
}

/// # Safety
/// `info` was filled by [`anofox_ml_model_info`].
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_free_model_info(info: AnofoxMlModelInfo) {
    anofox_ml_free_string(info.kind);
    anofox_ml_free_doubles(info.classes);
    anofox_ml_free_string(info.capabilities);
    anofox_ml_free_string(info.params);
    anofox_ml_free_string(info.crate_version);
    anofox_ml_free_strings(info.labels);
}

/// Version of this library (NUL-terminated, static).
#[no_mangle]
pub extern "C" fn anofox_ml_ffi_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// # Safety
/// `s` was returned by this library (or is NULL).
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// # Safety
/// `b` was returned by this library.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_free_bytes(b: AnofoxMlBytes) {
    if !b.data.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            b.data, b.len,
        )));
    }
}

/// # Safety
/// `d` was returned by this library.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_free_doubles(d: AnofoxMlDoubles) {
    if !d.data.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            d.data, d.len,
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_predict_round_trip_through_c_abi() {
        let x: Vec<f64> = (0..20)
            .flat_map(|i| [(i % 10) as f64, ((i * 3) % 7) as f64])
            .collect(); // 20 x 2
        let y: Vec<f64> = (0..20)
            .map(|i| if i % 10 >= 5 { 1.0 } else { 0.0 })
            .collect();
        let kind = CString::new("RandomForestClassifier").unwrap();
        let params = CString::new(r#"{"n_estimators": 10}"#).unwrap();
        let mut blob = AnofoxMlBytes {
            data: std::ptr::null_mut(),
            len: 0,
        };
        unsafe {
            let e = anofox_ml_fit(
                kind.as_ptr(),
                params.as_ptr(),
                x.as_ptr(),
                20,
                2,
                y.as_ptr(),
                std::ptr::null(),
                1,
                &mut blob,
            );
            assert!(e.is_null(), "{}", CStr::from_ptr(e).to_str().unwrap());
            let mut pred = AnofoxMlDoubles {
                data: std::ptr::null_mut(),
                len: 0,
            };
            let e = anofox_ml_predict(blob.data, blob.len, x.as_ptr(), 20, 2, &mut pred);
            assert!(e.is_null());
            assert_eq!(
                std::slice::from_raw_parts(pred.data, pred.len),
                y.as_slice()
            );
            let mut k = std::ptr::null_mut();
            assert!(anofox_ml_model_kind(blob.data, blob.len, &mut k).is_null());
            assert_eq!(
                CStr::from_ptr(k).to_str().unwrap(),
                "RandomForestClassifier"
            );
            anofox_ml_free_string(k);
            anofox_ml_free_doubles(pred);
            anofox_ml_free_bytes(blob);
        }
    }

    #[test]
    fn errors_are_reported_not_panicked() {
        let kind = CString::new("NoSuchModel").unwrap();
        let x = [1.0, 2.0];
        let mut blob = AnofoxMlBytes {
            data: std::ptr::null_mut(),
            len: 0,
        };
        unsafe {
            let e = anofox_ml_fit(
                kind.as_ptr(),
                std::ptr::null(),
                x.as_ptr(),
                2,
                1,
                std::ptr::null(),
                std::ptr::null(),
                1,
                &mut blob,
            );
            assert!(!e.is_null());
            anofox_ml_free_string(e);
            let mut pred = AnofoxMlDoubles {
                data: std::ptr::null_mut(),
                len: 0,
            };
            let e = anofox_ml_predict(b"garbage".as_ptr(), 7, x.as_ptr(), 2, 1, &mut pred);
            assert!(CStr::from_ptr(e)
                .to_str()
                .unwrap()
                .contains("not an anofox_ml model"));
            anofox_ml_free_string(e);
        }
    }

    #[test]
    fn estimators_json_lists_every_kind() {
        unsafe {
            let mut s = std::ptr::null_mut();
            assert!(anofox_ml_estimators_json(&mut s).is_null());
            let v: serde_json::Value =
                serde_json::from_str(CStr::from_ptr(s).to_str().unwrap()).unwrap();
            assert!(v.as_array().unwrap().len() > 100);
            anofox_ml_free_string(s);
        }
    }

    fn fit(kind: &str, params: &str, x: &[f64], n_cols: usize, y: Option<&[f64]>) -> AnofoxMlBytes {
        let kind = CString::new(kind).unwrap();
        let params = CString::new(params).unwrap();
        let mut blob = AnofoxMlBytes {
            data: std::ptr::null_mut(),
            len: 0,
        };
        unsafe {
            let e = anofox_ml_fit(
                kind.as_ptr(),
                params.as_ptr(),
                x.as_ptr(),
                x.len() / n_cols,
                n_cols,
                y.map_or(std::ptr::null(), <[f64]>::as_ptr),
                std::ptr::null(),
                1,
                &mut blob,
            );
            assert!(e.is_null(), "{}", CStr::from_ptr(e).to_str().unwrap());
        }
        blob
    }

    fn apply(
        blob: &AnofoxMlBytes,
        method: &str,
        x: &[f64],
        n_cols: usize,
    ) -> Result<(Vec<f64>, usize), String> {
        let method = CString::new(method).unwrap();
        let mut out = AnofoxMlDoubles {
            data: std::ptr::null_mut(),
            len: 0,
        };
        let mut cols = 0;
        unsafe {
            let e = anofox_ml_apply(
                blob.data,
                blob.len,
                method.as_ptr(),
                x.as_ptr(),
                x.len() / n_cols,
                n_cols,
                &mut out,
                &mut cols,
            );
            if !e.is_null() {
                let msg = CStr::from_ptr(e).to_str().unwrap().to_string();
                anofox_ml_free_string(e);
                return Err(msg);
            }
            let v = std::slice::from_raw_parts(out.data, out.len).to_vec();
            anofox_ml_free_doubles(out);
            Ok((v, cols))
        }
    }

    #[test]
    fn predict_proba_columns_follow_the_stored_classes() {
        // Three classes with non-contiguous labels.
        let x: Vec<f64> = (0..30).map(|i| (i % 15) as f64).collect();
        let y: Vec<f64> = (0..30).map(|i| [7.0, -1.0, 3.0][(i % 15) / 5]).collect();
        let blob = fit(
            "RandomForestClassifier",
            r#"{"n_estimators": 10}"#,
            &x,
            1,
            Some(&y),
        );
        let (proba, cols) = apply(&blob, "predict_proba", &x, 1).unwrap();
        let (pred, _) = apply(&blob, "predict", &x, 1).unwrap();
        assert_eq!(cols, 3);
        unsafe {
            let mut info = std::mem::zeroed::<AnofoxMlModelInfo>();
            assert!(anofox_ml_model_info(blob.data, blob.len, &mut info).is_null());
            let classes = std::slice::from_raw_parts(info.classes.data, info.classes.len);
            assert_eq!(classes, [-1.0, 3.0, 7.0]);
            for (row, p) in proba.chunks(cols).zip(&pred) {
                let best = (0..cols)
                    .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                    .unwrap();
                assert_eq!(classes[best], *p);
            }
            assert_eq!(info.n_features, 1);
            assert_eq!(info.format_version, 2);
            assert_eq!(
                CStr::from_ptr(info.kind).to_str().unwrap(),
                "RandomForestClassifier"
            );
            let caps = CStr::from_ptr(info.capabilities).to_str().unwrap();
            assert!(caps.split('\n').any(|c| c == "predict_proba"), "{caps}");
            let params: serde_json::Value =
                serde_json::from_str(CStr::from_ptr(info.params).to_str().unwrap()).unwrap();
            assert_eq!(params["n_estimators"], 10);
            anofox_ml_free_model_info(info);
            anofox_ml_free_bytes(blob);
        }
    }

    #[test]
    fn transform_returns_a_row_major_matrix() {
        let x = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let blob = fit("StandardScaler", "", &x, 2, None);
        let (t, cols) = apply(&blob, "transform", &x, 2).unwrap();
        assert_eq!(cols, 2);
        // Columns have mean 2 / 20 and population std sqrt(2/3) / 10 sqrt(2/3).
        let s = 1.5_f64.sqrt();
        let expected = [-s, -s, 0.0, 0.0, s, s];
        for (a, b) in t.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12, "{t:?}");
        }
        unsafe {
            let mut info = std::mem::zeroed::<AnofoxMlModelInfo>();
            assert!(anofox_ml_model_info(blob.data, blob.len, &mut info).is_null());
            assert!(info.classes.data.is_null());
            anofox_ml_free_model_info(info);
            anofox_ml_free_bytes(blob);
        }
    }

    #[test]
    fn wrong_feature_count_and_unknown_method_are_errors() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 0.0, 1.0, 1.0];
        let blob = fit("DecisionTreeRegressor", "", &x, 1, Some(&y));
        let e = apply(&blob, "predict", &x, 2).unwrap_err();
        assert!(e.contains("fitted on 1 features, got 2"), "{e}");
        let e = apply(&blob, "explode", &x, 1).unwrap_err();
        assert!(e.contains("unknown method"), "{e}");
        let e = apply(&blob, "predict_proba", &x, 1).unwrap_err();
        assert!(e.contains("does not support predict_proba"), "{e}");
        unsafe { anofox_ml_free_bytes(blob) };
    }

    fn fit_apply(
        kind: &str,
        params: &str,
        x: &[f64],
        n_cols: usize,
        methods: &str,
    ) -> Result<(Vec<f64>, usize), String> {
        let (kind, params, methods) = (
            CString::new(kind).unwrap(),
            CString::new(params).unwrap(),
            CString::new(methods).unwrap(),
        );
        let mut out = AnofoxMlDoubles {
            data: std::ptr::null_mut(),
            len: 0,
        };
        let mut cols = 0;
        unsafe {
            let e = anofox_ml_fit_apply(
                kind.as_ptr(),
                params.as_ptr(),
                x.as_ptr(),
                x.len() / n_cols,
                n_cols,
                std::ptr::null(),
                std::ptr::null(),
                methods.as_ptr(),
                1,
                &mut out,
                &mut cols,
            );
            if !e.is_null() {
                let msg = CStr::from_ptr(e).to_str().unwrap().to_string();
                anofox_ml_free_string(e);
                return Err(msg);
            }
            let v = std::slice::from_raw_parts(out.data, out.len).to_vec();
            anofox_ml_free_doubles(out);
            Ok((v, cols))
        }
    }

    /// Two well-separated blobs of 10 points each plus one far outlier.
    fn blobs() -> Vec<f64> {
        let mut x = Vec::new();
        for i in 0..10 {
            let d = i as f64 * 0.1;
            x.extend([d, d]);
        }
        for i in 0..10 {
            let d = i as f64 * 0.1;
            x.extend([10.0 + d, 10.0 - d]);
        }
        x.extend([50.0, -50.0]);
        x
    }

    #[test]
    fn fit_apply_returns_training_labels_and_scores() {
        let x = blobs();
        // Dbscan: labels from the fit; the outlier is noise (-1).
        let (labels, cols) = fit_apply(
            "Dbscan",
            r#"{"eps": 0.5, "min_samples": 3}"#,
            &x,
            2,
            "labels,predict",
        )
        .unwrap();
        assert_eq!(cols, 1);
        assert_eq!(labels.len(), 21);
        assert!(labels[..10].iter().all(|&l| l == labels[0]));
        assert!(labels[10..20].iter().all(|&l| l == labels[10]));
        assert_ne!(labels[0], labels[10]);
        assert_eq!(labels[20], -1.0);
        // LocalOutlierFactor: the outlier has the lowest score.
        let (scores, _) = fit_apply(
            "LocalOutlierFactor",
            r#"{"n_neighbors": 5}"#,
            &x,
            2,
            "score_samples,decision_function",
        )
        .unwrap();
        let lowest = (0..21)
            .min_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap();
        assert_eq!(lowest, 20);
        let e = fit_apply("StandardScaler", "", &x, 2, "labels,predict").unwrap_err();
        assert!(e.contains("supports none of"), "{e}");
    }

    #[test]
    fn fit_apply_returns_the_training_embedding() {
        let x = blobs();
        let (emb, cols) = fit_apply(
            "Isomap",
            r#"{"n_neighbors": 5, "n_components": 2}"#,
            &x[..40],
            2,
            "embedding,transform",
        )
        .unwrap();
        assert_eq!((emb.len(), cols), (40, 2));
        assert!(emb.iter().all(|v| v.is_finite()));
        // A transformer without a stored embedding falls back to transform.
        let (t, cols) = fit_apply("StandardScaler", "", &x, 2, "embedding,transform").unwrap();
        assert_eq!((t.len(), cols), (42, 2));
    }

    #[test]
    fn sample_weights_reach_the_estimator() {
        // Weighted mean of [0, 10] with weights [3, 1] is 2.5.
        let x = [0.0, 10.0];
        let w = [3.0, 1.0];
        let kind = CString::new("StandardScaler").unwrap();
        let mut blob = AnofoxMlBytes {
            data: std::ptr::null_mut(),
            len: 0,
        };
        unsafe {
            let e = anofox_ml_fit(
                kind.as_ptr(),
                std::ptr::null(),
                x.as_ptr(),
                2,
                1,
                std::ptr::null(),
                w.as_ptr(),
                1,
                &mut blob,
            );
            assert!(e.is_null(), "{}", CStr::from_ptr(e).to_str().unwrap());
        }
        let (t, _) = apply(&blob, "transform", &[2.5], 1).unwrap();
        assert!(t[0].abs() < 1e-12, "{t:?}");
        unsafe { anofox_ml_free_bytes(blob) };
    }

    #[test]
    fn fitted_state_is_json() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        let blob = fit("DecisionTreeRegressor", "", &x, 1, Some(&y));
        unsafe {
            let mut s = std::ptr::null_mut();
            let e = anofox_ml_model_fitted(blob.data, blob.len, &mut s);
            assert!(e.is_null(), "{}", CStr::from_ptr(e).to_str().unwrap());
            let v: serde_json::Value =
                serde_json::from_str(CStr::from_ptr(s).to_str().unwrap()).unwrap();
            assert!(v.is_object(), "{v}");
            anofox_ml_free_string(s);
            anofox_ml_free_bytes(blob);
        }
    }

    #[test]
    fn string_labels_round_trip() {
        let x: Vec<f64> = (0..30).map(|i| (i % 15) as f64).collect();
        let names = ["setosa", "virginica", "versicolor"];
        let y: Vec<CString> = (0..30)
            .map(|i| CString::new(names[(i % 15) / 5]).unwrap())
            .collect();
        let yp: Vec<*const c_char> = y.iter().map(|s| s.as_ptr()).collect();
        let kind = CString::new("DecisionTreeClassifier").unwrap();
        let mut blob = AnofoxMlBytes {
            data: std::ptr::null_mut(),
            len: 0,
        };
        unsafe {
            let e = anofox_ml_fit_labels(
                kind.as_ptr(),
                std::ptr::null(),
                x.as_ptr(),
                30,
                1,
                yp.as_ptr(),
                std::ptr::null(),
                1,
                &mut blob,
            );
            assert!(e.is_null(), "{}", CStr::from_ptr(e).to_str().unwrap());
            let mut out = AnofoxMlStrings {
                data: std::ptr::null_mut(),
                len: 0,
            };
            let e = anofox_ml_predict_labels(blob.data, blob.len, x.as_ptr(), 30, 1, &mut out);
            assert!(e.is_null());
            let got: Vec<&str> = std::slice::from_raw_parts(out.data, out.len)
                .iter()
                .map(|&p| CStr::from_ptr(p).to_str().unwrap())
                .collect();
            let want: Vec<&str> = y.iter().map(|s| s.to_str().unwrap()).collect();
            assert_eq!(got, want);
            anofox_ml_free_strings(out);
            let mut info = std::mem::zeroed::<AnofoxMlModelInfo>();
            assert!(anofox_ml_model_info(blob.data, blob.len, &mut info).is_null());
            let labels: Vec<&str> = std::slice::from_raw_parts(info.labels.data, info.labels.len)
                .iter()
                .map(|&p| CStr::from_ptr(p).to_str().unwrap())
                .collect();
            assert_eq!(labels, ["setosa", "versicolor", "virginica"]);
            anofox_ml_free_model_info(info);
            anofox_ml_free_bytes(blob);
        }
    }

    #[test]
    fn numeric_classes_print_as_numbers() {
        let meta = Meta {
            kind: "X".into(),
            n_features: 1,
            classes: None,
            labels: None,
            conformity: None,
            encoder: None,
        };
        assert_eq!(label_of(&meta, 1.0).unwrap(), "1");
        assert_eq!(label_of(&meta, -2.5).unwrap(), "-2.5");
    }
}
