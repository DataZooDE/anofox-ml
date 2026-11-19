//! Text and embedding columns besides the numeric features (CatBoost's
//! `text_features` and `embedding_features`): `fit_with_text`,
//! `fit_with_embeddings`, `fit_with_text_and_embeddings` and their
//! `predict*` counterparts. The BLOB records the number of text columns and
//! the width of every embedding column ([`Side`]), so predictions check them
//! and plain `ml_predict(model, x)` points to the right overload.
//!
//! Text arrives row-major (`n_rows × n_text` strings); embeddings row-major
//! too, every row holding its embedding columns one after the other
//! (`n_rows × sum(dims)` doubles).

use std::ffi::c_char;

use anofox_ml_registry::DynModel;
use ndarray::{Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::{
    decode, encode, encode_fitted, encode_labels, err, flatten, into_doubles, into_strings,
    label_of, make_model, matrix, on_worker, require_method, run_fit, str_arg, strings, with_model,
    AnofoxMlBytes, AnofoxMlDoubles, AnofoxMlStrings, Meta,
};

/// Side inputs of a model fitted with text and / or embedding columns.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub(crate) struct Side {
    pub text: usize,
    pub embeddings: Vec<usize>,
}

impl Side {
    /// The registry method family: `with_text`, `with_embeddings` or
    /// `with_text_and_embeddings`.
    fn suffix(&self) -> &'static str {
        match (self.text > 0, self.embeddings.is_empty()) {
            (true, true) => "with_text",
            (false, false) => "with_embeddings",
            _ => "with_text_and_embeddings",
        }
    }
}

/// Text columns (`text[c][r]`) and embedding matrices of `n_rows` rows.
struct Inputs {
    text: Vec<Vec<String>>,
    embeddings: Vec<Array2<f64>>,
}

impl Inputs {
    unsafe fn read(
        n_rows: usize,
        text: *const *const c_char,
        n_text: usize,
        emb: *const f64,
        dims: *const usize,
        n_emb: usize,
    ) -> Result<(Self, Side), String> {
        let cells = if n_text > 0 {
            strings(text, n_rows * n_text, "text")?
        } else {
            Vec::new()
        };
        let text = (0..n_text)
            .map(|c| {
                (0..n_rows)
                    .map(|r| cells[r * n_text + c].to_string())
                    .collect()
            })
            .collect();
        let dims = if n_emb > 0 {
            if dims.is_null() {
                return Err("embedding widths are NULL".into());
            }
            std::slice::from_raw_parts(dims, n_emb).to_vec()
        } else {
            Vec::new()
        };
        if dims.contains(&0) {
            return Err("an embedding column has no values".into());
        }
        let mut embeddings = Vec::with_capacity(n_emb);
        if n_emb > 0 {
            let flat = matrix(emb, n_rows, dims.iter().sum())?;
            let mut start = 0;
            for &d in &dims {
                embeddings.push(flat.slice(ndarray::s![.., start..start + d]).to_owned());
                start += d;
            }
        }
        let side = Side {
            text: n_text,
            embeddings: dims,
        };
        Ok((Self { text, embeddings }, side))
    }

    fn views(&self) -> Vec<ArrayView2<'_, f64>> {
        self.embeddings.iter().map(|e| e.view()).collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_impl<'a>(
    kind: &str,
    params: &str,
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, f64>,
    w: Option<ArrayView1<'a, f64>>,
    inputs: &'a Inputs,
    side: Side,
    labels: Option<Vec<String>>,
    threads: usize,
) -> Result<Vec<u8>, String> {
    if side.text == 0 && side.embeddings.is_empty() {
        return Err("no text or embedding columns: use ml_fit_agg".into());
    }
    let method = format!("fit_{}", side.suffix());
    let mut model = make_model(kind, params)?;
    require_method(kind, model.as_ref(), &method)?;
    let emb = inputs.views();
    run_fit(threads, || match side.suffix() {
        "with_text" => model.fit_with_text(x, &inputs.text, Some(y), w),
        "with_embeddings" => model.fit_with_embeddings(x, &emb, Some(y), w),
        _ => model.fit_with_text_and_embeddings(x, &inputs.text, &emb, Some(y), w),
    })?
    .map_err(|e| e.to_string())?;
    let blob = encode_fitted(kind, model.as_ref(), x.ncols(), Some(y), labels)?;
    let (mut meta, bytes) = decode(&blob)?;
    meta.side = Some(side);
    Ok(encode(&meta, bytes))
}

/// [`crate::anofox_ml_fit`] with text and embedding columns besides `x`.
/// `y` holds `n_rows` doubles, or (`y` NULL) `labels` holds `n_rows` string
/// targets; `text` holds `n_rows * n_text` strings, row-major; `emb` holds
/// `n_rows * sum(dims)` doubles, row-major, a row's embedding columns one
/// after the other; `dims` holds the `n_emb` widths.
///
/// # Safety
/// As [`crate::anofox_ml_fit`], with the buffers above.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_fit_features(
    kind: *const c_char,
    params: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    y: *const f64,
    labels: *const *const c_char,
    w: *const f64,
    text: *const *const c_char,
    n_text: usize,
    emb: *const f64,
    dims: *const usize,
    n_emb: usize,
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
        let (inputs, side) = Inputs::read(n_rows, text, n_text, emb, dims, n_emb)?;
        let w = (!w.is_null()).then(|| ArrayView1::from(std::slice::from_raw_parts(w, n_rows)));
        let (y, names) = if !y.is_null() {
            (std::slice::from_raw_parts(y, n_rows).to_vec(), None)
        } else if !labels.is_null() {
            let (names, codes) = encode_labels(&strings(labels, n_rows, "y")?);
            (codes, Some(names))
        } else {
            return Err("targets are required".into());
        };
        let y = ArrayView1::from(&y);
        fit_impl(kind, params, x, y, w, &inputs, side, names, threads)
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
        Err(p) => err(crate::panic_message("while fitting", p)),
    }
}

/// The error of a model without text / embedding columns given them, or of
/// one with them given other ones.
fn check_side(meta: &Meta, side: &Side) -> Result<(), String> {
    match &meta.side {
        None => Err(format!(
            "{} was fitted without text or embedding columns: use ml_predict(model, x)",
            meta.kind
        )),
        Some(s) if s != side => Err(format!(
            "{} was fitted with {} text columns and embeddings of widths {:?}, got {} and {:?}",
            meta.kind, s.text, s.embeddings, side.text, side.embeddings
        )),
        Some(_) => Ok(()),
    }
}

/// `predict` or `predict_proba` of a model fitted with side inputs; the
/// row-major output and its column count.
fn apply(
    blob: &[u8],
    method: &str,
    x: ArrayView2<f64>,
    inputs: &Inputs,
    side: &Side,
) -> Result<(Vec<f64>, usize), String> {
    with_model(blob, |meta, model: &dyn DynModel| {
        check_side(meta, side)?;
        crate::check_width(meta, x)?;
        let name = format!("{method}_{}", side.suffix());
        require_method(&meta.kind, model, &name)?;
        let emb = inputs.views();
        let out = on_worker(|| match (method, side.suffix()) {
            ("predict", "with_text") => model.predict_with_text(x, &inputs.text),
            ("predict", "with_embeddings") => model.predict_with_embeddings(x, &emb),
            ("predict", _) => model.predict_with_text_and_embeddings(x, &inputs.text, &emb),
            (_, "with_text") => model.predict_proba_with_text(x, &inputs.text),
            (_, "with_embeddings") => model.predict_proba_with_embeddings(x, &emb),
            _ => model.predict_proba_with_text_and_embeddings(x, &inputs.text, &emb),
        });
        Ok(flatten(out.map_err(|e| e.to_string())?))
    })
}

/// Apply `method` (`predict` or `predict_proba`) of a model fitted with
/// [`anofox_ml_fit_features`] to `n_rows` rows of `x` with their text and
/// embedding columns (laid out as for the fit). `out` receives
/// `n_rows * out_cols` doubles, row-major.
///
/// # Safety
/// As [`crate::anofox_ml_apply`], with the buffers of
/// [`anofox_ml_fit_features`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_apply_features(
    model: *const u8,
    model_len: usize,
    method: *const c_char,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    text: *const *const c_char,
    n_text: usize,
    emb: *const f64,
    dims: *const usize,
    n_emb: usize,
    out: *mut AnofoxMlDoubles,
    out_cols: *mut usize,
) -> *mut c_char {
    let run = || -> Result<(Vec<f64>, usize), String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let method = str_arg(method, "method")?;
        if method != "predict" && method != "predict_proba" {
            return Err(format!(
                "unknown method '{method}' (known: predict, predict_proba)"
            ));
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        let x = matrix(x, n_rows, n_cols)?;
        let (inputs, side) = Inputs::read(n_rows, text, n_text, emb, dims, n_emb)?;
        apply(blob, method, x, &inputs, &side)
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok((v, cols))) => {
            *out = into_doubles(v);
            *out_cols = cols;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(p) => err(crate::panic_message("while predicting", p)),
    }
}

/// [`anofox_ml_apply_features`] with `predict`, returning class labels as
/// strings.
///
/// # Safety
/// As [`anofox_ml_apply_features`]; release `out` with
/// [`crate::anofox_ml_free_strings`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_predict_labels_features(
    model: *const u8,
    model_len: usize,
    x: *const f64,
    n_rows: usize,
    n_cols: usize,
    text: *const *const c_char,
    n_text: usize,
    emb: *const f64,
    dims: *const usize,
    n_emb: usize,
    out: *mut AnofoxMlStrings,
) -> *mut c_char {
    let run = || -> Result<Vec<String>, String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let blob = std::slice::from_raw_parts(model, model_len);
        let x = matrix(x, n_rows, n_cols)?;
        let (inputs, side) = Inputs::read(n_rows, text, n_text, emb, dims, n_emb)?;
        let (v, cols) = apply(blob, "predict", x, &inputs, &side)?;
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
        Err(p) => err(crate::panic_message("while predicting", p)),
    }
}
