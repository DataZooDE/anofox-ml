//! Model interchange with anofox-ml in Rust: `ml_model_export` hands out a
//! model's registry bytes (`DynModel::to_bytes`, loadable with
//! `anofox_ml_registry::load(kind, bytes)`), `ml_model_import` wraps such
//! bytes into a model BLOB. The BLOB header ([`Meta`]) is not part of the
//! registry bytes: the import takes the feature count, and the classes of a
//! classifier (or its string labels) unless the fitted state lists them.

use std::ffi::c_char;

use crate::{decode, encode, err, str_arg, strings, y_columns, AnofoxMlBytes, Meta};

/// The registry bytes of a model BLOB.
fn export(blob: &[u8]) -> Result<Vec<u8>, String> {
    let (meta, bytes) = decode(blob)?;
    if meta.encoder.is_some() {
        return Err(format!(
            "{} is a VARCHAR encoder of this extension, not an anofox-ml model",
            meta.kind
        ));
    }
    if meta.conformity.is_some() {
        return Err(format!(
            "the {} model is conformalized; its calibration is not part of the anofox-ml \
             model: export the model before ml_conformalize_agg",
            meta.kind
        ));
    }
    Ok(bytes.to_vec())
}

/// Wrap registry `bytes` of `kind` into a model BLOB.
fn import(
    kind: &str,
    bytes: &[u8],
    n_features: usize,
    classes: Option<Vec<f64>>,
    labels: Option<Vec<String>>,
) -> Result<Vec<u8>, String> {
    if n_features == 0 {
        return Err("n_features must be positive".into());
    }
    let model = anofox_ml_registry::load(kind, bytes).map_err(|e| e.to_string())?;
    let proba = model.capabilities().contains(&"predict_proba");
    // The fitted state of most classifiers lists their classes.
    let fitted_classes = || -> Option<Vec<f64>> {
        let json: serde_json::Value = serde_json::from_str(&model.fitted_json().ok()?).ok()?;
        json.get("classes")?
            .as_array()?
            .iter()
            .map(|v| v.as_f64())
            .collect()
    };
    let classes = match (classes, &labels) {
        (Some(c), _) => Some(c),
        (None, Some(l)) => Some((0..l.len()).map(|i| i as f64).collect()),
        (None, None) if proba => Some(fitted_classes().ok_or_else(|| {
            format!(
                "{kind} has predict_proba but its fitted state does not list the classes: \
                 pass them, ml_model_import(kind, bytes, n_features, [class, ...])"
            )
        })?),
        (None, None) => None,
    };
    let meta = Meta {
        kind: kind.to_string(),
        n_features,
        classes: classes.filter(|_| proba),
        labels,
        conformity: None,
        encoder: None,
        y_columns: y_columns(model.as_ref()),
        side: None,
    };
    Ok(encode(&meta, bytes))
}

unsafe fn bytes_out(r: Result<Vec<u8>, String>, out: *mut AnofoxMlBytes) -> *mut c_char {
    match r {
        Ok(bytes) => {
            let mut b = bytes.into_boxed_slice();
            *out = AnofoxMlBytes {
                data: b.as_mut_ptr(),
                len: b.len(),
            };
            std::mem::forget(b);
            std::ptr::null_mut()
        }
        Err(e) => err(e),
    }
}

/// The registry bytes of a model BLOB (`DynModel::to_bytes`).
///
/// # Safety
/// `model` holds `model_len` bytes; `out` is writable; free it with
/// [`crate::anofox_ml_free_bytes`].
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_model_export(
    model: *const u8,
    model_len: usize,
    out: *mut AnofoxMlBytes,
) -> *mut c_char {
    if model.is_null() {
        return err("model is NULL");
    }
    let blob = std::slice::from_raw_parts(model, model_len);
    match std::panic::catch_unwind(|| export(blob)) {
        Ok(r) => bytes_out(r, out),
        Err(p) => err(crate::panic_message("while exporting", p)),
    }
}

/// A model BLOB from registry `bytes` of `kind` with `n_features` features.
/// `classes` (`n_classes` doubles) or `labels` (`n_classes` strings) give
/// the classes of a classifier; both may be NULL.
///
/// # Safety
/// `kind` is NUL-terminated; `bytes` holds `len` bytes; `classes` /
/// `labels` hold `n_classes` entries or are NULL; `out` is writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_model_import(
    kind: *const c_char,
    bytes: *const u8,
    len: usize,
    n_features: usize,
    classes: *const f64,
    labels: *const *const c_char,
    n_classes: usize,
    out: *mut AnofoxMlBytes,
) -> *mut c_char {
    let run = || -> Result<Vec<u8>, String> {
        let kind = str_arg(kind, "kind")?;
        if bytes.is_null() {
            return Err("bytes are NULL".into());
        }
        let bytes = std::slice::from_raw_parts(bytes, len);
        let classes =
            (!classes.is_null()).then(|| std::slice::from_raw_parts(classes, n_classes).to_vec());
        let labels = if labels.is_null() {
            None
        } else {
            let l: Vec<String> = strings(labels, n_classes, "labels")?
                .into_iter()
                .map(String::from)
                .collect();
            let mut sorted = l.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted != l {
                return Err(
                    "labels must be distinct and sorted: label i is the model's class i".into(),
                );
            }
            Some(l)
        };
        import(kind, bytes, n_features, classes, labels)
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(r) => bytes_out(r, out),
        Err(p) => err(crate::panic_message("while importing", p)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{apply_impl, fit_impl};
    use ndarray::{array, ArrayView1};

    #[test]
    fn export_import_round_trip() {
        let x = array![
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, 1.0],
            [5.0, 0.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let blob = fit_impl(
            "LogisticRegressor",
            "",
            x.view(),
            Some(ArrayView1::from(&y)),
            None,
            None,
            1,
        )
        .unwrap();
        let bytes = export(&blob).unwrap();
        // The registry loads the exported bytes on its own.
        assert!(anofox_ml_registry::load("LogisticRegressor", &bytes).is_ok());
        // Classes come from the fitted state.
        let back = import("LogisticRegressor", &bytes, 2, None, None).unwrap();
        assert_eq!(
            apply_impl(&back, "predict_proba", x.view()).unwrap(),
            apply_impl(&blob, "predict_proba", x.view()).unwrap()
        );
        assert_eq!(decode(&back).unwrap().0.classes, Some(vec![0.0, 1.0]));
        let e = import("OlsRegressor", &bytes, 2, None, None).unwrap_err();
        assert!(!e.is_empty());
        let e = import("LogisticRegressor", &bytes, 0, None, None).unwrap_err();
        assert!(e.contains("n_features"), "{e}");
    }
}
