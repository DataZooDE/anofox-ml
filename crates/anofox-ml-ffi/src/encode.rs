//! Categorical encoders for `VARCHAR` features, as scikit-learn's
//! `OneHotEncoder` and `OrdinalEncoder`: the categories of every column are
//! learned (sorted) and stored in the BLOB metadata, which needs no registry
//! model.

use std::ffi::c_char;

use serde::{Deserialize, Serialize};

use crate::{
    decode, encode, err, into_doubles, str_arg, strings, AnofoxMlBytes, AnofoxMlDoubles, Meta,
};

pub(crate) const KINDS: &[&str] = &["OneHotEncoder", "OrdinalEncoder"];

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Params {
    /// `"error"` (default) or `"ignore"`: an unseen category encodes as all
    /// zeros (one-hot) or `unknown_value` (ordinal).
    handle_unknown: Option<String>,
    /// Ordinal code of an unseen category with `handle_unknown = "ignore"`
    /// (default -1).
    unknown_value: Option<f64>,
    /// One-hot: `"first"` drops each column's first category, `"if_binary"`
    /// only for two-category columns (default: keep all).
    drop: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct EncoderFit {
    pub one_hot: bool,
    pub categories: Vec<Vec<String>>,
    pub ignore_unknown: bool,
    pub unknown_value: f64,
    /// One-hot: per column, whether its first category is dropped.
    pub dropped: Vec<bool>,
}

impl EncoderFit {
    /// Output width of one row.
    pub fn n_outputs(&self) -> usize {
        if self.one_hot {
            self.categories
                .iter()
                .zip(&self.dropped)
                .map(|(c, &d)| c.len() - usize::from(d))
                .sum()
        } else {
            self.categories.len()
        }
    }

    pub fn params_json(&self) -> serde_json::Value {
        let mut p = serde_json::json!({
            "handle_unknown": if self.ignore_unknown { "ignore" } else { "error" },
        });
        if self.one_hot {
            p["drop"] = serde_json::json!(self.dropped);
        } else {
            p["unknown_value"] = serde_json::json!(self.unknown_value);
        }
        p
    }
}

/// Learn the categories of `n_cols` columns from row-major `x`.
pub(crate) fn fit(
    kind: &str,
    params: &str,
    x: &[&str],
    n_cols: usize,
) -> Result<EncoderFit, String> {
    let p: Params = if params.trim().is_empty() {
        Params::default()
    } else {
        serde_json::from_str(params).map_err(|e| format!("invalid params JSON: {e}"))?
    };
    let one_hot = match kind {
        "OneHotEncoder" => true,
        "OrdinalEncoder" => false,
        _ => {
            return Err(format!(
                "{kind} does not take VARCHAR features (encoders: {KINDS:?})"
            ))
        }
    };
    let ignore_unknown = match p.handle_unknown.as_deref() {
        None | Some("error") => false,
        Some("ignore") => true,
        Some(h) => {
            return Err(format!(
                "handle_unknown must be \"error\" or \"ignore\", got \"{h}\""
            ))
        }
    };
    if !one_hot && p.drop.is_some() {
        return Err("OrdinalEncoder has no parameter 'drop'".into());
    }
    if one_hot && p.unknown_value.is_some() {
        return Err("OneHotEncoder has no parameter 'unknown_value'".into());
    }
    let mut categories: Vec<Vec<String>> = (0..n_cols)
        .map(|j| {
            let mut c: Vec<String> = x
                .iter()
                .skip(j)
                .step_by(n_cols)
                .map(|s| s.to_string())
                .collect();
            c.sort_unstable();
            c.dedup();
            c
        })
        .collect();
    categories.shrink_to_fit();
    let dropped = match p.drop.as_deref() {
        None => vec![false; n_cols],
        Some("first") => vec![true; n_cols],
        Some("if_binary") => categories.iter().map(|c| c.len() == 2).collect(),
        Some(d) => {
            return Err(format!(
                "drop must be \"first\", \"if_binary\" or null, got \"{d}\""
            ))
        }
    };
    Ok(EncoderFit {
        one_hot,
        categories,
        ignore_unknown,
        unknown_value: p.unknown_value.unwrap_or(-1.0),
        dropped,
    })
}

/// Encode row-major `x` (`n_cols` columns); returns `n_rows × n_outputs`.
pub(crate) fn transform(e: &EncoderFit, x: &[&str], n_cols: usize) -> Result<Vec<f64>, String> {
    if n_cols != e.categories.len() {
        return Err(format!(
            "the encoder was fitted on {} columns, got {n_cols}",
            e.categories.len()
        ));
    }
    let mut out = Vec::with_capacity(x.len() / n_cols.max(1) * e.n_outputs());
    for row in x.chunks(n_cols) {
        for (j, v) in row.iter().enumerate() {
            let cats = &e.categories[j];
            let code = cats.binary_search_by(|c| c.as_str().cmp(v)).ok();
            if code.is_none() && !e.ignore_unknown {
                return Err(format!(
                    "unknown category '{v}' in column {} (set handle_unknown to \"ignore\")",
                    j + 1
                ));
            }
            if e.one_hot {
                let skip = usize::from(e.dropped[j]);
                for k in skip..cats.len() {
                    out.push(if code == Some(k) { 1.0 } else { 0.0 });
                }
            } else {
                out.push(code.map_or(e.unknown_value, |k| k as f64));
            }
        }
    }
    Ok(out)
}

/// The encoder stored in a BLOB.
pub(crate) fn encoder_of(blob: &[u8]) -> Result<EncoderFit, String> {
    let (meta, _) = decode(blob)?;
    meta.encoder
        .ok_or_else(|| format!("{} does not encode VARCHAR features", meta.kind))
}

/// Fit an encoder (`OneHotEncoder` / `OrdinalEncoder`) on `n_rows × n_cols`
/// strings, row-major; `out` receives the BLOB.
///
/// # Safety
/// `kind` / `params` are NUL-terminated (`params` may be NULL); `x` holds
/// `n_rows * n_cols` NUL-terminated strings; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_fit_encoder(
    kind: *const c_char,
    params: *const c_char,
    x: *const *const c_char,
    n_rows: usize,
    n_cols: usize,
    out: *mut AnofoxMlBytes,
) -> *mut c_char {
    let run = || -> Result<Vec<u8>, String> {
        let kind = str_arg(kind, "kind")?;
        let params = if params.is_null() {
            ""
        } else {
            str_arg(params, "params")?
        };
        if n_rows == 0 || n_cols == 0 {
            return Err("empty feature matrix".into());
        }
        let x = strings(x, n_rows * n_cols, "x")?;
        let e = fit(kind, params, &x, n_cols)?;
        let meta = Meta {
            kind: kind.to_string(),
            n_features: n_cols,
            classes: None,
            labels: None,
            conformity: None,
            encoder: Some(e),
            y_columns: None,
        };
        Ok(encode(&meta, &[]))
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

/// Encode `n_rows × n_cols` strings (row-major) with an encoder BLOB; `out`
/// receives `n_rows * out_cols` doubles.
///
/// # Safety
/// `model` holds `model_len` bytes, `x` `n_rows * n_cols` NUL-terminated
/// strings; `out` / `out_cols` are writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_encode(
    model: *const u8,
    model_len: usize,
    x: *const *const c_char,
    n_rows: usize,
    n_cols: usize,
    out: *mut AnofoxMlDoubles,
    out_cols: *mut usize,
) -> *mut c_char {
    let run = || -> Result<(Vec<f64>, usize), String> {
        if model.is_null() {
            return Err("model is NULL".into());
        }
        let e = encoder_of(std::slice::from_raw_parts(model, model_len))?;
        let x = strings(x, n_rows * n_cols, "x")?;
        Ok((transform(&e, &x, n_cols)?, e.n_outputs()))
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok((v, cols))) => {
            *out = into_doubles(v);
            *out_cols = cols;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(p) => err(crate::panic_message("while encoding", p)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_hot_and_ordinal_like_scikit_learn() {
        // rows: (city, channel)
        let x = ["Berlin", "web", "Paris", "shop", "Berlin", "shop"];
        let oh = fit("OneHotEncoder", "", &x, 2).unwrap();
        assert_eq!(
            oh.categories,
            vec![vec!["Berlin", "Paris"], vec!["shop", "web"]]
        );
        assert_eq!(oh.n_outputs(), 4);
        assert_eq!(
            transform(&oh, &["Paris", "web"], 2).unwrap(),
            [0.0, 1.0, 0.0, 1.0]
        );
        assert!(transform(&oh, &["Rome", "web"], 2)
            .unwrap_err()
            .contains("unknown category 'Rome'"));
        let oh = fit(
            "OneHotEncoder",
            r#"{"handle_unknown": "ignore", "drop": "first"}"#,
            &x,
            2,
        )
        .unwrap();
        assert_eq!(transform(&oh, &["Rome", "web"], 2).unwrap(), [0.0, 1.0]);
        let ord = fit("OrdinalEncoder", r#"{"handle_unknown": "ignore"}"#, &x, 2).unwrap();
        assert_eq!(
            transform(&ord, &["Paris", "shop", "Rome", "web"], 2).unwrap(),
            [1.0, 0.0, -1.0, 1.0]
        );
        assert!(fit("Pca", "", &x, 2).unwrap_err().contains("VARCHAR"));
    }
}
