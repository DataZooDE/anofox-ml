//! Stateless text features: scikit-learn's `HashingVectorizer` (signed
//! murmur3 feature hashing of unigram tokens, optional L2 norm).

use std::ffi::c_char;

use anofox_ml_text::HashingVectorizer;
use serde::Deserialize;

use crate::{err, str_arg, AnofoxMlDoubles};

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Options {
    /// Flip the sign of half the buckets (default true), so collisions
    /// cancel out on average.
    alternate_sign: Option<bool>,
    /// `"l2"` (default) or `null` for raw counts (read from the raw JSON,
    /// since serde maps a JSON null to an absent field).
    #[allow(dead_code)]
    norm: Option<serde_json::Value>,
}

/// `(bucket, value)` pairs of one document, buckets ascending.
pub(crate) fn hash_document(
    text: &str,
    n_features: usize,
    options: &str,
) -> Result<Vec<(usize, f64)>, String> {
    if n_features == 0 {
        return Err("n_features must be positive".into());
    }
    let o: Options = if options.trim().is_empty() {
        Options::default()
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))?
    };
    let mut v = HashingVectorizer::new(n_features);
    v.alternate_sign = o.alternate_sign.unwrap_or(true);
    // A JSON null is deserialised as None, so tell "absent" from "null" by
    // the raw options.
    let raw: serde_json::Value = serde_json::from_str(if options.trim().is_empty() {
        "{}"
    } else {
        options
    })
    .map_err(|e| format!("invalid options JSON: {e}"))?;
    v.norm_l2 = match raw.get("norm") {
        None => true,
        Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::String(n)) if n == "l2" => true,
        Some(n) => return Err(format!("norm must be \"l2\" or null, got {n}")),
    };
    let csr = v.transform_sparse(&[text]);
    let mut row: Vec<(usize, f64)> = csr.row_iter(0).filter(|(_, x)| *x != 0.0).collect();
    row.sort_unstable_by_key(|e| e.0);
    Ok(row)
}

/// Hash one document into `n_features` buckets. With `dense`, `out`
/// receives `n_features` values; otherwise `2 * nnz` values: the buckets
/// (0-based), then their values.
///
/// # Safety
/// `text` / `options` are NUL-terminated (`options` may be NULL); `out` is
/// writable.
#[no_mangle]
pub unsafe extern "C" fn anofox_ml_hashing_vectorize(
    text: *const c_char,
    n_features: usize,
    options: *const c_char,
    dense: bool,
    out: *mut AnofoxMlDoubles,
) -> *mut c_char {
    let run = || -> Result<Vec<f64>, String> {
        let text = str_arg(text, "text")?;
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        let row = hash_document(text, n_features, options)?;
        Ok(if dense {
            let mut v = vec![0.0; n_features];
            for (j, x) in row {
                v[j] = x;
            }
            v
        } else {
            row.iter()
                .map(|e| e.0 as f64)
                .chain(row.iter().map(|e| e.1))
                .collect()
        })
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(v)) => {
            *out = crate::into_doubles(v);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while hashing"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // scikit-learn: HashingVectorizer(n_features=16).transform([DOC]) and
    // HashingVectorizer(n_features=16, alternate_sign=False, norm=None).
    const DOC: &str = "The quick brown fox jumps over the lazy dog the end";

    #[test]
    fn matches_scikit_learn() {
        let r = hash_document(DOC, 16, "").unwrap();
        let want = [
            (0, 0.242535625036),
            (7, -0.485071250073),
            (9, 0.242535625036),
            (11, 0.242535625036),
            (13, 0.242535625036),
            (14, -0.727606875109),
        ];
        assert_eq!(r.len(), want.len());
        for ((j, v), (wj, wv)) in r.iter().zip(want) {
            assert_eq!(*j, wj);
            assert!((v - wv).abs() < 1e-9, "{r:?}");
        }
        let r = hash_document(DOC, 16, r#"{"alternate_sign": false, "norm": null}"#).unwrap();
        assert_eq!(
            r,
            vec![
                (0, 1.0),
                (5, 2.0),
                (7, 2.0),
                (9, 1.0),
                (11, 1.0),
                (13, 1.0),
                (14, 3.0)
            ]
        );
        assert!(hash_document(DOC, 16, r#"{"norm": "l1"}"#)
            .unwrap_err()
            .contains("norm"));
    }
}
