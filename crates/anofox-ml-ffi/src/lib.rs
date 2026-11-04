//! C ABI over `anofox-ml-registry` for the `anofox_ml` DuckDB extension.
//!
//! Every function returns `NULL` on success or an error message (free it
//! with [`anofox_ml_free_string`]). Output buffers are allocated here and
//! released with the matching `anofox_ml_free_*` function.
//!
//! A model BLOB is `AMLX`, the kind as a `u16` little-endian length plus
//! UTF-8 bytes, then the registry's model bytes (anofox-ml's binary
//! persistence format). Feature matrices are row-major `n_rows × n_cols`
//! doubles.

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::ffi::{c_char, CStr, CString};
use std::hash::{Hash, Hasher};

use anofox_ml_registry::{DynModel, Out};
use ndarray::{ArrayView1, ArrayView2};

const MAGIC: &[u8; 4] = b"AMLX";

/// Cached decoded model: (BLOB hash, BLOB length, model).
type CachedModel = Option<(u64, usize, Box<dyn DynModel>)>;

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

thread_local! {
    /// One rayon thread per DuckDB worker: DuckDB already parallelises
    /// across groups, so estimators with their own parallelism (forests)
    /// must not spawn more threads.
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

fn encode(kind: &str, model: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(6 + kind.len() + model.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(kind.len() as u16).to_le_bytes());
    out.extend_from_slice(kind.as_bytes());
    out.extend_from_slice(model);
    out
}

fn decode(blob: &[u8]) -> Result<(&str, &[u8]), String> {
    let rest = blob
        .strip_prefix(MAGIC.as_slice())
        .ok_or("not an anofox_ml model BLOB")?;
    if rest.len() < 2 {
        return Err("truncated model BLOB".into());
    }
    let n = u16::from_le_bytes([rest[0], rest[1]]) as usize;
    let kind = rest
        .get(2..2 + n)
        .ok_or("truncated model BLOB")
        .and_then(|k| std::str::from_utf8(k).map_err(|_| "invalid model kind"))?;
    Ok((kind, &rest[2 + n..]))
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

fn fit_impl(
    kind: &str,
    params: &str,
    x: ArrayView2<f64>,
    y: Option<ArrayView1<f64>>,
    w: Option<ArrayView1<f64>>,
) -> Result<Vec<u8>, String> {
    let params: serde_json::Value = if params.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(params).map_err(|e| format!("invalid params JSON: {e}"))?
    };
    let mut model = anofox_ml_registry::make(kind, &params).map_err(|e| e.to_string())?;
    POOL.with(|p| p.install(|| model.fit(x, y, w)))
        .map_err(|e| e.to_string())?;
    let bytes = model.to_bytes().map_err(|e| e.to_string())?;
    Ok(encode(kind, &bytes))
}

/// Fit `kind` with JSON `params` on `x` (and `y`, `w` when non-NULL) and
/// return the model BLOB in `out`.
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
        fit_impl(kind, params, x, y, w)
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

fn predict_impl(blob: &[u8], x: ArrayView2<f64>) -> Result<Vec<f64>, String> {
    let mut h = DefaultHasher::new();
    blob.hash(&mut h);
    let key = h.finish();
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let hit = matches!(&*cache, Some((k, n, _)) if *k == key && *n == blob.len());
        if !hit {
            let (kind, bytes) = decode(blob)?;
            let model = anofox_ml_registry::load(kind, bytes).map_err(|e| e.to_string())?;
            *cache = Some((key, blob.len(), model));
        }
        let model = &cache.as_ref().expect("filled above").2;
        match POOL
            .with(|p| p.install(|| model.predict(x)))
            .map_err(|e| e.to_string())?
        {
            Out::A1(a) => Ok(a.to_vec()),
            Out::A2(a) if a.ncols() == 1 => Ok(a.column(0).to_vec()),
            Out::A2(a) => Err(format!("predict returned {} columns", a.ncols())),
        }
    })
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
        predict_impl(blob, matrix(x, n_rows, n_cols)?)
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
        Ok((kind, _)) => {
            *out = err(kind);
            std::ptr::null_mut()
        }
        Err(e) => err(e),
    }
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
}
