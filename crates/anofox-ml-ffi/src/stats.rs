//! Table-level statistics behind the `ml_*_agg` helper aggregates:
//! mutual information (scikit-learn `mutual_info_regression` /
//! `mutual_info_classif`), covariance (empirical, shrunk, Ledoit-Wolf, OAS, MinCovDet) and the
//! MeanShift bandwidth estimate.

use std::ffi::c_char;

use anofox_ml_preprocessing::mutual_information::{
    mutual_info_classif, mutual_info_regression, DiscreteFeatures,
};
use ndarray::{Array1, ArrayView2};
use serde::Deserialize;

use crate::{encode_labels, err, into_doubles, matrix, str_arg, strings, AnofoxMlDoubles};

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct MiOptions {
    /// The target is discrete (classes); default false, true for string labels.
    discrete_target: Option<bool>,
    /// `"auto"` (all continuous, default), `true` (all discrete), or a list
    /// of 1-based feature positions that are discrete.
    discrete_features: Option<serde_json::Value>,
    n_neighbors: Option<usize>,
    seed: Option<u64>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct CovOptions {
    /// `"ledoit_wolf"` (default), `"oas"`, `"empirical"`, `"shrunk"` or
    /// `"mcd"` (robust minimum covariance determinant).
    method: Option<String>,
    /// `shrunk`: the fixed shrinkage towards `μ I` (default 0.1).
    shrinkage: Option<f64>,
    /// `mcd`: fraction of rows in the raw support (default `(n + p + 1) / 2n`).
    support_fraction: Option<f64>,
    /// `mcd`: random seed of FastMCD (default 0).
    seed: Option<u64>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct BandwidthOptions {
    quantile: Option<f64>,
    n_samples: Option<usize>,
    seed: Option<u64>,
}

fn parse<T: Default + for<'de> Deserialize<'de>>(options: &str) -> Result<T, String> {
    if options.trim().is_empty() {
        Ok(T::default())
    } else {
        serde_json::from_str(options).map_err(|e| format!("invalid options JSON: {e}"))
    }
}

fn mutual_info(
    x: ArrayView2<f64>,
    y: Array1<f64>,
    labels: bool,
    options: &str,
) -> Result<Vec<f64>, String> {
    let o: MiOptions = parse(options)?;
    let discrete = match &o.discrete_features {
        None => DiscreteFeatures::Auto,
        Some(serde_json::Value::String(s)) if s == "auto" => DiscreteFeatures::Auto,
        Some(serde_json::Value::Bool(true)) => DiscreteFeatures::All,
        Some(serde_json::Value::Bool(false)) => DiscreteFeatures::Auto,
        Some(serde_json::Value::Array(a)) => DiscreteFeatures::Indices(
            a.iter()
                .map(|v| match v.as_u64() {
                    Some(i) if i >= 1 && (i as usize) <= x.ncols() => Ok(i as usize - 1),
                    _ => Err(format!(
                        "discrete_features: {v} is not a feature position 1..{}",
                        x.ncols()
                    )),
                })
                .collect::<Result<_, _>>()?,
        ),
        Some(v) => {
            return Err(format!(
                "discrete_features must be \"auto\", true or a list of positions, got {v}"
            ))
        }
    };
    let k = o.n_neighbors.unwrap_or(3);
    let seed = o.seed.unwrap_or(0);
    let r = if o.discrete_target.unwrap_or(labels) {
        mutual_info_classif(x, &y, &discrete, k, seed)
    } else {
        mutual_info_regression(x, &y, &discrete, k, seed)
    };
    r.map(|v| v.to_vec()).map_err(|e| e.to_string())
}

/// `d × d` covariance (row-major), the shrinkage coefficient (NaN for
/// `mcd`; 0 for `empirical`) and the `d` location values, as scikit-learn's
/// estimators of the same names.
fn covariance(x: ArrayView2<f64>, options: &str) -> Result<Vec<f64>, String> {
    use anofox_ml_core::FitUnsupervised;
    use anofox_ml_preprocessing::covariance as cv;
    let o: CovOptions = parse(options)?;
    let method = o.method.as_deref().unwrap_or("ledoit_wolf");
    if method != "shrunk" && o.shrinkage.is_some() {
        return Err("shrinkage applies to method \"shrunk\"".into());
    }
    if method != "mcd" && (o.support_fraction.is_some() || o.seed.is_some()) {
        return Err("support_fraction and seed apply to method \"mcd\"".into());
    }
    let fitted = match method {
        "ledoit_wolf" => cv::LedoitWolf::default().fit_view(x),
        "oas" => cv::Oas::default().fit_view(x),
        "empirical" => cv::EmpiricalCovariance::default().fit_view(x),
        "shrunk" => cv::ShrunkCovariance {
            shrinkage: o.shrinkage.unwrap_or(0.1),
            ..Default::default()
        }
        .fit_view(x),
        "mcd" => cv::MinCovDet {
            support_fraction: o.support_fraction,
            seed: o.seed.unwrap_or(0),
            ..Default::default()
        }
        .fit_view(x),
        m => {
            return Err(format!(
                "method must be \"ledoit_wolf\", \"oas\", \"empirical\", \"shrunk\" or \"mcd\", got \"{m}\""
            ))
        }
    }
    .map_err(|e| e.to_string())?;
    let shrinkage = match method {
        "empirical" => 0.0,
        _ => fitted.shrinkage().unwrap_or(f64::NAN),
    };
    let mut out: Vec<f64> = fitted
        .covariance()
        .as_standard_layout()
        .iter()
        .copied()
        .collect();
    out.push(shrinkage);
    out.extend(fitted.location().iter().copied());
    Ok(out)
}

fn bandwidth(x: ArrayView2<f64>, options: &str) -> Result<Vec<f64>, String> {
    let o: BandwidthOptions = parse(options)?;
    anofox_ml_cluster::estimate_bandwidth(
        x,
        o.quantile.unwrap_or(0.3),
        o.n_samples,
        o.seed.unwrap_or(0),
    )
    .map(|b| vec![b])
    .map_err(|e| e.to_string())
}

/// Statistic `name` (`mutual_info`, `covariance`, `bandwidth`) of `n × d`
/// row-major `x`; `mutual_info` also takes `y` (doubles) or `y_labels`
/// (strings). `out` receives the values (see each function).
///
/// # Safety
/// Strings are NUL-terminated (`options` may be NULL); arrays hold the
/// stated number of values; `out` is writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_stat(
    name: *const c_char,
    options: *const c_char,
    x: *const f64,
    n: usize,
    d: usize,
    y: *const f64,
    y_labels: *const *const c_char,
    out: *mut AnofoxMlDoubles,
) -> *mut c_char {
    let run = || -> Result<Vec<f64>, String> {
        let name = str_arg(name, "name")?;
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        let x = matrix(x, n, d)?;
        match name {
            "mutual_info" => {
                if !y_labels.is_null() {
                    let (_, codes) = encode_labels(&strings(y_labels, n, "y")?);
                    mutual_info(x, Array1::from(codes), true, options)
                } else if !y.is_null() {
                    mutual_info(
                        x,
                        Array1::from(std::slice::from_raw_parts(y, n).to_vec()),
                        false,
                        options,
                    )
                } else {
                    Err("mutual_info needs y".into())
                }
            }
            "covariance" => covariance(x, options),
            "bandwidth" => bandwidth(x, options),
            _ => Err(format!("unknown statistic '{name}'")),
        }
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(v)) => {
            *out = into_doubles(v);
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while computing a statistic"),
    }
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct EmbeddingOptions {
    n_components: Option<usize>,
    /// Drop the first (constant) eigenvector, default true.
    drop_first: Option<bool>,
}

/// Spectral embedding of a graph of `n` nodes from `m` weighted edges
/// (`src[e] -> dst[e]`, added in both directions); `out` receives
/// `n × out_cols` doubles, row-major (node order as the indices).
///
/// # Safety
/// `options` is NUL-terminated or NULL; `src` / `dst` / `w` hold `m`
/// values; `out` / `out_cols` are writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_ml_spectral_embedding(
    options: *const c_char,
    n: usize,
    src: *const u64,
    dst: *const u64,
    w: *const f64,
    m: usize,
    out: *mut AnofoxMlDoubles,
    out_cols: *mut usize,
) -> *mut c_char {
    let run = || -> Result<(Vec<f64>, usize), String> {
        let options = if options.is_null() {
            ""
        } else {
            str_arg(options, "options")?
        };
        let o: EmbeddingOptions = parse(options)?;
        if m == 0 || src.is_null() || dst.is_null() || w.is_null() {
            return Err("no edges".into());
        }
        let (s, d, w) = (
            std::slice::from_raw_parts(src, m),
            std::slice::from_raw_parts(dst, m),
            std::slice::from_raw_parts(w, m),
        );
        let mut edges = Vec::with_capacity(2 * m);
        for e in 0..m {
            let (i, j) = (s[e] as usize, d[e] as usize);
            if i >= n || j >= n {
                return Err("edge outside the node range".into());
            }
            edges.push((i, j, w[e]));
            edges.push((j, i, w[e]));
        }
        let k = o.n_components.unwrap_or(2);
        let emb =
            anofox_ml_manifold::spectral_embedding(n, &edges, k, o.drop_first.unwrap_or(true))
                .map_err(|e| e.to_string())?;
        let cols = emb.ncols();
        Ok((emb.as_standard_layout().iter().copied().collect(), cols))
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok((v, c))) => {
            *out = into_doubles(v);
            *out_cols = c;
            std::ptr::null_mut()
        }
        Ok(Err(e)) => err(e),
        Err(_) => err("internal error (panic) while embedding"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn informative_features_score_higher() {
        let n = 300;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                (i % 30) as f64
            } else {
                ((i * 7919) % 13) as f64
            }
        });
        let y = Array1::from_shape_fn(n, |i| (i % 30) as f64 * 2.0);
        let mi = mutual_info(x.view(), y, false, "").unwrap();
        assert!(mi[0] > 1.0 && mi[1] < 0.1, "{mi:?}");
        let c = covariance(x.view(), r#"{"method": "empirical"}"#).unwrap();
        assert_eq!(c.len(), 7); // 2 × 2 covariance, shrinkage, 2 location values
        assert!(c[1].abs() < c[0], "{c:?}");
        let lw = covariance(x.view(), "").unwrap();
        assert!(lw[4] >= 0.0 && lw[4] <= 1.0);
        assert!(bandwidth(x.view(), "").unwrap()[0] > 0.0);
        assert!(covariance(x.view(), r#"{"method": "x"}"#).is_err());
    }
}
