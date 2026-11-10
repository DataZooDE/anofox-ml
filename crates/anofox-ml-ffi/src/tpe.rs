//! Tree-structured Parzen estimator (Bergstra et al. 2011) for
//! `ml_tune_agg` with `search: "tpe"`, after Optuna's default univariate
//! `TPESampler`: the first `n_startup` trials are random; after that the
//! completed trials are split into the best `min(ceil(0.1 n), 25)` and the
//! rest, each dimension gets a Parzen estimator `l` over the best and `g`
//! over the rest, and of `n_ei` draws from `l` the one with the largest
//! `l(x) / g(x)` is proposed.
//!
//! Numeric dimensions use a mixture of Gaussians truncated to the range
//! (on the log scale for `log` dimensions, integers on `[lo - 0.5, hi + 0.5]`)
//! plus a prior component over the whole range; choices use counts plus a
//! uniform prior weight of 1.

use rand::rngs::StdRng;
use rand::Rng;
use serde_json::{json, Map, Value};

use crate::tune::{sample, Dim};

/// Next parameter set, given the successful trials so far as
/// (params, score).
pub(crate) fn propose(
    space: &[(String, Dim)],
    history: &[(&Map<String, Value>, f64)],
    minimise: bool,
    n_startup: usize,
    n_ei: usize,
    rng: &mut StdRng,
) -> Map<String, Value> {
    if history.len() < n_startup.max(1) {
        return space
            .iter()
            .map(|(n, d)| (n.clone(), sample(d, rng)))
            .collect();
    }
    // Best first; ties keep the earlier trial.
    let mut order: Vec<usize> = (0..history.len()).collect();
    order.sort_by(|&a, &b| {
        let (sa, sb) = (history[a].1, history[b].1);
        let c = if minimise {
            sa.total_cmp(&sb)
        } else {
            sb.total_cmp(&sa)
        };
        c.then(a.cmp(&b))
    });
    let n_good = (history.len() as f64 * 0.1).ceil().clamp(1.0, 25.0) as usize;
    let (good, bad) = order.split_at(n_good);
    space
        .iter()
        .map(|(name, dim)| {
            let obs = |idx: &[usize]| -> Vec<&Value> {
                idx.iter().filter_map(|&i| history[i].0.get(name)).collect()
            };
            let v = propose_dim(dim, &obs(good), &obs(bad), n_ei.max(1), rng);
            (name.clone(), v)
        })
        .collect()
}

fn propose_dim(dim: &Dim, good: &[&Value], bad: &[&Value], n_ei: usize, rng: &mut StdRng) -> Value {
    match dim {
        Dim::Choice(c) => {
            let weights = |obs: &[&Value]| -> Vec<f64> {
                let mut w = vec![1.0 / c.len() as f64; c.len()];
                for v in obs {
                    if let Some(k) = c.iter().position(|x| x == *v) {
                        w[k] += 1.0;
                    }
                }
                let s: f64 = w.iter().sum();
                w.iter().map(|x| x / s).collect()
            };
            let (l, g) = (weights(good), weights(bad));
            let mut best: Option<(usize, f64)> = None;
            for _ in 0..n_ei {
                let mut u = rng.gen::<f64>();
                let mut k = c.len() - 1;
                for (j, w) in l.iter().enumerate() {
                    if u < *w {
                        k = j;
                        break;
                    }
                    u -= w;
                }
                let score = l[k].ln() - g[k].ln();
                if best.is_none_or(|(_, s)| score > s) {
                    best = Some((k, score));
                }
            }
            c[best.expect("n_ei >= 1").0].clone()
        }
        Dim::Int { lo, hi, log } => {
            let (a, b) = (*lo as f64 - 0.5, *hi as f64 + 0.5);
            let (a, b) = if *log { (a.ln(), b.ln()) } else { (a, b) };
            let to = |v: &Value| v.as_f64().map(|v| if *log { v.ln() } else { v });
            let z = numeric(a, b, good, bad, to, n_ei, rng);
            let v = if *log { z.exp() } else { z };
            json!((v.round() as i64).clamp(*lo, *hi))
        }
        Dim::Float { lo, hi, log } => {
            if lo == hi {
                return json!(*lo);
            }
            let (a, b) = if *log { (lo.ln(), hi.ln()) } else { (*lo, *hi) };
            let to = |v: &Value| v.as_f64().map(|v| if *log { v.ln() } else { v });
            let z = numeric(a, b, good, bad, to, n_ei, rng);
            json!((if *log { z.exp() } else { z }).clamp(*lo, *hi))
        }
    }
}

/// Of `n_ei` draws from `l` on `[a, b]` (internal scale), the one with the
/// largest `ln l - ln g`.
fn numeric(
    a: f64,
    b: f64,
    good: &[&Value],
    bad: &[&Value],
    to: impl Fn(&Value) -> Option<f64>,
    n_ei: usize,
    rng: &mut StdRng,
) -> f64 {
    let l = Parzen::new(good.iter().filter_map(|v| to(v)).collect(), a, b);
    let g = Parzen::new(bad.iter().filter_map(|v| to(v)).collect(), a, b);
    let mut best: Option<(f64, f64)> = None;
    for _ in 0..n_ei {
        let x = l.sample(rng);
        let score = l.log_pdf(x) - g.log_pdf(x);
        if best.is_none_or(|(_, s)| score > s) {
            best = Some((x, score));
        }
    }
    best.expect("n_ei >= 1").0
}

/// Equal-weight mixture of Gaussians truncated to `[a, b]`.
struct Parzen {
    a: f64,
    b: f64,
    mus: Vec<f64>,
    sigmas: Vec<f64>,
}

impl Parzen {
    fn new(obs: Vec<f64>, a: f64, b: f64) -> Self {
        let width = b - a;
        // Observations plus the prior component in the middle of the range.
        let mut mus = obs;
        mus.push(0.5 * (a + b));
        let n = mus.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| mus[i].total_cmp(&mus[j]));
        let min_sigma = width / (1.0 + n as f64).min(100.0);
        let mut sigmas = vec![0.0; n];
        for (r, &i) in order.iter().enumerate() {
            let left = if r == 0 { a } else { mus[order[r - 1]] };
            let right = if r + 1 == n { b } else { mus[order[r + 1]] };
            sigmas[i] = (mus[i] - left).max(right - mus[i]).clamp(min_sigma, width);
        }
        sigmas[n - 1] = width;
        Parzen { a, b, mus, sigmas }
    }

    fn sample(&self, rng: &mut StdRng) -> f64 {
        let k = rng.gen_range(0..self.mus.len());
        let (mu, s) = (self.mus[k], self.sigmas[k]);
        for _ in 0..100 {
            let x = mu + s * std_normal(rng);
            if (self.a..=self.b).contains(&x) {
                return x;
            }
        }
        rng.gen_range(self.a..=self.b)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        let terms: Vec<f64> = self
            .mus
            .iter()
            .zip(&self.sigmas)
            .map(|(&mu, &s)| {
                let z = (x - mu) / s;
                let mass = (phi((self.b - mu) / s) - phi((self.a - mu) / s)).max(1e-300);
                -0.5 * z * z - s.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - mass.ln()
            })
            .collect();
        let m = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        m + terms.iter().map(|t| (t - m).exp()).sum::<f64>().ln() - (terms.len() as f64).ln()
    }
}

/// Box-Muller.
fn std_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = 1.0 - rng.gen::<f64>();
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Standard normal CDF (erf by Abramowitz & Stegun 7.1.26, |error| < 1.5e-7).
fn phi(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-x * x).exp();
    0.5 * (1.0 + if z < 0.0 { -erf } else { erf })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn phi_matches_known_values() {
        assert!((phi(0.0) - 0.5).abs() < 1e-7);
        assert!((phi(1.959_964) - 0.975).abs() < 1e-6);
        assert!((phi(-1.0) - 0.158_655_25).abs() < 1e-6);
    }

    #[test]
    fn parzen_density_integrates_to_one_on_the_range() {
        let p = Parzen::new(vec![0.2, 0.25, 0.9], 0.0, 1.0);
        let n = 10_000;
        let integral: f64 = (0..n)
            .map(|i| p.log_pdf((i as f64 + 0.5) / n as f64).exp() / n as f64)
            .sum();
        assert!((integral - 1.0).abs() < 1e-3, "{integral}");
    }

    #[test]
    fn proposals_concentrate_near_the_good_trials() {
        // Score = -(x - 3)^2: after the start-up trials the proposals should
        // move towards x = 3.
        let space = vec![(
            "x".to_string(),
            Dim::Float {
                lo: -10.0,
                hi: 10.0,
                log: false,
            },
        )];
        let mut rng = StdRng::seed_from_u64(0);
        let mut hist: Vec<(Map<String, Value>, f64)> = Vec::new();
        for _ in 0..60 {
            let h: Vec<(&Map<String, Value>, f64)> = hist.iter().map(|(p, s)| (p, *s)).collect();
            let p = propose(&space, &h, false, 10, 24, &mut rng);
            let x = p["x"].as_f64().unwrap();
            hist.push((p, -(x - 3.0).powi(2)));
        }
        let last: Vec<f64> = hist[50..]
            .iter()
            .map(|(p, _)| p["x"].as_f64().unwrap())
            .collect();
        let mean_dist = last.iter().map(|x| (x - 3.0).abs()).sum::<f64>() / last.len() as f64;
        assert!(mean_dist < 1.5, "{last:?}");
    }
}
