// src/bin/hb_spectrum_flux.rs
//
// HB (2D) GEP with Peierls flux (Hol = −1 via flux = π) and cusp geometry.
// Now with an automated Bisection tuner for eps_e to hit a target spectral ratio.

use std::fs::File;
use std::io::Write;

use anyhow::Result;
use clap::Parser;
use nalgebra::linalg::{Cholesky, SymmetricEigen};
use nalgebra::DMatrix;
use ndarray::Array1;
use ndarray_npy::{read_npy, NpzReader};
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;
use std::cmp::Ordering;

// ------------------------- CLI -------------------------

#[derive(Parser, Debug, Clone)]
#[command(author, version, about="HB 2D GEP with cusp + Peierls flux (Hol = −1)", long_about=None)]
struct Args {
    // grid / geometry
    #[arg(long, default_value_t = 512)] nx: usize,
    #[arg(long, default_value_t = 512)] ny: usize,
    #[arg(long, default_value_t = 1.0)] r_out: f64,
    #[arg(long, default_value_t = 0.40)] r_in: f64,

    // anisotropy / cusp
    #[arg(long, default_value_t = 0.02)] eps_b: f64,
    #[arg(long, default_value_t = 0.05)] eps_e: f64,

    // optional small potential (coercivity nudge): H ← H + τ I
    #[arg(long, default_value_t = 0.0)] tau: f64,

    // flux: default π (fermionic holonomy)
    #[arg(long, default_value_t = std::f64::consts::PI)] flux: f64,

    // solver block/iters
    #[arg(long, default_value_t = 6)] m: usize,
    #[arg(long)] k: Option<usize>,
    #[arg(long, default_value_t = 200_000)] max_it: usize,
    #[arg(long, default_value_t = 1e-8)] tol: f64,
    #[arg(long, default_value_t = 42)] seed: u64,
    #[arg(long, default_value_t = 50)] min_iters: usize,

    // logging / output
    #[arg(long, default_value_t = 100)] log_every: usize,
    #[arg(long, default_value_t = 1000)] checkpoint_every: usize,
    #[arg(long, default_value = "out/hb_flux")] out_prefix: String,

    // pipeline nicety (accepted, ignored)
    #[arg(long, default_value = "json")] save_format: String,
    #[arg(long, default_value = "npy")] vec_format: String,

    // external sparse operator (SciPy CSR .npz)
    #[arg(long)] csr_op: Option<String>, // path to NPZ with {data,indices,indptr,shape}
    #[arg(long, default_value_t = 1.0)] csr_scale: f64,

    // external scalar potential U(x): diagonal injection
    #[arg(long)] u_from: Option<String>, // path to .npy, length = nx*ny (row-major)
    #[arg(long, default_value_t = 1.0)] u_scale: f64,
    #[arg(long, default_value_t = 0.0)] u_floor: f64,

    // optional deflation
    #[arg(long)] deflate: Vec<String>,

    // optional reference λ1 for ratio
    #[arg(long)] lambda1_from: Option<String>,

    // ------------------- CUSP TUNER ARGS -------------------
    #[arg(long)] target_ratio: Option<f64>,
    #[arg(long)] search_eps_e_min: Option<f64>,
    #[arg(long)] search_eps_e_max: Option<f64>,
    #[arg(long, default_value_t = 10)] tuner_max_iters: usize,
    // -------------------------------------------------------
}

// ------------------------- Grid / Operator -------------------------

#[derive(Clone, Copy)]
struct GridParams {
    nx: usize,
    ny: usize,
    r_in: f64,
    r_out: f64,
    eps_b: f64,
    eps_e: f64,
}

struct CsrOp {
    n: usize,
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f64>,
    scale: f64,
}

struct Grid {
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    x0: f64,
    y0: f64,
    r_in: f64,
    r_out: f64,
    mask: Vec<bool>,
    k: Vec<f64>,
    m_w: Vec<f64>, // Mass diag: area / k (⟨·,·⟩_M = ∫ k^{-1} u v)
    active: usize,
    active_in_row: Vec<Vec<usize>>, // for each row j, the list of active i
}

impl Grid {
    fn new(p: GridParams) -> Self {
        let nx = p.nx;
        let ny = p.ny;
        let lx = 2.0 * p.r_out;
        let ly = 2.0 * p.r_out;
        let dx = lx / nx as f64;
        let dy = ly / ny as f64;
        let x0 = -p.r_out + 0.5 * dx;
        let y0 = -p.r_out + 0.5 * dy;

        let mut mask = vec![false; nx * ny];
        let mut k = vec![0.0; nx * ny];
        let mut m_w = vec![0.0; nx * ny];
        let mut active = 0usize;

        for j in 0..ny {
            for i in 0..nx {
                let x = x0 + i as f64 * dx;
                let y = y0 + j as f64 * dy;
                let r = (x * x + y * y).sqrt();
                let idx = j * nx + i;

                if r >= p.r_in && r <= p.r_out {
                    mask[idx] = true;
                    active += 1;

                    let theta = y.atan2(x);
                    let cos2t = (2.0 * theta).cos();
                    let cos_t = theta.cos();
                    let sgn = if cos_t >= 0.0 { 1.0 } else { -1.0 };

                    let mut kval = r * r * (1.0 + p.eps_b * cos2t + p.eps_e * sgn);
                    if kval < 1e-12 {
                        kval = 1e-12;
                    }
                    k[idx] = kval;

                    m_w[idx] = (dx * dy) / k[idx];
                }
            }
        }

        // per-row active index lists
        let mut active_in_row = vec![Vec::<usize>::new(); ny];
        for j in 0..ny {
            for i in 0..nx {
                if mask[j * nx + i] {
                    active_in_row[j].push(i);
                }
            }
        }

        Grid {
            nx,
            ny,
            dx,
            dy,
            x0,
            y0,
            r_in: p.r_in,
            r_out: p.r_out,
            mask,
            k,
            m_w,
            active,
            active_in_row,
        }
    }
    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
}

// ------------------------- HBFlux -------------------------

struct HBFlux<'g> {
    g: &'g Grid,
    inv_dx2: f64,
    inv_dy2: f64,
    flux: f64,
    tau: f64,
    diag: Vec<f64>, // Jacobi diagonal
    dth_e: Vec<f64>, // east-face Δθ
    dth_n: Vec<f64>, // north-face Δθ
    csr: Option<CsrOp>,
    u: Option<Vec<f64>>, // scalar potential, same length as grid
}

impl<'g> HBFlux<'g> {
    fn new(g: &'g Grid, flux: f64, tau: f64, csr: Option<CsrOp>, u: Option<Vec<f64>>) -> Self {
        let inv_dx2 = 1.0 / (g.dx * g.dx);
        let inv_dy2 = 1.0 / (g.dy * g.dy);

        let mut diag = vec![0.0f64; g.nx * g.ny];
        for j in 0..g.ny {
            for i in 0..g.nx {
                let idx = g.idx(i, j);
                if !g.mask[idx] {
                    continue;
                }
                let kc = g.k[idx];

                let mut sum = 0.0;
                if i + 1 < g.nx && g.mask[g.idx(i + 1, j)] {
                    sum += harmonic(kc, g.k[g.idx(i + 1, j)]) * inv_dx2;
                } else {
                    sum += kc * inv_dx2;
                }
                if i >= 1 && g.mask[g.idx(i - 1, j)] {
                    sum += harmonic(kc, g.k[g.idx(i - 1, j)]) * inv_dx2;
                } else {
                    sum += kc * inv_dx2;
                }
                if j + 1 < g.ny && g.mask[g.idx(i, j + 1)] {
                    sum += harmonic(kc, g.k[g.idx(i, j + 1)]) * inv_dy2;
                } else {
                    sum += kc * inv_dy2;
                }
                if j >= 1 && g.mask[g.idx(i, j - 1)] {
                    sum += harmonic(kc, g.k[g.idx(i, j - 1)]) * inv_dy2;
                } else {
                    sum += kc * inv_dy2;
                }

                sum += tau;
                if let Some(ref uu) = u {
                    sum += uu[idx];
                }

                diag[idx] = sum.max(1e-30);
            }
        }

        // Analytic, wrapped Peierls links (Δθ), atan2(y, x)
        #[inline]
        fn wrap_pm_pi(a: f64) -> f64 {
            let two_pi = std::f64::consts::TAU;
            let mut x = a - two_pi * (a / two_pi).round();
            if x <= -std::f64::consts::PI {
                x += two_pi;
            }
            if x > std::f64::consts::PI {
                x -= two_pi;
            }
            x
        }

        let n = g.nx * g.ny;
        let mut dth_e = vec![0.0f64; n];
        let mut dth_n = vec![0.0f64; n];

        for j in 0..g.ny {
            let y_c = g.y0 + (j as f64) * g.dy;
            for i in 0..g.nx {
                let idx = g.idx(i, j);
                if !g.mask[idx] {
                    continue;
                }
                let x_c = g.x0 + (i as f64) * g.dx;

                // east face from x_c → x_c+dx at fixed y_c
                if i + 1 < g.nx && g.mask[g.idx(i + 1, j)] {
                    let x_r = x_c + g.dx;
                    let a0 = y_c.atan2(x_c);
                    let a1 = y_c.atan2(x_r);
                    dth_e[idx] = wrap_pm_pi(a0 - a1);
                } else {
                    dth_e[idx] = 0.0;
                }

                // north face from y_c → y_c+dy at fixed x_c
                if j + 1 < g.ny && g.mask[g.idx(i, j + 1)] {
                    let y_t = y_c + g.dy;
                    let b1 = y_t.atan2(x_c);
                    let b0 = y_c.atan2(x_c);
                    dth_n[idx] = wrap_pm_pi(b1 - b0);
                } else {
                    dth_n[idx] = 0.0;
                }
            }
        }

        Self {
            g,
            inv_dx2,
            inv_dy2,
            flux,
            tau,
            diag,
            dth_e,
            dth_n,
            csr,
            u,
        }
    }

    // y = H x (complex)
    fn matvec(&self, x: &[Complex64], y: &mut [Complex64]) {
        let g = self.g;

        // zero y once; we only write active entries below
        for v in y.iter_mut() {
            *v = Complex64::new(0.0, 0.0);
        }

        y.par_chunks_mut(g.nx).enumerate().for_each(|(j, yrow)| {
            for &i in g.active_in_row[j].iter() {
                let idx = g.idx(i, j);

                let kc = g.k[idx];
                let xc = x[idx];
                let mut acc = Complex64::new(0.0, 0.0);

                // East
                if i + 1 < g.nx && g.mask[g.idx(i + 1, j)] {
                    let idx_e = g.idx(i + 1, j);
                    let ke = harmonic(kc, g.k[idx_e]);
                    let dth = self.dth_e[idx];
                    let phase =
                        Complex64::from_polar(1.0, self.flux * dth / std::f64::consts::TAU);
                    acc += ke * (phase * x[idx_e] - xc) * self.inv_dx2;
                } else {
                    acc += kc * (Complex64::new(0.0, 0.0) - xc) * self.inv_dx2;
                }

                // West (conjugate)
                if i >= 1 && g.mask[g.idx(i - 1, j)] {
                    let idx_w = g.idx(i - 1, j);
                    let kw = harmonic(kc, g.k[idx_w]);
                    let dth = self.dth_e[idx_w];
                    let phase =
                        Complex64::from_polar(1.0, -self.flux * dth / std::f64::consts::TAU);
                    acc += kw * (phase * x[idx_w] - xc) * self.inv_dx2;
                } else {
                    acc += kc * (Complex64::new(0.0, 0.0) - xc) * self.inv_dx2;
                }

                // North
                if j + 1 < g.ny && g.mask[g.idx(i, j + 1)] {
                    let idx_n = g.idx(i, j + 1);
                    let kn = harmonic(kc, g.k[idx_n]);
                    let dth = self.dth_n[idx];
                    let phase =
                        Complex64::from_polar(1.0, self.flux * dth / std::f64::consts::TAU);
                    acc += kn * (phase * x[idx_n] - xc) * self.inv_dy2;
                } else {
                    acc += kc * (Complex64::new(0.0, 0.0) - xc) * self.inv_dy2;
                }

                // South (conjugate)
                if j >= 1 && g.mask[g.idx(i, j - 1)] {
                    let idx_s = g.idx(i, j - 1);
                    let ks = harmonic(kc, g.k[idx_s]);
                    let dth = self.dth_n[idx_s];
                    let phase =
                        Complex64::from_polar(1.0, -self.flux * dth / std::f64::consts::TAU);
                    acc += ks * (phase * x[idx_s] - xc) * self.inv_dy2;
                } else {
                    acc += kc * (Complex64::new(0.0, 0.0) - xc) * self.inv_dy2;
                }

                let mut res = -acc + Complex64::new(self.tau, 0.0) * xc;

                // diagonal potential contribution
                if let Some(ref uu) = self.u {
                    res += Complex64::new(uu[idx], 0.0) * xc;
                }

                // add external CSR row * x
                if let Some(ref csr) = self.csr {
                    let start = csr.indptr[idx] as usize;
                    let end = csr.indptr[idx + 1] as usize;
                    let mut add = Complex64::new(0.0, 0.0);
                    let s = csr.scale;
                    for p in start..end {
                        let jcol = csr.indices[p] as usize;
                        if self.g.mask[jcol] {
                            // real scalar times complex (cheap)
                            add += x[jcol] * (csr.data[p] * s);
                        }
                    }
                    res += add;
                }

                yrow[i] = res;
            }
        });
    }

    fn apply_mass(&self, x: &[Complex64], y: &mut [Complex64]) {
        for i in 0..x.len() {
            y[i] = Complex64::new(self.g.m_w[i], 0.0) * x[i];
        }
    }
    fn apply_jacobi(&self, r: &[Complex64], out: &mut [Complex64]) {
        for i in 0..r.len() {
            let d = self.diag[i];
            out[i] = if d > 0.0 {
                r[i] / Complex64::new(d, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
        }
    }
}

// ------------------------- utilities / diagnostics -------------------------

#[inline(always)]
fn harmonic(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        0.0
    } else {
        2.0 * a * b / (a + b)
    }
}

#[inline(always)]
fn normalize_angle(a: f64) -> f64 {
    let mut th = a;
    while th <= -std::f64::consts::PI {
        th += std::f64::consts::TAU;
    }
    while th > std::f64::consts::PI {
        th -= std::f64::consts::TAU;
    }
    th
}

// Robust NPZ loader (accepts "data.npy", "data", and nested paths)
fn load_csr_npz(path: &str, expected_n: usize, scale: f64) -> anyhow::Result<CsrOp> {
    use anyhow::{bail, Context};

    let f = std::fs::File::open(path).with_context(|| format!("open npz: {}", path))?;
    let mut npz = NpzReader::new(f).with_context(|| format!("parse npz: {}", path))?;

    let names = npz.names().with_context(|| "npz: list names")?;

    let resolve = |key: &str| -> anyhow::Result<String> {
        let want_dot = format!("{key}.npy");
        if names.iter().any(|n| n == &want_dot) {
            return Ok(want_dot);
        }
        if names.iter().any(|n| n == key) {
            return Ok(key.to_string());
        }
        if let Some(n) = names
            .iter()
            .find(|n| n.rsplit('/').next().map(|b| b) == Some(&want_dot[..]))
        {
            return Ok(n.clone());
        }
        if let Some(n) = names.iter().find(|n| {
            let base = n.rsplit('/').next().unwrap_or(n);
            base.strip_suffix(".npy").unwrap_or(base) == key
        }) {
            return Ok(n.clone());
        }
        bail!("key '{}' not found in {} (have: {:?})", key, path, names);
    };

    let k_data = resolve("data")?;
    let k_indices = resolve("indices")?;
    let k_indptr = resolve("indptr")?;
    let k_shape = resolve("shape")?;

    let data: Array1<f64> = npz
        .by_name(&k_data)
        .with_context(|| format!("npz missing {}", k_data))?;
    let indices: Array1<i32> = npz
        .by_name(&k_indices)
        .with_context(|| format!("npz missing {}", k_indices))?;
    let indptr: Array1<i32> = npz
        .by_name(&k_indptr)
        .with_context(|| format!("npz missing {}", k_indptr))?;
    let shape: Array1<i64> = npz
        .by_name(&k_shape)
        .with_context(|| format!("npz missing {}", k_shape))?;

    anyhow::ensure!(shape.len() == 2, "shape must have 2 entries, got {}", shape.len());
    let (nr, nc) = (shape[0] as usize, shape[1] as usize);
    anyhow::ensure!(nr == nc, "csr_op must be square (got {}x{})", nr, nc);
    anyhow::ensure!(nr == expected_n, "csr_op dim {} != nx*ny {}", nr, expected_n);

    Ok(CsrOp {
        n: nr,
        indptr: indptr.to_vec(),
        indices: indices.to_vec(),
        data: data.to_vec(),
        scale,
    })
}

// ---------- Loader for external potential (.npy) ----------
fn load_potential_npy(
    path: &str,
    nx: usize,
    ny: usize,
    scale: f64,
    floor: f64,
) -> anyhow::Result<Vec<f64>> {
    let arr: Array1<f64> = read_npy(path)?;
    let n = nx * ny;
    anyhow::ensure!(arr.len() == n, "u-from length {} does not match nx*ny {}", arr.len(), n);
    let mut v = arr.to_vec();
    for x in &mut v {
        if *x < floor {
            *x = floor;
        }
        *x *= scale;
    }
    Ok(v)
}

// --- Plaquette Diagnostic (uses interior active cells only) ---
fn diag_plaquette(g: &Grid, op: &HBFlux, res: usize) {
    if g.nx < 2 || g.ny < 2 {
        return;
    }
    let mut max_curl_a = 0.0;
    let mut max_curl_i = 0.0;
    let step_x = (g.nx.max(1) / res).max(1);
    let step_y = (g.ny.max(1) / res).max(1);

    for j in 0..g.ny - 1 {
        for i in 0..g.nx - 1 {
            let idx = g.idx(i, j);
            let idx_e = g.idx(i + 1, j);
            let idx_n = g.idx(i, j + 1);
            let idx_ne = g.idx(i + 1, j + 1);

            if !g.mask[idx] || !g.mask[idx_e] || !g.mask[idx_n] || !g.mask[idx_ne] {
                continue;
            }

            let e00 = op.dth_e[idx];
            let n00 = op.dth_n[idx];
            let e01 = op.dth_e[idx_n];
            let n10 = op.dth_n[idx_e];
            let curl = normalize_angle(e00 + n00 - e01 - n10).abs();

            if curl > max_curl_a {
                max_curl_a = curl;
            }
            if i % step_x == 0 && j % step_y == 0 {
                if curl > max_curl_i {
                    max_curl_i = curl;
                }
            }
        }
    }
    println!(
        "[diag] Max plaquette |Δθ loop| (any={}, interior={}) = {:.3e}, {:.3e}",
        g.nx * g.ny, res, max_curl_a, max_curl_i
    );
}

// ------------------------- Small algebra / logging -------------------------

fn project_block_b(X: &mut [Vec<Complex64>], D: &[Vec<Complex64>], g: &Grid) {
    if D.is_empty() {
        return;
    }
    for v in X.iter_mut() {
        for d in D {
            let mut alpha = Complex64::new(0.0, 0.0);
            for i in 0..v.len() {
                if g.mask[i] {
                    alpha += d[i].conj() * Complex64::new(g.m_w[i], 0.0) * v[i];
                }
            }
            for i in 0..v.len() {
                if g.mask[i] {
                    v[i] -= alpha * d[i];
                }
            }
        }
    }
}

fn b_orthonormalize(x: &mut Vec<Vec<Complex64>>, g: &Grid) {
    let m = x.len();
    if m == 0 {
        return;
    }
    let n = x[0].len();

    let mut b = DMatrix::<Complex64>::zeros(m, m);
    let mut tr = 0.0;
    for i in 0..m {
        for j in i..m {
            let mut s = Complex64::new(0.0, 0.0);
            for k in 0..n {
                if g.mask[k] {
                    s += x[i][k].conj() * Complex64::new(g.m_w[k], 0.0) * x[j][k];
                }
            }
            b[(i, j)] = s;
            b[(j, i)] = s.conj();
        }
        tr += b[(i, i)].re;
    }
    let mut chol = Cholesky::new(b.clone());
    if chol.is_none() {
        let mut delta = 1e-12 * (tr / (m as f64) + 1.0);
        for _ in 0..6 {
            let mut breg = b.clone();
            for i in 0..m {
                breg[(i, i)] += Complex64::new(delta, 0.0);
            }
            if let Some(ch) = Cholesky::new(breg) {
                chol = Some(ch);
                break;
            }
            delta *= 10.0;
        }
    }
    let chol = chol.expect("B not SPD");
    let l = chol.l();
    let linv = l.clone().try_inverse().expect("L^{-1}");

    let p = m;
    let mut out = vec![vec![Complex64::new(0.0, 0.0); n]; p];
    for c in 0..p {
        for r in 0..p {
            let alpha = linv[(r, c)];
            if alpha != Complex64::new(0.0, 0.0) {
                for i in 0..n {
                    if g.mask[i] {
                        out[c][i] += alpha * x[r][i];
                    }
                }
            }
        }
    }
    *x = out;
}

fn gram_ax(x: &Vec<Vec<Complex64>>, ax: &Vec<Vec<Complex64>>, g: &Grid) -> DMatrix<Complex64> {
    let m = x.len();
    let mut t = DMatrix::<Complex64>::zeros(m, m);
    for i in 0..m {
        for j in i..m {
            let mut s = Complex64::new(0.0, 0.0);
            for k in 0..x[i].len() {
                if g.mask[k] {
                    s += x[i][k].conj() * ax[j][k];
                }
            }
            t[(i, j)] = s;
            t[(j, i)] = s.conj();
        }
    }
    t
}

fn gram_b(x: &Vec<Vec<Complex64>>, g: &Grid) -> DMatrix<Complex64> {
    let m = x.len();
    let mut t = DMatrix::<Complex64>::zeros(m, m);
    for i in 0..m {
        for j in i..m {
            let mut s = Complex64::new(0.0, 0.0);
            for k in 0..x[i].len() {
                if g.mask[k] {
                    s += x[i][k].conj() * Complex64::new(g.m_w[k], 0.0) * x[j][k];
                }
            }
            t[(i, j)] = s;
            t[(j, i)] = s.conj();
        }
    }
    t
}

#[inline]
fn mat_mul_small(S: &Vec<Vec<Complex64>>, Z: &DMatrix<Complex64>, g: &Grid) -> Vec<Vec<Complex64>> {
    let n = S[0].len();
    let p = S.len();
    let m = Z.ncols();
    let mut out = vec![vec![Complex64::new(0.0, 0.0); n]; m];
    for c in 0..m {
        for r in 0..p {
            let alpha = Z[(r, c)];
            if alpha != Complex64::new(0.0, 0.0) {
                for i in 0..n {
                    if g.mask[i] {
                        out[c][i] += alpha * S[r][i];
                    }
                }
            }
        }
    }
    out
}

fn sym_eig_small_gen(
    T: &DMatrix<Complex64>,
    B: &DMatrix<Complex64>,
) -> (Vec<f64>, DMatrix<Complex64>) {
    let mut chol = Cholesky::new(B.clone());
    if chol.is_none() {
        let m = B.nrows();
        let mut tr = 0.0;
        for i in 0..m {
            tr += B[(i, i)].re;
        }
        let mut delta = 1e-14 * (tr / (m as f64) + 1.0);
        for _ in 0..6 {
            let mut breg = B.clone();
            for i in 0..m {
                breg[(i, i)] = B[(i, i)] + Complex64::new(delta, 0.0);
            }
            if let Some(ch) = Cholesky::new(breg) {
                chol = Some(ch);
                break;
            }
            delta *= 10.0;
        }
    }
    let chol = chol.expect("B not SPD");
    let l = chol.l();
    let linv = l.clone().try_inverse().expect("L inv");

    let mut c = linv.adjoint() * T * linv.clone();
    c = (c.clone() + c.adjoint()) * Complex64::new(0.5, 0.0);

    let c_re: DMatrix<f64> = c.map(|z| z.re);
    let se = SymmetricEigen::new(c_re);

    let evals0: Vec<f64> = se.eigenvalues.as_slice().to_vec();
    let z_re = se.eigenvectors;
    let z_c: DMatrix<Complex64> = z_re.map(|r| Complex64::new(r, 0.0));
    let Y_uns = linv.adjoint() * z_c;

    let mcols = evals0.len();
    let mut idx: Vec<usize> = (0..mcols).collect();
    idx.sort_by(|&i, &j| {
        let a = evals0[i];
        let b = evals0[j];
        if a.is_nan() && b.is_nan() {
            Ordering::Equal
        } else if a.is_nan() {
            Ordering::Greater
        } else if b.is_nan() {
            Ordering::Less
        } else {
            a.partial_cmp(&b).unwrap()
        }
    });

    let evals = idx.iter().map(|&i| evals0[i]).collect::<Vec<_>>();
    let mut Y = DMatrix::<Complex64>::zeros(Y_uns.nrows(), mcols);
    for (new_c, &old_c) in idx.iter().enumerate() {
        Y.set_column(new_c, &Y_uns.column(old_c));
    }
    (evals, Y)
}

// ------------------------- Run logging -------------------------

#[derive(Serialize)]
struct RunMeta {
    nx: usize,
    ny: usize,
    r_in: f64,
    r_out: f64,
    eps_b: f64,
    eps_e: f64,
    flux: f64,
    tau: f64,
    m: usize,
    max_it: usize,
    tol: f64,
    out_prefix: String,
    u_from: Option<String>,
    u_scale: f64,
    u_floor: f64,
}

fn write_eigs_json(prefix: &str, iter: usize, max_rel: f64, evals: &[f64]) -> Result<()> {
    let path = format!("{}-eigs.json", prefix);
    let mut f = File::create(&path)?;
    writeln!(
        f,
        "{}",
        json!({
            "iter": iter, "max_rel": max_rel,
            "evals": evals.iter().enumerate().map(|(i,&l)| json!({"i":i, "lambda": l})).collect::<Vec<_>>()
        })
    )?;
    Ok(())
}

fn write_run_json(
    prefix: &str,
    meta: &RunMeta,
    max_rel: f64,
    evals: &[f64],
    ratio_mu: Option<f64>,
    note: &str,
) -> Result<()> {
    let path = format!("{}-run.json", prefix);
    let mut f = File::create(&path)?;
    let j = json!({
        "meta": meta,
        "max_rel": max_rel,
        "evals": evals.iter().enumerate().map(|(i,&l)| json!({"i":i, "lambda": l})).collect::<Vec<_>>(),
        "ratio_mu": ratio_mu.unwrap_or(0.0),
        "note": note
    });
    writeln!(f, "{}", j)?;
    Ok(())
}

// ------------------------- Tuner: solve_for_ratio -------------------------

fn solve_for_ratio(args: &Args, eps_e_val: f64) -> Result<(f64, f64)> {
    let mut current_args = args.clone();
    current_args.eps_e = eps_e_val;

    let gp = GridParams {
        nx: current_args.nx,
        ny: current_args.ny,
        r_in: current_args.r_in,
        r_out: current_args.r_out,
        eps_b: current_args.eps_b,
        eps_e: current_args.eps_e,
    };
    let grid = Grid::new(gp);

    let csr_opt = if let Some(ref p) = current_args.csr_op {
        Some(load_csr_npz(p, grid.nx * grid.ny, current_args.csr_scale)?)
    } else {
        None
    };

    let u_opt = if let Some(ref path) = current_args.u_from {
        Some(load_potential_npy(
            path,
            grid.nx,
            grid.ny,
            current_args.u_scale,
            current_args.u_floor,
        )?)
    } else {
        None
    };

    let hop = HBFlux::new(&grid, current_args.flux, current_args.tau, csr_opt, u_opt);
    let nx = grid.nx;
    let ny = grid.ny;

    let mut rng = StdRng::seed_from_u64(current_args.seed);

    // Angular seeding
    let mut X: Vec<Vec<Complex64>> = Vec::with_capacity(current_args.m);
    let ang = |i: usize, j: usize| -> f64 {
        let x = grid.x0 + (i as f64) * grid.dx;
        let y = grid.y0 + (j as f64) * grid.dy;
        y.atan2(x)
    };
    // X0 random
    {
        let mut v = vec![Complex64::new(0.0, 0.0); nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                if grid.mask[idx] {
                    v[idx] = Complex64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
                }
            }
        }
        X.push(v);
    }
    // X1
    if current_args.m >= 2 {
        let mut v = vec![Complex64::new(0.0, 0.0); nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                if grid.mask[idx] {
                    let th = ang(i, j);
                    v[idx] = Complex64::from_polar(1.0, th)
                        * Complex64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
                }
            }
        }
        X.push(v);
    }
    // X2
    if current_args.m >= 3 {
        let mut v = vec![Complex64::new(0.0, 0.0); nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                if grid.mask[idx] {
                    let th = ang(i, j);
                    v[idx] = Complex64::from_polar(1.0, -th)
                        * Complex64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
                }
            }
        }
        X.push(v);
    }
    while X.len() < current_args.m {
        let mut v = vec![Complex64::new(0.0, 0.0); nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                if grid.mask[idx] {
                    let th = ang(i, j);
                    let sgn = if (X.len() % 2) == 0 { 1.0 } else { -1.0 };
                    v[idx] = Complex64::from_polar(1.0, sgn * th)
                        * Complex64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
                }
            }
        }
        X.push(v);
    }
    if !current_args.deflate.is_empty() {
        return Err(anyhow::anyhow!(
            "Deflation not supported in tuner mode for simplicity/speed. Run tuner first."
        ));
    }
    b_orthonormalize(&mut X, &grid);

    // workspaces
    let mut AX = vec![vec![Complex64::new(0.0, 0.0); nx * ny]; current_args.m];
    let mut MX = vec![vec![Complex64::new(0.0, 0.0); nx * ny]; current_args.m];
    let mut R = vec![vec![Complex64::new(0.0, 0.0); nx * ny]; current_args.m];
    let mut W = vec![vec![Complex64::new(0.0, 0.0); nx * ny]; current_args.m];

    let mut evals = vec![0.0f64; current_args.m];
    let mut evals_last = vec![0.0f64; current_args.m];
    let mut max_rel = f64::INFINITY;
    let mut locked = vec![false; current_args.m];
    let lock_thresh = current_args.tol.max(1e-9);

    for it in 0..current_args.max_it {
        for j in 0..current_args.m {
            hop.matvec(&X[j], &mut AX[j]);
            hop.apply_mass(&X[j], &mut MX[j]);
        }
        let T0 = gram_ax(&X, &AX, &grid);
        let B0 = gram_b(&X, &grid);
        let (_theta0, Y0) = sym_eig_small_gen(&T0, &B0);
        let mut Xrtz = mat_mul_small(&X, &Y0, &grid);
        b_orthonormalize(&mut Xrtz, &grid);
        X = Xrtz;

        for j in 0..current_args.m {
            hop.matvec(&X[j], &mut AX[j]);
            hop.apply_mass(&X[j], &mut MX[j]);
        }
        let T = gram_ax(&X, &AX, &grid);
        let B = gram_b(&X, &grid);
        let (lam, _Y) = sym_eig_small_gen(&T, &B);
        for j in 0..current_args.m {
            evals[j] = lam[j];
        }
        evals_last.clone_from_slice(&evals);

        max_rel = 0.0;
        for j in 0..current_args.m {
            for i in 0..X[j].len() {
                R[j][i] = AX[j][i] - Complex64::new(evals[j], 0.0) * MX[j][i];
            }
            let mut nr2 = 0.0;
            for i in 0..X[j].len() {
                if grid.mask[i] {
                    nr2 += R[j][i].norm_sqr();
                }
            }
            let rel = nr2.sqrt() / evals[j].abs().max(1.0);
            if rel > max_rel {
                max_rel = rel;
            }
            if it >= current_args.min_iters / 2 && rel <= lock_thresh {
                locked[j] = true;
            }
        }
        if locked.iter().all(|&b| b) {
            break;
        }

        for j in 0..current_args.m {
            if locked[j] {
                for i in 0..W[j].len() {
                    W[j][i] = Complex64::new(0.0, 0.0);
                }
                continue;
            }
            hop.apply_jacobi(&R[j], &mut W[j]);
            let omega = 0.5;
            for _ in 0..4 {
                hop.matvec(&W[j], &mut AX[j]);
                for i in 0..W[j].len() {
                    AX[j][i] = R[j][i] - AX[j][i];
                }
                hop.apply_jacobi(&AX[j], &mut MX[j]);
                for i in 0..W[j].len() {
                    W[j][i] += Complex64::new(omega, 0.0) * MX[j][i];
                }
            }
        }

        let mut bn;
        for j in 0..current_args.m {
            if locked[j] {
                continue;
            }
            bn = 0.0;
            for i in 0..X[j].len() {
                if grid.mask[i] {
                    bn += grid.m_w[i] * (W[j][i].re * W[j][i].re + W[j][i].im * W[j][i].im);
                }
            }
            if bn.sqrt() < 1e-18 {
                for i in 0..W[j].len() {
                    if grid.mask[i] {
                        let ii = i % nx;
                        let jj = i / nx;
                        let x = grid.x0 + (ii as f64) * grid.dx;
                        let y = grid.y0 + (jj as f64) * grid.dy;
                        let th = y.atan2(x);
                        W[j][i] = Complex64::from_polar(1.0, if j % 2 == 0 { th } else { -th })
                            * Complex64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
                    } else {
                        W[j][i] = Complex64::new(0.0, 0.0);
                    }
                }
            }
        }

        let mut S: Vec<Vec<Complex64>> = Vec::with_capacity(2 * current_args.m);
        for j in 0..current_args.m {
            S.push(X[j].clone());
        }
        for j in 0..current_args.m {
            S.push(W[j].clone());
        }
        b_orthonormalize(&mut S, &grid);

        let mut HS: Vec<Vec<Complex64>> =
            (0..S.len()).map(|_| vec![Complex64::new(0.0, 0.0); nx * ny]).collect();
        for j in 0..S.len() {
            hop.matvec(&S[j], &mut HS[j]);
        }
        let A_small = gram_ax(&S, &HS, &grid);
        let B_small = gram_b(&S, &grid);
        let (mu, Z) = sym_eig_small_gen(&A_small, &B_small);
        let k = current_args.m.min(mu.len());
        let Zm = Z.columns(0, k).into_owned();

        let mut Xnew = mat_mul_small(&S, &Zm, &grid);
        b_orthonormalize(&mut Xnew, &grid);
        X = Xnew;
    }

    for j in 0..current_args.m {
        hop.matvec(&X[j], &mut AX[j]);
        hop.apply_mass(&X[j], &mut MX[j]);
    }
    let Tfin = gram_ax(&X, &AX, &grid);
    let Bfin = gram_b(&X, &grid);
    let (theta_fin, _Yfin) = sym_eig_small_gen(&Tfin, &Bfin);
    let final_valid =
        theta_fin.len() >= 2 && theta_fin.iter().all(|v| v.is_finite() && *v > 0.0);
    if final_valid {
        let kfin = current_args.m.min(theta_fin.len());
        for j in 0..kfin {
            evals[j] = theta_fin[j];
        }
    } else {
        evals.clone_from_slice(&evals_last);
    }

    let l1 = evals.get(0).copied().unwrap_or(0.0);
    let l2 = evals.get(1).copied().unwrap_or(l1);
    let ratio = if l1 > 0.0 { l2 / l1 } else { 0.0 };

    Ok((l1, ratio))
}

// ------------------------- Main (tuner + final) -------------------------

fn final_solve_and_log(args: &Args) -> Result<(f64, f64)> {
    let gp = GridParams {
        nx: args.nx,
        ny: args.ny,
        r_in: args.r_in,
        r_out: args.r_out,
        eps_b: args.eps_b,
        eps_e: args.eps_e,
    };
    let grid = Grid::new(gp);

    // define csr_opt in this scope
    let csr_opt = if let Some(ref p) = args.csr_op {
        Some(load_csr_npz(p, grid.nx * grid.ny, args.csr_scale)?)
    } else {
        None
    };

    let u_opt = if let Some(ref path) = args.u_from {
        Some(load_potential_npy(
            path, grid.nx, grid.ny, args.u_scale, args.u_floor,
        )?)
    } else {
        None
    };

    let hop = HBFlux::new(&grid, args.flux, args.tau, csr_opt, u_opt);

    println!("Active cells: {}", grid.active);
    diag_plaquette(&grid, &hop, 32);

    let nx = grid.nx;
    let ny = grid.ny;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut X: Vec<Vec<Complex64>> = Vec::with_capacity(args.m);
    let ang = |i: usize, j: usize| -> f64 {
        let x = grid.x0 + (i as f64) * grid.dx;
        let y = grid.y0 + (j as f64) * grid.dy;
        y.atan2(x)
    };
    {
        let mut v = vec![Complex64::new(0.0, 0.0); nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                if grid.mask[idx] {
                    v[idx] = Complex64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
                }
            }
        }
        X.push(v);
    }
    if args.m >= 2 {
        let mut v = vec![Complex64::new(0.0, 0.0); nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                if grid.mask[idx] {
                    let th = ang(i, j);
                    v[idx] = Complex64::from_polar(1.0, th)
                        * Complex64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
                }
            }
        }
        X.push(v);
    }
    if args.m >= 3 {
        let mut v = vec![Complex64::new(0.0, 0.0); nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                if grid.mask[idx] {
                    let th = ang(i, j);
                    v[idx] = Complex64::from_polar(1.0, -th)
                        * Complex64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
                }
            }
        }
        X.push(v);
    }
    while X.len() < args.m {
        let mut v = vec![Complex64::new(0.0, 0.0); nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                if grid.mask[idx] {
                    let th = ang(i, j);
                    let sgn = if (X.len() % 2) == 0 { 1.0 } else { -1.0 };
                    v[idx] = Complex64::from_polar(1.0, sgn * th)
                        * Complex64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
                }
            }
        }
        X.push(v);
    }

    if !args.deflate.is_empty() {
        let mut D: Vec<Vec<Complex64>> = Vec::new();
        for path in &args.deflate {
            let bytes = std::fs::read(path)?;
            let n = (bytes.len() / 16).max(1);
            let mut v = vec![Complex64::new(0.0, 0.0); n];
            for i in 0..n {
                let mut re = [0u8; 8];
                re.copy_from_slice(&bytes[16 * i..16 * i + 8]);
                let mut im = [0u8; 8];
                im.copy_from_slice(&bytes[16 * i + 8..16 * i + 16]);
                v[i] = Complex64::new(f64::from_le_bytes(re), f64::from_le_bytes(im));
            }
            let mut nrm = 0.0;
            for i in 0..v.len() {
                if grid.mask[i] {
                    nrm += grid.m_w[i] * (v[i].re * v[i].re + v[i].im * v[i].im);
                }
            }
            let nrm = nrm.sqrt().max(1e-30);
            for i in 0..v.len() {
                if grid.mask[i] {
                    v[i] /= Complex64::new(nrm, 0.0);
                } else {
                    v[i] = Complex64::new(0.0, 0.0);
                }
            }
            D.push(v);
        }
        project_block_b(&mut X, &D, &grid);
    }
    b_orthonormalize(&mut X, &grid);

    // workspaces
    let mut AX = vec![vec![Complex64::new(0.0, 0.0); nx * ny]; args.m];
    let mut MX = vec![vec![Complex64::new(0.0, 0.0); nx * ny]; args.m];
    let mut R = vec![vec![Complex64::new(0.0, 0.0); nx * ny]; args.m];
    let mut W = vec![vec![Complex64::new(0.0, 0.0); nx * ny]; args.m];

    let mut evals = vec![0.0f64; args.m];
    let mut evals_last = vec![0.0f64; args.m];
    let mut max_rel = f64::INFINITY;
    let mut last_checkpoint = 0usize;
    let mut locked = vec![false; args.m];
    let lock_thresh = args.tol.max(1e-9);

    for it in 0..args.max_it {
        for j in 0..args.m {
            hop.matvec(&X[j], &mut AX[j]);
            hop.apply_mass(&X[j], &mut MX[j]);
        }
        let T0 = gram_ax(&X, &AX, &grid);
        let B0 = gram_b(&X, &grid);
        let (_theta0, Y0) = sym_eig_small_gen(&T0, &B0);
        let mut Xrtz = mat_mul_small(&X, &Y0, &grid);
        b_orthonormalize(&mut Xrtz, &grid);
        X = Xrtz;

        for j in 0..args.m {
            hop.matvec(&X[j], &mut AX[j]);
            hop.apply_mass(&X[j], &mut MX[j]);
        }
        let T = gram_ax(&X, &AX, &grid);
        let B = gram_b(&X, &grid);
        let (lam, _Y) = sym_eig_small_gen(&T, &B);
        for j in 0..args.m {
            evals[j] = lam[j];
        }
        evals_last.clone_from_slice(&evals);

        max_rel = 0.0;
        for j in 0..args.m {
            for i in 0..X[j].len() {
                R[j][i] = AX[j][i] - Complex64::new(evals[j], 0.0) * MX[j][i];
            }
            let mut nr2 = 0.0;
            for i in 0..X[j].len() {
                if grid.mask[i] {
                    nr2 += R[j][i].norm_sqr();
                }
            }
            let rel = nr2.sqrt() / evals[j].abs().max(1.0);
            if rel > max_rel {
                max_rel = rel;
            }
            if it >= args.min_iters / 2 && rel <= lock_thresh {
                locked[j] = true;
            }
        }
        if locked.iter().all(|&b| b) {
            break;
        }

        if args.log_every == 0 || it % args.log_every == 0 {
            let l1 = evals.get(0).copied().unwrap_or(0.0);
            let l2 = evals.get(1).copied().unwrap_or(l1);
            let r = if l1 > 0.0 { l2 / l1 } else { 0.0 };
            println!(
                "iter {:6}  max_rel={:.3e}  λ1={:.8e}  λ2={:.8e}  ratio={:.6e}",
                it, max_rel, l1, l2, r
            );
        }
        if args.checkpoint_every > 0 && it % args.checkpoint_every == 0 && it != last_checkpoint {
            write_eigs_json(&args.out_prefix, it, max_rel, &evals).ok();
            last_checkpoint = it;
        }
        if it >= args.min_iters && max_rel <= args.tol {
            break;
        }

        for j in 0..args.m {
            if locked[j] {
                for i in 0..W[j].len() {
                    W[j][i] = Complex64::new(0.0, 0.0);
                }
                continue;
            }
            hop.apply_jacobi(&R[j], &mut W[j]);
            let omega = 0.5;
            for _ in 0..4 {
                hop.matvec(&W[j], &mut AX[j]);
                for i in 0..W[j].len() {
                    AX[j][i] = R[j][i] - AX[j][i];
                }
                hop.apply_jacobi(&AX[j], &mut MX[j]);
                for i in 0..W[j].len() {
                    W[j][i] += Complex64::new(omega, 0.0) * MX[j][i];
                }
            }
        }
        for j in 0..args.m {
            if locked[j] {
                continue;
            }
            let mut bn = 0.0;
            for i in 0..X[j].len() {
                if grid.mask[i] {
                    bn += grid.m_w[i] * (W[j][i].re * W[j][i].re + W[j][i].im * W[j][i].im);
                }
            }
            if bn.sqrt() < 1e-18 {
                for i in 0..W[j].len() {
                    if grid.mask[i] {
                        let ii = i % nx;
                        let jj = i / nx;
                        let x = grid.x0 + (ii as f64) * grid.dx;
                        let y = grid.y0 + (jj as f64) * grid.dy;
                        let th = y.atan2(x);
                        W[j][i] = Complex64::from_polar(1.0, if j % 2 == 0 { th } else { -th })
                            * Complex64::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
                    } else {
                        W[j][i] = Complex64::new(0.0, 0.0);
                    }
                }
            }
        }

        let mut S: Vec<Vec<Complex64>> = Vec::with_capacity(2 * args.m);
        for j in 0..args.m {
            S.push(X[j].clone());
        }
        for j in 0..args.m {
            S.push(W[j].clone());
        }
        b_orthonormalize(&mut S, &grid);

        let mut HS: Vec<Vec<Complex64>> =
            (0..S.len()).map(|_| vec![Complex64::new(0.0, 0.0); nx * ny]).collect();
        for j in 0..S.len() {
            hop.matvec(&S[j], &mut HS[j]);
        }
        let A_small = gram_ax(&S, &HS, &grid);
        let B_small = gram_b(&S, &grid);
        let (mu, Z) = sym_eig_small_gen(&A_small, &B_small);
        let k = args.m.min(mu.len());
        let Zm = Z.columns(0, k).into_owned();

        let mut Xnew = mat_mul_small(&S, &Zm, &grid);
        b_orthonormalize(&mut Xnew, &grid);
        X = Xnew;
    }

    for j in 0..args.m {
        hop.matvec(&X[j], &mut AX[j]);
        hop.apply_mass(&X[j], &mut MX[j]);
    }
    let Tfin = gram_ax(&X, &AX, &grid);
    let Bfin = gram_b(&X, &grid);
    let (theta_fin, _Yfin) = sym_eig_small_gen(&Tfin, &Bfin);
    let final_valid = theta_fin.len() >= 2 && theta_fin.iter().all(|v| v.is_finite() && *v > 0.0);
    if final_valid {
        let kfin = args.m.min(theta_fin.len());
        for j in 0..kfin {
            evals[j] = theta_fin[j];
        }
    } else {
        evals.clone_from_slice(&evals_last);
    }

    write_eigs_json(&args.out_prefix, args.max_it, max_rel, &evals).ok();

    let meta = RunMeta {
        nx: grid.nx,
        ny: grid.ny,
        r_in: grid.r_in,
        r_out: grid.r_out,
        eps_b: args.eps_b,
        eps_e: args.eps_e,
        flux: args.flux,
        tau: args.tau,
        m: args.m,
        max_it: args.max_it,
        tol: args.tol,
        out_prefix: args.out_prefix.clone(),
        u_from: args.u_from.clone(),
        u_scale: args.u_scale,
        u_floor: args.u_floor,
    };
    let ratio_mu = if let Some(path) = &args.lambda1_from {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let l1 = j
            .get("evals")
            .and_then(|a| a.as_array())
            .and_then(|a| a.get(0))
            .and_then(|e| e.get("lambda"))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        if l1 > 0.0 {
            Some(*evals.get(1).unwrap_or(&evals[0]) / l1)
        } else {
            None
        }
    } else {
        None
    };
    write_run_json(
        &args.out_prefix,
        &meta,
        max_rel,
        &evals,
        ratio_mu,
        "flux GEP run (E/M-matched)",
    )
    .ok();

    // keep the original message
    println!("Saved run log to {}-run.json", args.out_prefix);

    // === FULL EIGENVALUE REPORT ===
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&i, &j| {
        let a = evals[i];
        let b = evals[j];
        if a.is_nan() && b.is_nan() {
            Ordering::Equal
        } else if a.is_nan() {
            Ordering::Greater
        } else if b.is_nan() {
            Ordering::Less
        } else {
            a.partial_cmp(&b).unwrap()
        }
    });

    if !order.is_empty() {
        let l1_final = evals[order[0]];

        println!(
            "\n=== Eigenvalue Report (requested m = {}, found = {}) ===",
            args.m,
            evals.len()
        );
        println!("  λ1 = {:.12e}", l1_final);
        println!("------------------------------------------------------------");
        println!("#  rank  index  lambda (abs)         lambda/λ1");
        println!("------------------------------------------------------------");

        for (rank, &idx) in order.iter().enumerate() {
            let lam = evals[idx];
            let ratio = if l1_final != 0.0 { lam / l1_final } else { f64::NAN };
            println!(
                "{:>3} {:>5} {:>6}  {:>14.6e}  {:>14.6e}",
                rank,
                rank + 1,
                idx,
                lam,
                ratio
            );
        }
        println!("------------------------------------------------------------\n");
    }

    let l1 = evals.get(0).copied().unwrap_or(0.0);
    let l2 = evals.get(1).copied().unwrap_or(l1);
    let ratio = if l1 > 0.0 { l2 / l1 } else { 0.0 };

    if evals.len() >= 2 {
        println!("λ1={:.12e}  λ2={:.12e}  ratio={:.12e}", l1, l2, ratio);
    } else if !evals.is_empty() {
        println!("λ1={:.12e}", l1);
    } else {
        println!("(no eigenvalues returned)");
    }

    if let Some(r) = ratio_mu {
        println!("ratio_mu (vs ref) = {:.12}", r);
    }

    Ok((l1, ratio))
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    if let Some(kwant) = args.k {
        args.m = kwant.max(1);
    }

    if let Some(target_ratio) = args.target_ratio {
        let mut a_eps = args.search_eps_e_min.unwrap_or(0.01);
        let mut b_eps = args.search_eps_e_max.unwrap_or(1.0);
        let tol_eps = 1e-4;

        println!("Starting Cusp Tuner (Bisection)");
        println!("  Target Ratio: {:.6}", target_ratio);
        println!("  Search Range: [{}, {}]", a_eps, b_eps);
        println!("  Max Iterations: {}", args.tuner_max_iters);

        let r_a = solve_for_ratio(&args, a_eps)?.1;
        let r_b = solve_for_ratio(&args, b_eps)?.1;

        if r_a > target_ratio || r_b < target_ratio {
            println!("\n[ERROR] Initial bounding failed:");
            println!("  Ratio at eps_e={:.6} is {:.6}", a_eps, r_a);
            println!("  Ratio at eps_e={:.6} is {:.6}", b_eps, r_b);
            return Ok(());
        }

        let mut c_eps = (a_eps + b_eps) / 2.0;
        let mut best_eps_e = c_eps;
        let mut _final_ratio = 0.0;

        for k in 0..args.tuner_max_iters {
            c_eps = (a_eps + b_eps) / 2.0;
            let current_ratio = solve_for_ratio(&args, c_eps)?.1;
            _final_ratio = current_ratio;
            best_eps_e = c_eps;

            println!(
                "TUNER ITER {:2}: eps_e={:.6} -> Ratio={:.6} (Target={:.6})",
                k, c_eps, current_ratio, target_ratio
            );

            if (b_eps - a_eps).abs() < tol_eps {
                println!("\n[SUCCESS] Converged to eps_e={:.6} within tolerance.", c_eps);
                break;
            }

            if current_ratio < target_ratio {
                a_eps = c_eps;
            } else {
                b_eps = c_eps;
            }
        }

        println!("\n[FINAL RUN] Using tuned eps_e={:.12}", best_eps_e);
        let mut final_args = args.clone();
        final_args.eps_e = best_eps_e;
        let (l1, ratio) = final_solve_and_log(&final_args)?;
        println!(
            "\nTUNER RESULT: Achieved Ratio={:.12} for eps_e={:.12}",
            ratio, best_eps_e
        );
        println!("Final λ1={:.12e}, λ2={:.12e}", l1, l1 * ratio);
    } else {
        println!(
            "HB (flux) | grid {}x{}, r_in={}, r_out={}, eps_e={}",
            args.nx, args.ny, args.r_in, args.r_out, args.eps_e
        );
        final_solve_and_log(&args)?;
    }

    Ok(())
}
