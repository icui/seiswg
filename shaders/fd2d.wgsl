// =========================================================================
// seispie-wg  –  2-D finite-difference elastic / micropolar solver
//               running as WebGPU compute shaders
//
// Buffer layout
// ─────────────
//   binding 0  : Params (uniform)
//   binding 1  : fields  – packed f32 array, F_* slices of length npt each
//   binding 2  : idata   – i32 array  [src_id(nsrc) | rec_id(nrec)]
//   binding 3  : obs     – f32 array  5 × nrec×nt components (x,y,z,yi,yc)
//   binding 4  : stf     – f32 array  3 × nsrc×nt components (x,y,z)
//
// Field IDs (F_*)  – each occupies params.npt consecutive f32 values
// =========================================================================

// ─────────────────────────── params uniform ───────────────────────────────

struct Params {
    nx        : u32,   // grid points in x
    nz        : u32,   // grid points in z
    nt        : u32,   // time steps
    npt       : u32,   // total grid points = nx * nz

    dx        : f32,   // spatial step x  [m]
    dz        : f32,   // spatial step z  [m]
    dt        : f32,   // time step      [s]
    nsrc      : u32,   // number of sources

    nrec      : u32,   // number of receivers
    it        : u32,   // current time index (0 … nt-1)
    isrc      : i32,   // active source (-1 = all sources combined)
    sh        : u32,   // 1 → run SH component

    psv       : u32,   // 1 → run PSV component
    spin      : u32,   // 1 → include micropolar (spin) coupling
    abs_left  : u32,   // 1 → absorb on left edge
    abs_right : u32,   // 1 → absorb on right edge

    abs_bottom: u32,   // 1 → absorb on bottom edge
    abs_top   : u32,   // 1 → absorb on top edge
    abs_width : u32,   // width (grid pts) of absorbing layer
    abs_alpha : f32,   // decay coefficient

    // padding to 96 bytes (multiple of 16)
    _pad0     : u32,
    _pad1     : u32,
    _pad2     : u32,
    _pad3     : u32,
}

// ──────────────── field IDs (each slice = npt f32 values) ─────────────────

// SH wavefield
const F_VY     : u32 = 0u;
const F_UY     : u32 = 1u;
const F_SXY    : u32 = 2u;
const F_SZY    : u32 = 3u;
const F_DSY    : u32 = 4u;
const F_DVYDX  : u32 = 5u;
const F_DVYDZ  : u32 = 6u;

// PSV wavefield
const F_VX     : u32 = 7u;
const F_VZ     : u32 = 8u;
const F_UX     : u32 = 9u;
const F_UZ     : u32 = 10u;
const F_SXX    : u32 = 11u;
const F_SZZ    : u32 = 12u;
const F_SXZ    : u32 = 13u;
const F_DSX    : u32 = 14u;
const F_DSZ    : u32 = 15u;
const F_DVXDX  : u32 = 16u;
const F_DVXDZ  : u32 = 17u;
const F_DVZDX  : u32 = 18u;
const F_DVZDZ  : u32 = 19u;

// Micropolar (spin) coupling wavefield
const F_VY_C    : u32 = 20u;
const F_UY_C    : u32 = 21u;
const F_SYX_C   : u32 = 22u;
const F_SYY_C   : u32 = 23u;
const F_SYZ_C   : u32 = 24u;
const F_DSY_C   : u32 = 25u;
const F_DVYDX_C : u32 = 26u;
const F_DVYDZ_C : u32 = 27u;
const F_DUZDX   : u32 = 28u;
const F_DUXDZ   : u32 = 29u;

// Model parameters
const F_LAM   : u32 = 30u;
const F_MU    : u32 = 31u;
const F_NU    : u32 = 32u;
const F_J     : u32 = 33u;
const F_LAM_C : u32 = 34u;
const F_MU_C  : u32 = 35u;
const F_NU_C  : u32 = 36u;
const F_RHO   : u32 = 37u;
const F_BOUND : u32 = 38u;

// Adjoint sensitivity kernels
const F_K_LAM : u32 = 39u;
const F_K_MU  : u32 = 40u;
const F_K_RHO : u32 = 41u;
const F_GSUM  : u32 = 42u;
const F_GTMP  : u32 = 43u;

// N_FIELDS = 44

// ─────────── obs buffer – component offsets (each = nrec*nt f32s) ─────────
const OBS_X  : u32 = 0u;
const OBS_Y  : u32 = 1u;
const OBS_Z  : u32 = 2u;
const OBS_YI : u32 = 3u;   // spin: isolated rotation
const OBS_YC : u32 = 4u;   // spin: curl rotation

// ─────────── stf buffer – component offsets (each = nsrc*nt f32s) ─────────
const STF_X : u32 = 0u;
const STF_Y : u32 = 1u;
const STF_Z : u32 = 2u;

// ────────────────────────────── bindings ──────────────────────────────────

@group(0) @binding(0) var<uniform>            params : Params;
@group(0) @binding(1) var<storage, read_write> fields : array<f32>;
@group(0) @binding(2) var<storage, read>       idata  : array<i32>;
@group(0) @binding(3) var<storage, read_write> obs    : array<f32>;
@group(0) @binding(4) var<storage, read>       stf    : array<f32>;

// ────────────────────────────── helpers ───────────────────────────────────

// Read / write / accumulate into a named field slice.
fn gf(id: u32, k: u32) -> f32              { return fields[id * params.npt + k]; }
fn sf(id: u32, k: u32, v: f32)             { fields[id * params.npt + k] = v; }
fn af(id: u32, k: u32, v: f32)             { fields[id * params.npt + k] += v; }

fn src_id(s: u32) -> u32 { return u32(idata[s]); }
fn rec_id(r: u32) -> u32 { return u32(idata[params.nsrc + r]); }

// Decompose flat index k into (i, j) grid coordinates.
fn ij_from(k: u32) -> vec2<u32> {
    let nz = params.nz;
    let j = k % nz;
    let i = (k - j) / nz;
    return vec2<u32>(i, j);
}

// ── 4th-order staggered finite differences ──────────────────────────────

// Backward differences (centred on k):
//   ∂f/∂x ≈ [9(f_i - f_{i-1}) - (f_{i+1} - f_{i-2})] / (8 dx)
fn diff_x(fid: u32, i: u32, k: u32) -> f32 {
    if i < 2u || i >= params.nx - 2u { return 0.0; }
    let nz = params.nz;
    let dx = params.dx;
    return ( 9.0 * (gf(fid, k) - gf(fid, k - nz))
           - (gf(fid, k + nz) - gf(fid, k - 2u * nz)) ) / (8.0 * dx);
}

fn diff_z(fid: u32, j: u32, k: u32) -> f32 {
    if j < 2u || j >= params.nz - 2u { return 0.0; }
    let dz = params.dz;
    return ( 9.0 * (gf(fid, k) - gf(fid, k - 1u))
           - (gf(fid, k + 1u) - gf(fid, k - 2u)) ) / (8.0 * dz);
}

// Forward differences (centred on k+½):
fn diff_x1(fid: u32, i: u32, k: u32) -> f32 {
    if i < 1u || i >= params.nx - 2u { return 0.0; }
    let nz = params.nz;
    let dx = params.dx;
    return ( 9.0 * (gf(fid, k + nz) - gf(fid, k))
           - (gf(fid, k + 2u * nz) - gf(fid, k - nz)) ) / (8.0 * dx);
}

fn diff_z1(fid: u32, j: u32, k: u32) -> f32 {
    if j < 1u || j >= params.nz - 2u { return 0.0; }
    let dz = params.dz;
    return ( 9.0 * (gf(fid, k + 1u) - gf(fid, k))
           - (gf(fid, k + 2u) - gf(fid, k - 1u)) ) / (8.0 * dz);
}

// =========================================================================
//  Initialisation kernels
// =========================================================================

/// Convert (vp, vs) stored in (F_LAM, F_MU) to Lamé parameters (λ, µ).
/// Dispatch: ceil(npt / 64) workgroups × 64 threads.
@compute @workgroup_size(64)
fn vps2lm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.npt { return; }
    let vp  = gf(F_LAM, k);
    let vs  = gf(F_MU,  k);
    let rho = gf(F_RHO, k);
    sf(F_MU,  k, rho * vs * vs);
    if vp > vs {
        sf(F_LAM, k, rho * (vp * vp - 2.0 * vs * vs));
    } else {
        sf(F_LAM, k, 0.0);
    }
}

/// Build the absorbing-boundary Gaussian taper stored in F_BOUND.
/// Dispatch: ceil(npt / 64) workgroups × 64 threads.
@compute @workgroup_size(64)
fn set_bound(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.npt { return; }
    let ij    = ij_from(k);
    let i     = ij.x;
    let j     = ij.y;
    let nx    = params.nx;
    let nz    = params.nz;
    let w     = params.abs_width;
    let alpha = params.abs_alpha;
    var b     = 1.0f;

    if params.abs_left != 0u && (i + 1u) < w {
        let aw = alpha * f32(w - i - 1u);
        b *= exp(-aw * aw);
    }
    if params.abs_right != 0u && i > (nx - w) {
        let aw = alpha * f32(w + i - nx);
        b *= exp(-aw * aw);
    }
    if params.abs_bottom != 0u && j > (nz - w) {
        let aw = alpha * f32(w + j - nz);
        b *= exp(-aw * aw);
    }
    if params.abs_top != 0u && (j + 1u) < w {
        let aw = alpha * f32(w - j - 1u);
        b *= exp(-aw * aw);
    }
    sf(F_BOUND, k, b);
}

// =========================================================================
//  SH wavefield  (vy, uy, sxy, szy)
// =========================================================================

/// ∇·σ^SH → DSY    (divergence of SH stress)
@compute @workgroup_size(64)
fn div_sy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    if k >= params.npt { return; }
    let ij = ij_from(k);
    sf(F_DSY, k, diff_x(F_SXY, ij.x, k) + diff_z(F_SZY, ij.y, k));
}

/// Add source-time-function (STF_Y component) to DSY.
/// Dispatch: nsrc workgroups × 1 thread each.
@compute @workgroup_size(1)
fn stf_dsy(@builtin(workgroup_id) wgid: vec3<u32>) {
    let ib = wgid.x;
    if params.isrc >= 0i && i32(ib) != params.isrc { return; }
    let ks  = ib * params.nt + params.it;
    let km  = src_id(ib);
    let off = STF_Y * params.nsrc * params.nt;
    af(F_DSY, km, stf[off + ks]);
}

/// v_y ← bound·(v_y + dt·DSY/ρ),   u_y ← u_y + v_y·dt
@compute @workgroup_size(64)
fn add_vy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k   = gid.x;
    if k >= params.npt { return; }
    let dt     = params.dt;
    let vy_new = gf(F_BOUND, k) * (gf(F_VY, k) + dt * gf(F_DSY, k) / gf(F_RHO, k));
    sf(F_VY, k, vy_new);
    af(F_UY, k, vy_new * dt);
}

/// ∂v_y/∂x → DVYDX,   ∂v_y/∂z → DVYDZ   (staggered forward differences)
@compute @workgroup_size(64)
fn div_vy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    if k >= params.npt { return; }
    let ij = ij_from(k);
    sf(F_DVYDX, k, diff_x1(F_VY, ij.x, k));
    sf(F_DVYDZ, k, diff_z1(F_VY, ij.y, k));
}

/// σ_xy ← σ_xy + dt·µ·∂v_y/∂x,   σ_zy ← σ_zy + dt·µ·∂v_y/∂z
@compute @workgroup_size(64)
fn add_sy_sh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    if k >= params.npt { return; }
    let mu_dt = gf(F_MU, k) * params.dt;
    af(F_SXY, k, mu_dt * gf(F_DVYDX, k));
    af(F_SZY, k, mu_dt * gf(F_DVYDZ, k));
}

/// Record u_y at receivers → obs[OBS_Y] component.
/// Dispatch: nrec workgroups × 1 thread each.
@compute @workgroup_size(1)
fn save_obs_y_sh(@builtin(workgroup_id) wgid: vec3<u32>) {
    let ir  = wgid.x;
    let km  = rec_id(ir);
    let kr  = ir * params.nt + params.it;
    obs[OBS_Y * params.nrec * params.nt + kr] = gf(F_UY, km);
}

// =========================================================================
//  PSV wavefield  (vx, vz, ux, uz, sxx, szz, sxz)
// =========================================================================

/// ∇·σ^PSV → (DSX, DSZ)
@compute @workgroup_size(64)
fn div_sxz(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    if k >= params.npt { return; }
    let ij = ij_from(k);
    let i  = ij.x;
    let j  = ij.y;
    sf(F_DSX, k, diff_x(F_SXX, i, k) + diff_z(F_SXZ, j, k));
    sf(F_DSZ, k, diff_x(F_SXZ, i, k) + diff_z(F_SZZ, j, k));
}

/// Apply micropolar body-couple divergence: add ∂Σ_yy^c/∂z to DSX,
/// subtract ∂Σ_yy^c/∂x from DSZ, and add 2·Σ_yy^c to DSY_C.
@compute @workgroup_size(64)
fn div_sxyz_c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k   = gid.x;
    if k >= params.npt { return; }
    let ij  = ij_from(k);
    let syy = gf(F_SYY_C, k);
    af(F_DSX,   k,  diff_z(F_SYY_C, ij.y, k));
    af(F_DSZ,   k, -diff_x(F_SYY_C, ij.x, k));
    af(F_DSY_C, k, 2.0 * syy);
}

/// Divergence of micropolar shear stresses → DSY_C
@compute @workgroup_size(64)
fn div_sy_c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    if k >= params.npt { return; }
    let ij = ij_from(k);
    sf(F_DSY_C, k, diff_x(F_SYX_C, ij.x, k) + diff_z(F_SYZ_C, ij.y, k));
}

/// Add STF_X and STF_Z components to DSX and DSZ (PSV source injection).
/// Dispatch: nsrc workgroups × 1 thread each.
@compute @workgroup_size(1)
fn stf_dsxz(@builtin(workgroup_id) wgid: vec3<u32>) {
    let ib = wgid.x;
    if params.isrc >= 0i && i32(ib) != params.isrc { return; }
    let ks  = ib * params.nt + params.it;
    let km  = src_id(ib);
    let nst = params.nsrc * params.nt;
    af(F_DSX, km, stf[STF_X * nst + ks]);
    af(F_DSZ, km, stf[STF_Z * nst + ks]);
}

/// v_x, v_z velocity update + displacements u_x, u_z.
@compute @workgroup_size(64)
fn add_vxz(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k   = gid.x;
    if k >= params.npt { return; }
    let rho   = gf(F_RHO,   k);
    let bound = gf(F_BOUND, k);
    let dt    = params.dt;
    let vx    = bound * (gf(F_VX, k) + dt * gf(F_DSX, k) / rho);
    let vz    = bound * (gf(F_VZ, k) + dt * gf(F_DSZ, k) / rho);
    sf(F_VX, k, vx);
    sf(F_VZ, k, vz);
    af(F_UX, k, vx * dt);
    af(F_UZ, k, vz * dt);
}

/// Staggered velocity-gradient tensor for PSV.
@compute @workgroup_size(64)
fn div_vxz(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    if k >= params.npt { return; }
    let ij = ij_from(k);
    let i  = ij.x;
    let j  = ij.y;
    sf(F_DVXDX, k, diff_x1(F_VX, i, k));
    sf(F_DVXDZ, k, diff_z1(F_VX, j, k));
    sf(F_DVZDX, k, diff_x1(F_VZ, i, k));
    sf(F_DVZDZ, k, diff_z1(F_VZ, j, k));
}

/// PSV stress tensor update (Hooke's law).
@compute @workgroup_size(64)
fn add_sxz(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k      = gid.x;
    if k >= params.npt { return; }
    let lam    = gf(F_LAM,   k);
    let mu     = gf(F_MU,    k);
    let dt     = params.dt;
    let dvxdx  = gf(F_DVXDX, k);
    let dvxdz  = gf(F_DVXDZ, k);
    let dvzdx  = gf(F_DVZDX, k);
    let dvzdz  = gf(F_DVZDZ, k);
    af(F_SXX, k, dt * ((lam + 2.0*mu)*dvxdx + lam*dvzdz));
    af(F_SZZ, k, dt * ((lam + 2.0*mu)*dvzdz + lam*dvxdx));
    af(F_SXZ, k, dt * mu * (dvxdz + dvzdx));
}

// =========================================================================
//  Micropolar (spin) wavefield  (vy_c, uy_c, syx_c, syy_c, syz_c)
// =========================================================================

/// Micro-rotation velocity update using micro-inertia J.
@compute @workgroup_size(64)
fn add_vy_c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k     = gid.x;
    if k >= params.npt { return; }
    let bound = gf(F_BOUND, k);
    let dt    = params.dt;
    let vy_c  = bound * (gf(F_VY_C, k) + dt * gf(F_DSY_C, k) / gf(F_J, k));
    sf(F_VY_C, k, vy_c);
    af(F_UY_C, k, vy_c * dt);
}

/// ∂v_y^c/∂x → DVYDX_C,   ∂v_y^c/∂z → DVYDZ_C
@compute @workgroup_size(64)
fn div_vy_c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    if k >= params.npt { return; }
    let ij = ij_from(k);
    sf(F_DVYDX_C, k, diff_x1(F_VY_C, ij.x, k));
    sf(F_DVYDZ_C, k, diff_z1(F_VY_C, ij.y, k));
}

/// Micropolar couple-stress tensor update.
@compute @workgroup_size(64)
fn add_sy_c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k       = gid.x;
    if k >= params.npt { return; }
    let nu      = gf(F_NU,      k);
    let mu_c    = gf(F_MU_C,    k);
    let nu_c    = gf(F_NU_C,    k);
    let vy_c    = gf(F_VY_C,    k);
    let dvydx_c = gf(F_DVYDX_C, k);
    let dvydz_c = gf(F_DVYDZ_C, k);
    let dvxdz   = gf(F_DVXDZ,   k);
    let dvzdx   = gf(F_DVZDX,   k);
    let dt      = params.dt;
    // Σ_yy^c += 2ν·(-v_y^c - ½(∂v_z/∂x − ∂v_x/∂z))
    af(F_SYY_C, k, 2.0 * dt * nu * (-vy_c - 0.5 * (dvzdx - dvxdz)));
    af(F_SYX_C, k, dt * (mu_c + nu_c) * dvydx_c);
    af(F_SYZ_C, k, dt * (mu_c + nu_c) * dvydz_c);
}

/// Compute displacement gradient components needed for spin seismogram:
///   DUZDX ← ∂u_z/∂x,   DUXDZ ← ∂u_x/∂z
@compute @workgroup_size(64)
fn compute_du_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    if k >= params.npt { return; }
    let ij = ij_from(k);
    sf(F_DUZDX, k, diff_x1(F_UZ, ij.x, k));
    sf(F_DUXDZ, k, diff_z1(F_UX, ij.y, k));
}

// =========================================================================
//  Receiver-record kernels  (dispatch nrec workgroups × 1 thread each)
// =========================================================================

/// u_x → obs[OBS_X]
@compute @workgroup_size(1)
fn save_obs_x(@builtin(workgroup_id) wgid: vec3<u32>) {
    let ir = wgid.x;
    obs[OBS_X * params.nrec * params.nt + ir * params.nt + params.it]
        = gf(F_UX, rec_id(ir));
}

/// u_z → obs[OBS_Z]
@compute @workgroup_size(1)
fn save_obs_z(@builtin(workgroup_id) wgid: vec3<u32>) {
    let ir = wgid.x;
    obs[OBS_Z * params.nrec * params.nt + ir * params.nt + params.it]
        = gf(F_UZ, rec_id(ir));
}

/// u_y^c (micro-rotation) + isolated / curl decomposition → obs[OBS_Y / OBS_YI / OBS_YC].
/// Must be dispatched AFTER compute_du_grad.
@compute @workgroup_size(1)
fn save_obs_ry(@builtin(workgroup_id) wgid: vec3<u32>) {
    let ir    = wgid.x;
    let km    = rec_id(ir);
    let kt    = ir * params.nt + params.it;
    let nobs  = params.nrec * params.nt;
    let uy_c  = gf(F_UY_C, km);
    let curl  = -(gf(F_DUZDX, km) - gf(F_DUXDZ, km));
    let iso   = uy_c - 0.5 * curl;
    obs[OBS_Y  * nobs + kt] = uy_c;
    obs[OBS_YI * nobs + kt] = iso;
    obs[OBS_YC * nobs + kt] = curl;
}

// =========================================================================
//  Adjoint / sensitivity kernel accumulation
// =========================================================================

/// Cross-correlation imaging condition for µ:
///   k_µ += (∂v_y/∂x · ∂v_y^fw/∂x + ∂v_y/∂z · ∂v_y^fw/∂z) · Δt_adj
/// In the adjoint run, F_DVYDX_C / F_DVYDZ_C re-use the forward wavefield.
@compute @workgroup_size(64)
fn interaction_muy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k   = gid.x;
    if k >= params.npt { return; }
    // abs_alpha is repurposed by the adjoint runner as ndt = adj_interval * dt
    af(F_K_MU, k,
        (gf(F_DVYDX, k) * gf(F_DVYDX_C, k) + gf(F_DVYDZ, k) * gf(F_DVYDZ_C, k))
        * params.abs_alpha);
}

// =========================================================================
//  Gaussian smoothing (for kernel regularisation)
// =========================================================================

/// Horizontal pass: convolve K_MU rows with 1-D Gaussian, store in GTMP.
/// sigma is stored in abs_alpha (repurposed here; caller sets it before dispatch).
@compute @workgroup_size(64)
fn gaussian_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    if k >= params.npt { return; }
    let ij    = ij_from(k);
    let i     = ij.x;
    let j     = ij.y;
    let sigma = params.abs_alpha;   // sigma re-used via params
    let nx    = params.nx;
    var sumx = 0.0f;
    for (var n = 0u; n < nx; n++) {
        let d = f32(i32(i) - i32(n));
        let w = exp(-d * d / (2.0 * sigma * sigma));
        sumx += w * gf(F_K_MU, n * params.nz + j);
    }
    sf(F_GTMP, k, sumx);
}

/// Vertical pass: convolve GTMP columns with 1-D Gaussian, normalise by GSUM,
/// write result back to K_MU.
@compute @workgroup_size(64)
fn gaussian_z(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k   = gid.x;
    if k >= params.npt { return; }
    let ij  = ij_from(k);
    let i   = ij.x;
    let j   = ij.y;
    let sigma = params.abs_alpha;
    let nz  = params.nz;
    var sumz = 0.0f;
    for (var n = 0u; n < nz; n++) {
        let d = f32(i32(j) - i32(n));
        let w = exp(-d * d / (2.0 * sigma * sigma));
        sumz += w * gf(F_GTMP, i * nz + n);
    }
    let gsum = gf(F_GSUM, k);
    sf(F_K_MU, k, select(0.0, sumz / gsum, gsum > 1e-30));
}

// =========================================================================
//  Adjoint source injection and imaging-condition helpers
// =========================================================================

/// Inject adjoint STF (pre-loaded into obs[OBS_Y] slot) at receiver positions.
/// Call with dispatch_n(nrec) during the adjoint time loop.
@compute @workgroup_size(1)
fn adj_dsy(@builtin(workgroup_id) wgid: vec3<u32>) {
    let ir = wgid.x;
    let km = rec_id(ir);
    let kr = ir * params.nt + params.it;
    af(F_DSY, km, obs[OBS_Y * params.nrec * params.nt + kr]);
}

/// ∂(v_y)/∂x → DVYDX,  ∂(v_y)/∂z → DVYDZ  (adjoint velocity gradient)
@compute @workgroup_size(64)
fn div_uy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    if k >= params.npt { return; }
    let ij = ij_from(k);
    sf(F_DVYDX, k, diff_x1(F_VY, ij.x, k));
    sf(F_DVYDZ, k, diff_z1(F_VY, ij.y, k));
}

/// ∂(fw_vy stored in F_DSY)/∂x → DVYDX_C,  ∂/∂z → DVYDZ_C  (fw velocity gradients)
@compute @workgroup_size(64)
fn div_fw(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = gid.x;
    if k >= params.npt { return; }
    let ij = ij_from(k);
    sf(F_DVYDX_C, k, diff_x1(F_DSY, ij.x, k));
    sf(F_DVYDZ_C, k, diff_z1(F_DSY, ij.y, k));
}

/// Initialise F_GSUM for Gaussian normalisation.  sigma = params.abs_alpha.
/// Dispatch: ceil(npt / 64) workgroups.
@compute @workgroup_size(64)
fn init_gsum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k     = gid.x;
    if k >= params.npt { return; }
    let ij    = ij_from(k);
    let sigma = params.abs_alpha;
    var sumx  = 0.0f;
    for (var n = 0u; n < params.nx; n++) {
        let d = f32(i32(ij.x) - i32(n));
        sumx += exp(-d * d / (2.0 * sigma * sigma));
    }
    var sumz  = 0.0f;
    for (var n = 0u; n < params.nz; n++) {
        let d = f32(i32(ij.y) - i32(n));
        sumz += exp(-d * d / (2.0 * sigma * sigma));
    }
    sf(F_GSUM, k, sumx * sumz);
}
