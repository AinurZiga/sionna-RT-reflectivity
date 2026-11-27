"""Joint physical-contour EV quadrature (NumPy/SciPy, complex128).

Public API
----------
evaluate_ev(points, ...)
    Compute raw T_e, T_o, H_e, A_e for points[..., 4] containing UNSQUARED
    alpha, beta, chi, w. The identity is T_e = H_e + E*A_e/beta.
check_reference(result, reference_inner=functions2._inner, ...)
    Compare T_e and T_o to independently adaptive integration of the supplied
    reference kernel. The two components are scaled separately.
diagnostics(result)
    Summarize convergence and the phase-split identity.
load_ev_table(path, ...), EVInterpolator(path, ...)
    Load a generated five-field .npz and evaluate joint even/odd interpolation
    in TensorFlow with one S=0.1 switch. TensorFlow is an optional dependency
    imported only when constructing the interpolator.
generate_ev_table(...)
    Generate all five normalized fields and raw integrals, saving to ./data.
    Accept four explicit node arrays or automatic grids specified by counts.

H_e and A_e are RAW amplitudes: no lambda gate, r, or interpolation normalizer
is applied by evaluate_ev. Table generation and TF interpolation are separate
entry points below. Only the physical
contour is implemented. The previously tested domain is alpha,beta,chi in
[1e-6, 3], w in [0.1, 0.999999]; positive arguments outside it are accepted
but have not been validated uniformly.

Convergence is estimated by changing quadrature order. This is not a rigorous
error bound and does not estimate the tail beyond u_max (default 7), roundoff,
or an error shared by both orders. Tiny values near cancellations may reach
the double-precision error floor. Inspect component flags and reference checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import roots_legendre, wofz

__all__ = ["EVResult", "evaluate_ev", "check_reference", "diagnostics", "ev_phase",
           "EVInterpolator", "load_ev_table", "generate_ev_table"]
COMPONENTS = ("T_e", "T_o", "H_e", "A_e")
_D = np.exp(0.25j * np.pi)
_SQRT_PI = np.sqrt(np.pi)


def _points_array(points):
    if np.iscomplexobj(points):
        raise ValueError("points must contain real, unsquared parameters.")
    p = np.asarray(points, dtype=np.float64)
    if p.ndim < 1 or p.shape[-1] != 4:
        raise ValueError("Expected shape (..., 4), ordered alpha, beta, chi, w.")
    if not np.all(np.isfinite(p)):
        raise ValueError("All parameters must be finite.")
    if np.any(p[..., :3] <= 0) or np.any((p[..., 3] <= 0) | (p[..., 3] >= 1)):
        raise ValueError("Require alpha,beta,chi > 0 and 0 < w < 1.")
    return p


def ev_phase(points):
    """Return E for real unsquared points[..., 4]."""
    p = _points_array(points)
    _, b, c, w = np.moveaxis(p, -1, 0)
    q = (1 - w) * (1 + w)
    return np.exp((-2 * b * c * np.sqrt(q) + 1j * (b*b - q*c*c)) / (w*w))


@dataclass(frozen=True)
class EVResult:
    """Arrays preserve the batch shape of the input.

    values[..., :] is ordered T_e, T_o, H_e, A_e. error_estimate is the absolute
    change between the final two quadratures, independently for each component.
    Convergence flags and error_estimate are None when check=False.
    """

    points: np.ndarray
    values: np.ndarray
    n_used: np.ndarray
    error_estimate: Optional[np.ndarray]
    converged_components: Optional[np.ndarray]
    u_max: float

    @property
    def T_e(self):
        return self.values[..., 0]

    @property
    def T_o(self):
        return self.values[..., 1]

    @property
    def H_e(self):
        return self.values[..., 2]

    @property
    def A_e(self):
        return self.values[..., 3]

    @property
    def converged(self):
        if self.converged_components is None:
            return None
        return np.all(self.converged_components, axis=-1)


@lru_cache(maxsize=16)
def _legendre(n):
    x, weight = roots_legendre(n)
    t, weight = (x + 1) / 2, weight / 2
    t.setflags(write=False)
    weight.setflags(write=False)
    return t, weight


def _chunk(points, n_quad, u_max, reuse_alpha):
    """Fuse all four outputs and reuse W values across repeated beta,chi,w."""
    if reuse_alpha:
        triples, inverse = np.unique(points[:, 1:], axis=0, return_inverse=True)
        alpha, ai = np.unique(points[:, 0], return_inverse=True)
        za = (-1j * np.pi * wofz(-alpha * np.conj(_D)))[ai, None]
    else:
        triples = points[:, 1:]
        inverse = np.arange(len(points))
        za = -1j * np.pi * wofz(-points[:, 0, None] * np.conj(_D))

    b0, c0, w0 = triples.T[:, :, None]
    q0 = (1 - w0) * (1 + w0)
    k0 = w0 * _D / np.sqrt(q0)
    B0 = b0 / np.sqrt(q0)
    t, weights = _legendre(n_quad)
    scale = np.minimum(1.0, c0)
    span = np.arcsinh(u_max / scale)
    u = scale * np.sinh(span * t)
    jac = scale * span * np.cosh(span * t)
    tau0 = np.sqrt(c0*c0 - 1j * u*u)
    pm0, pp0 = (1j*tau0 - B0)/k0, (1j*tau0 + B0)/k0

    # These expensive calls do not depend on alpha.
    wm, wp, wmp = wofz(pm0), wofz(pp0), wofz(-pp0)
    direct_weight = 2*u/tau0 * jac * weights * np.exp(-u*u)
    # tau-chi is evaluated without subtracting close square roots.
    tau_minus_chi = -1j * u*u / (tau0 + c0)
    amplitude_weight = 2*u/tau0 * jac * weights * np.exp(
        -u*u/(w0*w0) - 2*b0*np.sqrt(q0)/(w0*w0)*tau_minus_chi
    )

    a, b, c, w = points.T[:, :, None]
    q = (1-w)*(1+w)
    rq = np.sqrt(q)
    kap = w*_D/rq
    B = b/rq
    pa = a*np.conj(_D)
    tau, pm, pp = tau0[inverse], pm0[inverse], pp0[inverse]
    wm, wp, wmp = wm[inverse], wp[inverse], wmp[inverse]
    dm, dp = pm*pm - pa*pa, pp*pp - pa*pa
    fm, fp, fmp = wm/dm, wp/dp, wmp/dp
    t1 = za/(2*pa*(pa-pm)*(pa-pp))
    t2 = za/(2*pa*(pa+pm)*(pa+pp))
    denom = 1j*kap*kap

    # Direct T_e is evaluated independently of the split reconstruction.
    ke = (t1+t2 - 0.5j*np.pi*kap/B*(fm+fmp))/denom
    # Cancel B analytically in the odd beta-pole terms.
    odd_alpha = (-2/kap) * (
        t1*pa*(1j*tau-kap*pa) - t2*pa*(1j*tau+kap*pa)
    )
    ko = (odd_alpha + 1j*np.pi*(pm*fm - pp*fmp))/denom

    delta = B/kap
    mid = 1j*tau/kap
    diff = (fp-fm)/(2*delta)
    near = abs(delta) < 1e-5*np.minimum(1, np.minimum(abs(mid-pa), abs(mid+pa)))
    if np.any(near):
        z = mid[near]
        p = np.broadcast_to(pa, mid.shape)[near]
        wz = wofz(z)
        den = z*z-p*p
        diff[near] = ((-2*z*wz + 2j/_SQRT_PI)/den - 2*z*wz/(den*den))
    kh = (t1+t2 + 1j*np.pi*diff)/denom
    # This is beta times the raw residue kernel: there is no division by beta.
    ka = -np.pi*rq/(kap*dp)
    coefficient = a*a*c*(c*c + b*b/q)/_SQRT_PI
    wd = direct_weight[inverse] * coefficient
    wa = amplitude_weight[inverse] * coefficient
    return np.stack([
        np.sum(wd*ke, axis=1), np.sum(wd*ko, axis=1),
        np.sum(wd*kh, axis=1), np.sum(wa*ka, axis=1),
    ], axis=-1)


def _fixed(points, n_quad, chunk_size, u_max, reuse_alpha):
    out = np.empty((len(points), 4), dtype=np.complex128)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            out[start:stop] = _chunk(points[start:stop], n_quad, u_max, reuse_alpha)
    return out


def evaluate_ev(points, *, n_quad=128, check=True, max_quad=512,
                rtol=1e-7, atol=0.0, chunk_size=256, u_max=7.0,
                reuse_alpha=True, on_failure="raise", contour="physical"):
    """Compute T_e,T_o,H_e,A_e for unsquared points[..., 4].

    With check=True, compare n_quad to twice that order, then increase only
    failed rows, up to max_quad. Each component must satisfy
        abs(I_new-I_old) <= atol + rtol*abs(I_new).
    rtol and atol may be scalars or length-four arrays in output order.
    The default atol=0 uses relative comparison for every component. A common
    absolute tolerance of 1e-12 would hide errors in many physically small odd
    values. Near zeros, use explicit component tolerances or inspect failures.

    on_failure='raise' rejects nonconverged/nonfinite rows. 'return' keeps the
    computed values and flags for inspection; it does not mark them successful.
    With check=False, only finiteness is enforced and flags are None.

    Repeated beta,chi,w triples share W evaluations within each chunk when
    reuse_alpha=True. For grid generation, place all alpha values for a triple
    next to one another. General batches are supported as well.
    """
    if contour != "physical":
        raise ValueError("Only contour='physical' is implemented.")
    for name, value in [("n_quad", n_quad), ("max_quad", max_quad), ("chunk_size", chunk_size)]:
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer.")
    if n_quad < 8 or chunk_size < 1 or (check and max_quad <= n_quad):
        raise ValueError("Require n_quad >= 8, chunk_size >= 1, and max_quad > n_quad when checking.")
    if not np.isfinite(u_max) or u_max <= 0:
        raise ValueError("u_max must be positive and finite.")
    if on_failure not in ("raise", "return"):
        raise ValueError("on_failure must be 'raise' or 'return'.")
    rt, at = np.broadcast_to(np.asarray(rtol, float), (4,)), np.broadcast_to(np.asarray(atol, float), (4,))
    if not np.all(np.isfinite(rt)) or not np.all(np.isfinite(at)) or np.any(rt < 0) or np.any(at < 0):
        raise ValueError("Tolerances must be finite and nonnegative.")
    p = _points_array(points)
    shape = p.shape[:-1]
    flat = p.reshape(-1, 4)
    values = _fixed(flat, n_quad, chunk_size, u_max, reuse_alpha)
    used = np.full(len(flat), n_quad, dtype=np.int32)
    error = flags = None
    if check:
        error = np.full((len(flat), 4), np.inf)
        flags = np.zeros((len(flat), 4), dtype=bool)
        pending = np.arange(len(flat))
        order = n_quad
        while len(pending) and order < max_quad:
            order = min(2*order, max_quad)
            newer = _fixed(flat[pending], order, chunk_size, u_max, reuse_alpha)
            change = abs(newer-values[pending])
            finite = np.isfinite(newer) & np.isfinite(values[pending])
            ok = finite & (change <= at + rt*abs(newer))
            error[pending], flags[pending] = change, ok
            values[pending], used[pending] = newer, order
            pending = pending[~np.all(ok, axis=1)]
        failed = ~np.all(flags, axis=1)
    else:
        failed = ~np.all(np.isfinite(values), axis=1)
    if np.any(failed) and on_failure == "raise":
        bad = np.flatnonzero(failed)
        raise RuntimeError(
            f"EV quadrature failed for {len(bad)} rows; first flat indices: {bad[:8].tolist()}. "
            "Use on_failure='return' to inspect flags, increase max_quad, or check the reference."
        )
    return EVResult(p.copy(), values.reshape(shape+(4,)), used.reshape(shape),
                    None if error is None else error.reshape(shape+(4,)),
                    None if flags is None else flags.reshape(shape+(4,)), float(u_max))


def diagnostics(result):
    """Return component convergence and two differently scaled identity errors."""
    p = result.points
    residue = ev_phase(p)*result.A_e/p[..., 1]
    residual = abs(result.T_e-(result.H_e+residue))
    tiny = np.finfo(float).tiny
    relative = residual/np.maximum(abs(result.T_e), tiny)
    conditioned = residual/np.maximum(abs(result.H_e)+abs(residue)+abs(result.T_e), tiny)
    finite = np.all(np.isfinite(result.values), axis=-1)
    return {
        "components": COMPONENTS,
        "finite": finite,
        "converged": result.converged,
        "converged_components": result.converged_components,
        "n_used": result.n_used,
        "error_estimate": result.error_estimate,
        "identity_relative_error": relative,
        "identity_scaled_error": conditioned,
    }


def check_reference(result, *, reference_inner: Callable, max_points=16,
                    rtol=1e-10, atol=1e-11):
    """Adaptive spot checks using the supplied functions2._inner callback.

    The callback signature is (tau, alpha_squared, beta_squared, w, contour)
    and it must return the reference EVEN and ODD inner kernels, in that order.
    Its own pole branches are preserved with contour='physical'. H_e and A_e
    are checked by order refinement and the split identity, not this callback.

    atol applies after independently scaling the even and odd components.
    The comparison shares the finite cutoff u_max and is not a tail check.
    This routine is intentionally scalar/adaptive and is for spot checks only.
    """
    p = result.points.reshape(-1, 4)
    values = result.values.reshape(-1, 4)[:, :2]
    if max_points is not None and (not isinstance(max_points, (int, np.integer)) or max_points < 1):
        raise ValueError("max_points must be None or a positive integer.")
    count = len(p) if max_points is None else min(len(p), max_points)
    indices = np.unique(np.linspace(0, len(p)-1, count, dtype=int)) if count else np.array([], dtype=int)
    reference = np.empty((len(indices), 2), dtype=np.complex128)
    quad_error = np.empty(len(indices))
    for j, ind in enumerate(indices):
        a, b, c, w = p[ind]
        if not np.all(np.isfinite(values[ind])):
            raise ValueError(f"Cannot scale a nonfinite result at flat index {ind}.")
        q = (1-w)*(1+w)
        kap = w*_D/np.sqrt(q)
        ce = a*a*c*(c*c+b*b/q)/_SQRT_PI
        coefficient = np.array([ce, -2*ce/kap])
        scale = np.maximum(abs(values[ind]), np.finfo(float).tiny)

        def integrand(u):
            tau = np.sqrt(c*c-1j*u*u)
            return (coefficient/scale * 2*u/tau * np.exp(-u*u)
                    * reference_inner(tau, a*a, b*b, w, "physical"))

        breaks = sorted({x for x in [c, 2*c, 4*c, 1.0] if 0 < x < result.u_max})
        val, err = quad_vec(integrand, 0, result.u_max, points=breaks,
                            epsabs=atol, epsrel=rtol, limit=400)
        reference[j], quad_error[j] = val*scale, err
    absolute = abs(values[indices]-reference)
    relative = absolute/np.maximum(abs(reference), np.finfo(float).tiny)
    return {"flat_indices": indices, "points": p[indices], "reference": reference,
            "absolute_error": absolute, "relative_error": relative,
            "adaptive_scaled_error_estimate": quad_error}


class EVInterpolator:
    """Load a generated .npz once; evaluate jointly in TensorFlow.

    Usage: table = EVInterpolator('./data/ev_a5_b5_c5_w5.npz')
           T_e, T_o = table(points)  # points[..., 4], unsquared parameters

    Outputs are complex64 tensors with shape points.shape[:-1]. Inputs are
    converted to float32. Bounds are checked at float32 resolution. A scalar
    query has shape (4,); arbitrary batch shapes and empty batches are allowed.
    Constants are retained on the requested device (or TF's default device).
    No Python/SciPy callbacks occur during graph evaluation. The integrator
    remains usable without installing TensorFlow: TF is imported only here.

    One common S=0.1 switch selects either two low-S fields or three phase
    fields. Physical-coordinate weights are shared. r is explicit; lambda is
    already included in the tables. The known switch discontinuities and odd
    interpolation errors are not repaired by this implementation.

    Unconverged source nodes are rejected unless explicitly allowed. Allowing
    them is useful for debugging a table, not a declaration of accuracy.
    """

    def __init__(self, path, *, allow_unconverged=False, device=None):
        import itertools
        import json
        from contextlib import nullcontext
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow is required only for EVInterpolator.") from exc

        self._tf = tf
        self.path = str(path)
        keys = ("alpha_nodes", "beta_nodes", "chi_nodes", "w_nodes")
        field_names = ("R_e_T", "R_e_H", "R_A", "R_o_0", "R_o_B")
        with np.load(path, allow_pickle=False) as f:
            missing = set(keys + field_names + ("metadata", "converged_components")) - set(f.files)
            if missing:
                raise ValueError(f"Missing table entries: {sorted(missing)}")
            metadata = json.loads(str(f["metadata"].item()))
            if metadata.get("formula_version") != "joint_shared_A_v1":
                raise ValueError("Unsupported normalization/formula version.")
            if metadata.get("contour") != "physical":
                raise ValueError("Expected physical-contour data.")
            if float(metadata.get("S_star", -1)) != 0.1:
                raise ValueError("This implementation requires the common S_star=0.1.")
            if metadata.get("axis_order") != ["alpha", "beta", "chi", "w"]:
                raise ValueError("Unsupported axis order.")
            if float(metadata.get("beta_c_factor", 0.999)) != 0.999:
                raise ValueError("Unsupported r coefficient.")
            if metadata.get("odd_normalization", "geometry_2/residual2") != "geometry_2/residual2":
                raise ValueError("Unsupported odd residual normalization.")
            axes64 = [np.asarray(f[k], dtype=np.float64) for k in keys]
            for j, axis in enumerate(axes64):
                if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
                    raise ValueError(f"Invalid axis {keys[j]}.")
                if np.any(axis <= 0) or np.any(np.diff(axis) <= 0) or (j == 3 and np.any(axis >= 1)):
                    raise ValueError(f"Axis {keys[j]} must be positive and strictly increasing; w < 1.")
            self.grid_shape = tuple(len(a) for a in axes64)
            if np.prod(self.grid_shape, dtype=np.int64) > np.iinfo(np.int32).max:
                raise ValueError("Table is too large for int32 flat indices.")
            flags = np.asarray(f["converged_components"])
            if flags.dtype != np.bool_ or flags.shape != self.grid_shape + (4,):
                raise ValueError("Invalid per-component convergence flags.")
            self.unconverged_nodes = int(np.count_nonzero(~np.all(flags, axis=-1)))
            if self.unconverged_nodes and not allow_unconverged:
                raise ValueError(
                    f"Table has {self.unconverged_nodes} unconverged nodes. "
                    "Inspect them or use allow_unconverged=True explicitly for debugging."
                )
            fields = {}
            for k in field_names:
                value = np.asarray(f[k])
                if value.shape != self.grid_shape or not np.iscomplexobj(value):
                    raise ValueError(f"Invalid shape/type for {k}.")
                with np.errstate(over="ignore", invalid="ignore"):
                    value = value.astype(np.complex64)
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite complex64 values in {k}.")
                fields[k] = value

        axes = [a.astype(np.float32) for a in axes64]
        for j, axis in enumerate(axes):
            if (np.any(np.diff(axis) <= 0) or not np.isfinite(axis).all()
                    or axis[0] <= 0 or (j == 3 and axis[-1] >= 1)):
                raise ValueError("Axis nodes collapse, overflow, or reach a singular boundary in float32.")
        self.metadata = metadata
        self.axes_numpy = tuple(a.copy() for a in axes)
        scope = tf.device(device) if device is not None else nullcontext()
        with scope:
            self._axes = tuple(tf.constant(a) for a in axes)
            self._invgaps = tuple(tf.constant(1/np.diff(a)) for a in axes)
            self._lower = tf.constant([a[0] for a in axes], tf.float32)
            self._upper = tf.constant([a[-1] for a in axes], tf.float32)
            na, nb, nc, nw = self.grid_shape
            self._strides = tf.constant([nb*nc*nw, nc*nw, nw, 1], tf.int32)
            self._bits = tf.constant(list(itertools.product([0, 1], repeat=4)), tf.int32)
            self._low_fields = tf.constant(np.stack([
                fields["R_e_T"], fields["R_o_0"]
            ], axis=-1).reshape(-1, 2))
            self._phase_fields = tf.constant(np.stack([
                fields["R_e_H"], fields["R_A"], fields["R_o_B"]
            ], axis=-1).reshape(-1, 3))
            c1 = _SQRT_PI * _D
            self._c1 = tf.constant(c1, tf.complex64)
            self._b1 = tf.constant(c1/(4-np.pi), tf.complex64)
            self._c2 = tf.constant(1j*(3*np.pi-8)/(4-np.pi), tf.complex64)
            self._c3 = tf.constant(2j*c1*(np.pi-3)/(4-np.pi), tf.complex64)
            self._d = tf.constant(_D, tf.complex64)
        self._compiled = tf.function(
            self._evaluate_flat, input_signature=[tf.TensorSpec([None, 4], tf.float32)],
            autograph=False,
        )

    def _z(self, x):
        return self._tf.cast(x, self._tf.complex64)

    def _den(self, x):
        z = self._z(x)
        return 1+z*(self._b1+z*(self._c2+z*self._c3))

    def _phi(self, x):
        z = self._z(x)
        return z*(self._c1+z*(self._c2+z*self._c3))/self._den(x)

    def _sigma(self, x):
        tf = self._tf
        return tf.complex(tf.zeros_like(x), 2*x*x)*(1+(self._b1-self._c1)*self._z(x))/self._den(x)

    def _psi(self, x, y):
        tf = self._tf
        rho = x*x/(x*x+y*y)
        u = tf.complex(tf.zeros_like(x), 2*x*(x*x+y*y)/y*tf.atan(y/x))
        v = tf.complex(tf.zeros_like(x), 2*x*x)
        return u/(self._z(1-rho)+(u-self._z(rho)*v)/self._phi(x)
                  +self._z(rho)*v/self._sigma(x))

    def _q_functions(self, a, b):
        t, u = self._z(a+b), self._z(a*b)
        c1, b1, c2, c3 = self._c1, self._b1, self._c2, self._c3
        pe = (c1*(1+t*(b1+t*(c2+t*c3))) + u*(c2*b1-c3-c1*c2)
              +u*t*(c2*c2-2*c1*c3)+c2*c3*u*t*t+c3*c3*u*u*t)
        dd = self._den(a)*self._den(b)
        qe = u*pe/(t*dd)
        po = c1+c2*t+c3*(t*t-u)-(c1-b1)*u*(c2+c3*t)
        tf = self._tf
        qo = tf.complex(tf.zeros_like(a), 2*a*a*b*b/(a+b))*po/dd
        return qe, qo

    def _cells(self, p):
        tf = self._tf
        indices, fractions = [], []
        for j, axis in enumerate(self._axes):
            last = self.grid_shape[j]-2
            # Binary search supports arbitrary strictly increasing node arrays.
            i = tf.searchsorted(axis, p[:, j], side="right", out_type=tf.int32)-1
            i = tf.clip_by_value(i, 0, last)
            t = (p[:, j]-tf.gather(axis, i))*tf.gather(self._invgaps[j], i)
            indices.append(i)
            fractions.append(tf.clip_by_value(t, 0.0, 1.0))
        idx = tf.stack(indices, axis=-1)
        t = tf.stack(fractions, axis=-1)
        corners = idx[:, None, :]+self._bits[None, :, :]
        flat = tf.reduce_sum(corners*self._strides, axis=-1)
        wt = tf.reduce_prod(tf.where(
            tf.cast(self._bits[None, :, :], tf.bool), t[:, None, :], 1-t[:, None, :]
        ), axis=-1)
        return flat, wt

    def _interpolate(self, fields, ids, weights):
        tf = self._tf
        return tf.reduce_sum(tf.gather(fields, ids)*self._z(weights[:, :, None]), axis=1)

    def _low(self, p, ids, weights):
        tf = self._tf
        a, b, c, w = tf.unstack(p, axis=1)
        q = (1-w)*(1+w)
        rq = tf.sqrt(q)
        qe, qo = self._q_functions(a, b)
        pc = self._phi(c)
        factor = 1+q*c*c/(b*b)
        pg = self._phi(a)*self._psi(c, b/rq)*self._z(
            tf.atan((b+a*w)/(c*rq))/tf.atan(b/(c*rq))
        )
        eta = a*a/(a*a+q)
        nt = self._z(1-eta)*pg+self._z(eta*factor)*pc*qe
        m = pc*qo*self._z(factor)*self._phi((a+b)/rq)
        scale = a/rq+b/(w*rq)+c/w
        theta = tf.atan((q*c-w*b-w*w*a)/(rq*(b+w*a+w*c)))
        j1 = self._d*self._z(np.pi**1.5*w/q*(a+b)-2*_SQRT_PI*c/w)
        g0 = self._z(-2*theta/(np.pi*w))
        g1 = self._z(-2/(np.pi*w))*(
            self._z(rq/np.pi)*j1+self._d*self._z(
                2*theta/_SQRT_PI*(c+a+b+(a+b)/rq)
            )
        )
        base = m*(g0+g1/self._z(1+scale*scale))
        n0 = m*self._z(scale*scale/(1+scale*scale))
        ip = self._interpolate(self._low_fields, ids, weights)
        return tf.stack([nt*ip[:, 0], base+n0*ip[:, 1]], axis=-1)

    def _phase(self, p, ids, weights):
        tf = self._tf
        a, b, c, w = tf.unstack(p, axis=1)
        q = (1-w)*(1+w)
        rq = tf.sqrt(q)
        common = self._psi(c, (b+a*w)/rq)
        nh, nb = self._phi(a)*common, self._sigma(a)*common
        x, y = c/w, a/rq
        na = self._c1*self._z(a*a*c*c*c/(x*(x*x+y*y)))*self._psi(x, y)
        e = tf.exp(tf.complex(-2*b*c*rq/(w*w), (b*b-q*c*c)/(w*w)))
        v = w*w*q/(2*(q+a*a))
        ar = tf.sqrt((tf.sqrt(c*c*c*c+v*v)+c*c)/2)
        br = v/(2*ar)
        bc = tf.minimum(b, 0.999*c*rq)
        r = tf.complex(2*rq*ar/(w*w), -(2*rq*br+2*bc)/(w*w))
        ip = self._interpolate(self._phase_fields, ids, weights)
        amp = na*ip[:, 1]
        return tf.stack([nh*ip[:, 0]+e*amp/self._z(b),
                         nb*ip[:, 2]+e*r*amp], axis=-1)

    def _evaluate_flat(self, p):
        tf = self._tf
        p = tf.debugging.check_numerics(p, "EV parameters must be finite")
        checks = [tf.debugging.assert_greater_equal(p, self._lower, message="Below EV table bounds"),
                  tf.debugging.assert_less_equal(p, self._upper, message="Above EV table bounds")]
        with tf.control_dependencies(checks):
            p = tf.identity(p)
        ids, wt = self._cells(p)
        s = p[:, 2]**2*(1-p[:, 3])*(1+p[:, 3])/p[:, 3]**2
        low_mask = s < tf.constant(0.1, tf.float32)
        ilow = tf.cast(tf.where(low_mask)[:, 0], tf.int32)
        ihigh = tf.cast(tf.where(~low_mask)[:, 0], tf.int32)

        def branch(indices, fn):
            return tf.cond(tf.size(indices) > 0, lambda: fn(
                tf.gather(p, indices), tf.gather(ids, indices), tf.gather(wt, indices)
            ), lambda: tf.zeros([0, 2], tf.complex64))

        low = branch(ilow, self._low)
        high = branch(ihigh, self._phase)
        indices = tf.concat([ilow, ihigh], axis=0)[:, None]
        values = tf.concat([low, high], axis=0)
        return tf.scatter_nd(indices, values, tf.stack([tf.shape(p)[0], 2]))

    def __call__(self, points):
        tf = self._tf
        p = tf.convert_to_tensor(points)
        if p.dtype.is_complex or not (p.dtype.is_floating or p.dtype.is_integer):
            raise ValueError("Expected real, unsquared EV parameters.")
        p = tf.cast(p, tf.float32)
        checks = [tf.debugging.assert_rank_at_least(p, 1),
                  tf.debugging.assert_equal(tf.shape(p)[-1], 4, message="Expected points[..., 4]")]
        with tf.control_dependencies(checks):
            shape = tf.shape(p)[:-1]
            out = self._compiled(tf.reshape(p, [-1, 4]))
        return tf.reshape(out[:, 0], shape), tf.reshape(out[:, 1], shape)

    def evaluate(self, points):
        """Alias for table(points). Safe to call inside an outer tf.function."""
        return self(points)


def load_ev_table(path, *, allow_unconverged=False, device=None):
    """Create an EVInterpolator once; subsequent calls use its retained tensors."""
    return EVInterpolator(path, allow_unconverged=allow_unconverged, device=device)


from pathlib import Path
import json
import time
import numpy as np


def _ev_fields(p, raw):
    # All algebra is evaluated in complex128 before casting table fields.
    a, b, c, w = p.T
    q = (1-w)*(1+w)
    rq = np.sqrt(q)
    d = np.exp(0.25j*np.pi)
    c1 = np.sqrt(np.pi)*d
    b1 = c1/(4-np.pi)
    c2 = 1j*(3*np.pi-8)/(4-np.pi)
    c3 = 2j*c1*(np.pi-3)/(4-np.pi)

    def den(x):
        return 1+x*(b1+x*(c2+x*c3))

    def phi(x):
        return x*(c1+x*(c2+x*c3))/den(x)

    def sigma(x):
        return 2j*x*x*(1+(b1-c1)*x)/den(x)

    def psi(x, y):
        rho = x*x/(x*x+y*y)
        u = 2j*x*(x*x+y*y)/y*np.arctan(y/x)
        v = 2j*x*x
        return u/((1-rho)+(u-rho*v)/phi(x)+rho*v/sigma(x))

    t, u = a+b, a*b
    pe = (c1*(1+t*(b1+t*(c2+t*c3)))
          + u*(c2*b1-c3-c1*c2) + u*t*(c2*c2-2*c1*c3)
          + c2*c3*u*t*t + c3*c3*u*u*t)
    qe = u*pe/(t*den(a)*den(b))
    po = c1+c2*t+c3*(t*t-u)-(c1-b1)*u*(c2+c3*t)
    qo = 2j*a*a*b*b*po/(t*den(a)*den(b))

    common = psi(c, (b+a*w)/rq)
    nh = phi(a)*common
    nb = sigma(a)*common
    x, y = c/w, a/rq
    na = np.sqrt(np.pi)*d*a*a*c**3*psi(x, y)/(x*(x*x+y*y))
    pg = (phi(a)*psi(c, b/rq)
          * np.arctan((b+a*w)/(c*rq))/np.arctan(b/(c*rq)))
    eta = a*a/(a*a+q)
    nt = (1-eta)*pg+eta*phi(c)*qe*(1+q*c*c/(b*b))

    m = phi(c)*qo*(1+q*c*c/(b*b))*phi((a+b)/rq)
    zeta = a/rq+b/(w*rq)+c/w
    theta = np.arctan((q*c-w*b-w*w*a)/(rq*(b+w*a+w*c)))
    j1 = d*(np.pi**1.5*w/q*(a+b)-2*np.sqrt(np.pi)*c/w)
    g0 = -2*theta/(np.pi*w)
    g1 = -2/(np.pi*w)*(rq*j1/np.pi
         + 2*d*theta/np.sqrt(np.pi)*(c+a+b+(a+b)/rq))
    baseline = m*(g0+g1/(1+zeta*zeta))
    n0 = m*zeta*zeta/(1+zeta*zeta)

    z = np.clip(2*b/(c*rq)-1, 0, 1)
    lam = 1-z**3*(10-15*z+6*z*z)
    bc = np.minimum(b, 0.999*c*rq)
    r = 2*rq/(w*w)*(np.sqrt(c*c-0.5j*w*w/(1+a*a/q))-1j*bc/rq)
    e = ev_phase(p)
    te, to, he, ae = raw.T
    shared_a = lam*ae
    h = he+(1-lam)*e/b*ae
    background = to-e*r*shared_a
    return np.column_stack([
        te/nt, h/nh, shared_a/na,
        (to-baseline)/n0, background/nb,
    ])


def generate_ev_table(
    n_alpha=5, n_beta=5, n_chi=5, n_w=5, *,
    alpha_nodes=None, beta_nodes=None, chi_nodes=None, w_nodes=None,
    alpha_range=(1e-6, 3.0), beta_range=(1e-6, 3.0),
    chi_range=(1e-6, 3.0), w_range=(0.1, 0.999999),
    n_quad=128, max_quad=1024, rtol=1e-6, atol=0.0,
    batch_size=256, data_dir='./data', filename=None, overwrite=False,
):
    """Generate a tensor-product grid and save a complete .npz archive.

    Explicit *_nodes arrays override the corresponding count and range. Axes
    can be mixed: e.g. automatic alpha,beta,chi and manual w. Without arrays,
    use geometric alpha,beta,chi and a geometric-q w grid. Each axis needs at
    least two strictly increasing, positive nodes; w must be below one. Nodes
    must also remain distinct and nonsingular after conversion to float32.

    Save raw complex128 values, five complex64 normalized fields, exact axes,
    and quadrature flags. Finite but unconverged nodes are retained and reported;
    the loader rejects them by default. Interpolation accuracy is not checked
    here. Saving occurs after the entire calculation; interrupted generation
    has no resume checkpoint. Existing files require overwrite=True.
    """
    counts_requested = (n_alpha, n_beta, n_chi, n_w)
    supplied = (alpha_nodes, beta_nodes, chi_nodes, w_nodes)
    bounds = (alpha_range, beta_range, chi_range, w_range)
    axes = []
    for j, (nodes, n, interval) in enumerate(zip(supplied, counts_requested, bounds)):
        if nodes is None:
            if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 2:
                raise ValueError('Each automatic axis needs an integer node count >= 2.')
            lo, hi = interval
            if not np.isfinite([lo, hi]).all() or not 0 < lo < hi or (j == 3 and hi >= 1):
                raise ValueError('Require positive increasing bounds, and w_max < 1.')
            if j < 3:
                axis = np.geomspace(lo, hi, n)
            else:
                logq = np.linspace(np.log((1-lo)*(1+lo)), np.log((1-hi)*(1+hi)), n)
                axis = np.sqrt(-np.expm1(logq))
                axis[0], axis[-1] = lo, hi
        else:
            if np.iscomplexobj(nodes):
                raise ValueError('Node arrays must be real.')
            axis = np.array(nodes, dtype=np.float64, copy=True)
        if (axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
                or np.any(axis <= 0) or np.any(np.diff(axis) <= 0)
                or (j == 3 and np.any(axis >= 1))):
            raise ValueError(f'Invalid node array for axis {j}: require strictly increasing positive nodes; w < 1.')
        with np.errstate(over='ignore', under='ignore'):
            axis32 = axis.astype(np.float32)
        if (not np.isfinite(axis32).all() or axis32[0] <= 0
                or np.any(np.diff(axis32) <= 0) or (j == 3 and axis32[-1] >= 1)):
            raise ValueError(f'Axis {j} collapses or reaches a singular boundary in float32.')
        axes.append(axis)
    counts = tuple(len(axis) for axis in axes)
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer.')
    folder = Path(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    na, nb, nc, nw = counts
    if filename is None:
        filename = f'ev_a{na}_b{nb}_c{nc}_w{nw}.npz'
    if Path(filename).name != str(filename) or Path(filename).suffix != '.npz':
        raise ValueError('filename must be a basename ending in .npz; use data_dir for its folder.')
    path = folder/filename
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} exists; choose another data_dir or set overwrite=True.')

    total = int(np.prod(counts))
    raw = np.empty((total, 4), np.complex128)
    fields = np.empty((total, 5), np.complex64)
    converged = np.empty((total, 4), bool)
    n_used = np.empty(total, np.int32)
    quad_change = np.empty((total, 4), np.float64)
    started = time.perf_counter()
    # Alpha varies fastest during calculation, allowing W-value reuse.
    work_shape = (nb, nc, nw, na)
    for start in range(0, total, batch_size):
        stop = min(start+batch_size, total)
        ib, ic, iw, ia = np.unravel_index(np.arange(start, stop), work_shape)
        target = np.ravel_multi_index((ia, ib, ic, iw), counts)
        p = np.column_stack([axes[0][ia], axes[1][ib], axes[2][ic], axes[3][iw]])
        res = evaluate_ev(p, n_quad=n_quad, max_quad=max_quad, rtol=rtol,
                          atol=atol, chunk_size=batch_size, check=True,
                          reuse_alpha=True, on_failure='return')
        values = _ev_fields(p, res.values)
        with np.errstate(over='ignore', invalid='ignore'):
            values32 = values.astype(np.complex64)
        if not np.isfinite(res.values).all() or not np.isfinite(values32).all():
            raise FloatingPointError(f'Nonfinite raw/normalized values in work rows {start}:{stop}.')
        raw[target], fields[target] = res.values, values32
        converged[target] = res.converged_components
        n_used[target], quad_change[target] = res.n_used, res.error_estimate
        print(f'\r{stop}/{total} nodes; {time.perf_counter()-started:.1f} s', end='', flush=True)
    print()
    metadata = dict(
        formula_version='joint_shared_A_v1', S_star=0.1, contour='physical',
        w_grid='explicit' if w_nodes is not None else 'geometric_q',
        axis_generation=['explicit' if x is not None else ('geometric_q' if j == 3 else 'geometric')
                         for j, x in enumerate(supplied)],
        beta_c_factor=0.999, odd_normalization='geometry_2/residual2',
        n_quad=int(n_quad), max_quad=int(max_quad),
        rtol=np.asarray(rtol).tolist(), atol=np.asarray(atol).tolist(),
        u_max=7.0, axis_order=['alpha','beta','chi','w'],
        raw_components=['T_e','T_o','H_e','A_e'],
        fields=['R_e_T','R_e_H','R_A','R_o_0','R_o_B'],
        quadrature_converged=bool(converged.all()),
        interpolation_accuracy_validated=False,
    )
    payload = dict(
        alpha_nodes=axes[0], beta_nodes=axes[1], chi_nodes=axes[2], w_nodes=axes[3],
        raw_values=raw.reshape(counts+(4,)),
        converged_components=converged.reshape(counts+(4,)),
        n_used=n_used.reshape(counts),
        quadrature_change=quad_change.reshape(counts+(4,)),
        metadata=np.array(json.dumps(metadata)),
    )
    for j, name in enumerate(metadata['fields']):
        payload[name] = fields[:, j].reshape(counts)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **payload)
    temp.replace(path)
    failed = np.flatnonzero(~converged.all(axis=1))
    print(f'Saved: {path.resolve()}')
    print(f'Grid: {counts}; unconverged nodes: {len(failed)}/{total}')
    if len(failed):
        print('Inspect convergence flags before using this table. First flat indices:', failed[:10])
    return path

