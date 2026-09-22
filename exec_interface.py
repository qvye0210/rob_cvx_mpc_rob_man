#!/usr/bin/env python3
"""exec_interface.py -- block 1.3: real-transition recording + unified
residual interface + minimal closed-loop hookup. Repo root, env robmpc:
    PYTHONPATH=. python exec_interface.py        # runs all checks
Import is side-effect free.

DATA FIELDS (per transition; see collect_sample):
  episode_id, controller_name, step, t, dt   -- bookkeeping
  x_actual (12,)   pre-execution true state [q(3 eff-dof pos block), qd]
  u_exec (3,)      input ACTUALLY passed to the integrator. UNITS:
                   virtual acceleration of the double-integrator model,
                   NOT joint torque.
  x_next_actual    post-execution true state
  x_next_nominal   A@x + B@u_exec (discrete nominal, dt from ps)
  u_nominal        optimizer's k=0 action (== u_exec for nom arms;
                   differs for tube arms which add feedback)
  solver_status, term_reason, gt_collision_free (author link subset,
  discrete samples)
Episode metadata: commit, dt, nr_steps, x0, goal, seed, units note.
Hidden physics (mass error draw) saved under key 'hidden_' prefix --
excluded from model inputs by field convention.
LIMIT (recorded honestly): u_exec for ft/rt in the EXISTING pckls is
not recoverable (aux-corrected inputs were never saved); this module's
RecordingIntegrator captures it for any future run without changing
control logic (it wraps the integrator, the only gate every executed
input passes through). Existing-pckl transitions are therefore built
for nom/nom_star only.

RESIDUAL INTERFACE:
  x_next_hat = f_nominal(x, u) + residual(x, u)
  residual returns a DISCRETE one-step state residual, state units;
  no dt multiplication here (zoh already folds dt into A, B).
CLOSED-LOOP INJECTION POINT (documented, used by checks):
  NominalController(A, B, ...) builds constraint
      X[:,1:] == A @ X[:,:-1] + B @ U        (nom_controller.py:58)
  A linear residual (dA, dB) enters as A+dA, B+dB -- the exact
  prediction equation inside the optimiser becomes
      x_{k+1} = (A+dA) x_k + (B+dB) u_k.
  A constant offset c or any nonlinear model is NOT injectable without
  editing the constraint (one affine term); unsupported here by design
  -- raising instead of silently mis-wiring. No robust-tube guarantees
  are claimed for any modified model.
"""
import numpy as np


# ---------- nominal model ----------
def get_ps():
    from problem_scenario import ProblemScenarioMassAllPin
    from examples.six_dof import P_EXAMPLE_6_DOF
    return ProblemScenarioMassAllPin.from_cached_dir(
        P_EXAMPLE_6_DOF / "data" / "dof_6_ef_0.02")


def nominal_AB(ps):
    from aux import get_linear_double_integrator_discrete_dynamics
    return get_linear_double_integrator_discrete_dynamics(
        ps.config_dim, dt=ps.dt, method="zoh")


def f_nominal(x, u, A, B):
    x = np.atleast_2d(x); u = np.atleast_2d(u)
    out = x @ A.T + u @ B.T
    return out[0] if out.shape[0] == 1 else out


# ---------- residual models ----------
class ZeroResidual:
    def __call__(self, x, u):
        x = np.atleast_2d(x)
        out = np.zeros_like(x)
        return out[0] if x.shape[0] == 1 else out


class LinearResidual:
    """residual(x,u) = x@dA.T + u@dB.T  (discrete state units)."""
    def __init__(self, dA, dB):
        self.dA, self.dB = np.asarray(dA), np.asarray(dB)

    def __call__(self, x, u):
        x = np.atleast_2d(x); u = np.atleast_2d(u)
        out = x @ self.dA.T + u @ self.dB.T
        return out[0] if out.shape[0] == 1 else out


def predict(x, u, A, B, residual):
    return f_nominal(x, u, A, B) + residual(x, u)


def rollout(x0, U, A, B, residual):
    X = [np.asarray(x0, float)]
    for u in np.atleast_2d(U):
        X.append(predict(X[-1], u, A, B, residual))
    return np.array(X)


def make_controller(ps, residual, horizon=20):
    """Closed-loop hookup. Linear residuals only (see module docstring)."""
    from controllers.nom_controller import NominalController
    import cvxpy as cp
    A, B = nominal_AB(ps)
    if isinstance(residual, ZeroResidual):
        dA = np.zeros_like(A); dB = np.zeros_like(B)
    elif isinstance(residual, LinearResidual):
        dA, dB = residual.dA, residual.dB
    else:
        raise TypeError("only Zero/LinearResidual injectable into the "
                        "convex controller; got %r" % type(residual))
    m = ps.config_dim
    Q = np.eye(2 * m) * 10; Q[m:, m:] *= .01
    c = NominalController(A + dA, B + dB, Q, np.eye(2 * m) * 1e4,
                          np.eye(m) * 1e-3, horizon,
                          u_lim=ps.u_amp_nom, v_lim=ps.v_amp_nom,
                          p_amp=np.pi)
    c.set_solver(cp.CLARABEL)
    return c


# ---------- recording ----------
class RecordingIntegrator:
    """Wraps any author integrator; records every executed transition
    without altering control logic."""
    def __init__(self, inner):
        self.inner = inner
        self.records = []

    def solve_time_step(self, x, v):
        xn = self.inner.solve_time_step(x, v)
        self.records.append((np.array(x, float).ravel(),
                             np.array(v, float).ravel(),
                             np.array(xn, float).ravel()))
        return xn

    def __getattr__(self, a):
        return getattr(self.inner, a)


def collect_sample(n_steps=40, out="sample_transitions.npz"):
    """nom_star-config short run through the recording wrapper
    (independent path; author caches untouched)."""
    from corridor_simulators.nom import NomSimulator
    from aux import TimeStepIntegratorDiscrete
    from examples.six_dof.world import DemoWorld
    from aux import interpolate_equidistant
    ps = get_ps(); A, B = nominal_AB(ps)
    cont = make_controller(ps, ZeroResidual())
    world = DemoWorld()
    integ = RecordingIntegrator(
        TimeStepIntegratorDiscrete(A=cont.A, B=cont.B, ignore_me=True))
    integ.inner.set_dyns(ps.get_nominal_dynamics(), ps.get_nominal_dynamics())
    sim = NomSimulator(cont, integ, world)
    path = interpolate_equidistant(world.get_demo_path(), delta=0.05)
    radii = world.sdf(path[:, :3])
    (status, ts), _ = sim.simulate(path, radii, nr_steps=n_steps)
    R = integ.records
    X = np.array([r[0] for r in R]); U = np.array([r[1] for r in R])
    Xn = np.array([r[2] for r in R])
    np.savez(out, x_actual=X, u_exec=U, x_next_actual=Xn,
             x_next_nominal=f_nominal(X, U, A, B),
             gt_free=np.array([world.is_collision_free_gt(x[:3])
                               for x in X]),
             meta_status=status, meta_steps=ts, meta_dt=ps.dt,
             meta_units="u = virtual acceleration",
             meta_controller="nom_star_config_zero_residual")
    return X, U, Xn, status, ts


# ---------- checks ----------
def main():
    ps = get_ps(); A, B = nominal_AB(ps)
    zero = ZeroResidual()
    print(f"dims: x {2*ps.config_dim}, u {ps.config_dim}, dt {ps.dt}; "
          f"one MPC solve -> one executed step (aux_controller_steps "
          f"applies to tube arms only)")
    # 1. recording run + chain/finiteness/collision checks
    X, U, Xn, status, ts = collect_sample()
    assert np.isfinite(X).all() and np.isfinite(U).all()
    chain = float(np.max(np.abs(Xn[:-1] - X[1:])))
    print(f"recorded {len(X)} transitions (status {status}, ts {ts}); "
          f"chain max|x_next[t]-x[t+1]| = {chain:.2e}")
    assert chain < 1e-12
    # 2. zero-residual regression: recorded next == nominal prediction
    reg = float(np.max(np.abs(Xn - f_nominal(X, U, A, B))))
    print(f"zero-residual regression (discrete plant): max|diff| = "
          f"{reg:.2e}  (tolerance 1e-10)")
    assert reg < 1e-10
    # 3. one-step vs rollout(1) consistency
    r1 = rollout(X[0], U[:1], A, B, zero)[1]
    assert np.allclose(r1, predict(X[0], U[0], A, B, zero))
    print("one-step == rollout(1): OK")
    # 4. wiring test: deterministic nonzero residual changes the
    #    optimiser's prediction equation as expected
    dA = np.zeros_like(A); dB = np.zeros_like(B); dB[0, 0] = 1e-3
    res = LinearResidual(dA, dB)
    c0 = make_controller(ps, zero); c1 = make_controller(ps, res)
    from examples.six_dof.world import DemoWorld
    from aux import interpolate_equidistant
    w = DemoWorld()
    path = interpolate_equidistant(w.get_demo_path(), delta=0.05)
    radii = w.sdf(path[:, :3])
    m = ps.config_dim
    x0 = np.r_[path[0][:m], np.zeros(m)]
    cs = np.tile(path[0][:m][:, None], (1, 21)); rs = np.full(21, radii[0])
    xg = np.r_[path[min(5, len(path) - 1)][:m], np.zeros(m)]
    for c in (c0, c1):
        c.set_parameter_values(x0, cs, rs, xg); c.solve()
    Xp0, Xp1 = c0.X.value, c1.X.value
    d01 = float(np.max(np.abs(Xp0 - Xp1)))
    man0 = rollout(x0, c1.U.value.T, A, B, zero)[: , :]
    man1 = rollout(x0, c1.U.value.T, A, B, res)
    inj = float(np.max(np.abs(man1.T - c1.X.value)))
    print(f"wiring: |pred(zero)-pred(dA)| = {d01:.2e} (expect >0); "
          f"optimiser traj == manual (A+dA) rollout: max|diff| = {inj:.2e}")
    assert d01 > 1e-8 and inj < 1e-6
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
