"""
Tests of xmodmap2.deformation.

    KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_xmodmap2_deformation.py -q

* "all-mixed" reproduces xmodmap (Hamiltonian, shooting, grid / backward shooting, gradients);
* every closed-form flow matches a fine RK4 integration of Hamilton's equations;
* closed-form trajectories conserve H, can be inverted by backward shooting and are differentiable;
* the Deformation API (mode strings, cS renaming, transform, flow_map, energies, weight).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import xmodmap  # noqa: E402
import xmodmap2  # noqa: E402
from xmodmap2.deformation import Deformation, parse_mode  # noqa: E402

F64 = torch.float64
SIGMA = [0.2, 0.1, 0.05]


def make_state(N=150, d=2, dtype=torch.float32, scale=0.3, seed=0, planar=False):
    g = torch.Generator().manual_seed(seed)
    qx = torch.rand(N, d, generator=g, dtype=dtype)
    if planar:
        qx[:, -1] = 0
    qw = 0.5 + torch.rand(N, 1, generator=g, dtype=dtype)
    px = scale * torch.randn(N, d, generator=g, dtype=dtype) / N ** 0.5
    pw = scale * torch.randn(N, 1, generator=g, dtype=dtype) / N ** 0.5
    X = torch.rand(40, d, generator=g, dtype=dtype)
    wX = 0.5 + torch.rand(40, 1, generator=g, dtype=dtype)
    return px, pw, qx, qw, X, wX


def assert_close(a, b, rtol, atol, msg=""):
    err = (a - b).abs().max().item()
    assert torch.allclose(a, b, rtol=rtol, atol=atol), f"{msg}: max abs error {err:.3e}"


def assert_states_close(s1, s2, rtol=1e-7, atol=1e-9, names=("px", "pw", "qx", "qw", "X", "wX")):
    assert len(s1) == len(s2)
    for a, b, n in zip(s1, s2, names):
        assert_close(a, b, rtol, atol, n)


# ---------------------------------------------------------------------------
# Mode strings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mode, expected",
    [
        ("rigid", ["rotation", "translation"]),
        ("similarity", ["rotation", "translation", "scaling"]),
        ("affine", ["translation", "linear"]),
        ("all-mixed", ["non-rigid", "rotation", "translation", "scaling"]),
        ("Non_Rigid", ["non-rigid"]),
        ("translation + rotation", ["rotation", "translation"]),
        ("nonrigid+rigid", ["non-rigid", "rotation", "translation"]),
        ("affine+non-rigid", ["non-rigid", "translation", "linear"]),
    ],
)
def test_parse_mode(mode, expected):
    assert parse_mode(mode) == expected


@pytest.mark.parametrize("mode", ["shear", "affine+rotation", "linear+similarity"])
def test_parse_mode_errors(mode):
    with pytest.raises(ValueError):
        parse_mode(mode)


def test_cS_is_rejected_with_explanation():
    _, _, qx, _, _, _ = make_state()
    with pytest.raises(TypeError, match="cNR"):
        Deformation("rigid", qx, cS=10.0)


def test_nonrigid_needs_sigma():
    _, _, qx, _, _, _ = make_state()
    with pytest.raises(ValueError, match="sigma"):
        Deformation("non-rigid", qx)


# ---------------------------------------------------------------------------
# Equivalence with xmodmap ("all-mixed")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "d, dimEff, single",
    [
        (2, 2, False),
        (3, 3, False),
        (3, 3, True),
        pytest.param(
            3, 2, False,
            marks=pytest.mark.xfail(
                raises=IndexError,
                strict=True,
                reason="bug in xmodmap: getATauAlpha indexes the 1-D barycenter as xc[:, :2] when d=3, dimEff=2",
            ),
        ),
    ],
)
def test_all_mixed_matches_xmodmap(d, dimEff, single):
    px, pw, qx, qw, X, wX = make_state(d=d, planar=(dimEff == 2 and d == 3))
    px.requires_grad_(True)
    pw.requires_grad_(True)
    qx.requires_grad_(True)
    qw.requires_grad_(True)
    kw = dict(cA=2.0, cT=0.5, dimEff=dimEff, single=single)

    old_H = xmodmap.deformation.Hamiltonian(SIGMA, qx.detach(), cS=10.0, **kw)
    old_shoot = xmodmap.deformation.Shooting(SIGMA, qx.detach(), cS=10.0, **kw)
    new = Deformation("all-mixed", qx.detach(), sigma=SIGMA, cNR=10.0, **kw)

    h_old, h_new = old_H(px, pw, qx, qw), new.hamiltonian(px, pw, qx, qw)
    assert_close(h_old, h_new, 1e-5, 1e-7, "H")

    s_old, s_new = old_shoot(px, pw, qx, qw)[-1], new(px, pw, qx, qw)
    assert_states_close(s_old, s_new, rtol=1e-4, atol=1e-5)

    g_old = torch.autograd.grad(s_old[2].square().sum() + h_old, (px, pw))
    g_new = torch.autograd.grad(s_new[2].square().sum() + h_new, (px, pw))
    for a, b in zip(g_old, g_new):
        assert_close(a, b, 1e-3, 1e-3 * a.abs().max().item(), "gradient")


def test_grid_and_backward_shooting_match_xmodmap(capsys):
    px, pw, qx, qw, X, wX = make_state(d=2)
    for v in (px, pw, qx, qw):
        v.requires_grad_(True)
    kw = dict(cA=1.0, cT=1.0, dimEff=2, single=False)
    new = Deformation("all-mixed", qx.detach(), sigma=SIGMA, cNR=10.0, **kw)

    old_grid = xmodmap.deformation.ShootingGrid(SIGMA, qx.detach(), cS=10.0, **kw)
    s_old = old_grid(px, pw, qx, qw, X, wX)[-1]
    s_new = new.hamiltonian.shoot(px, pw, qx, qw, X=X, wX=wX)[-1]
    assert_states_close(s_old, s_new, rtol=1e-4, atol=1e-5)

    old_back = xmodmap.deformation.ShootingBackwards(SIGMA, qx.detach(), cS=10.0, **kw)
    b_old = old_back(*s_new[:4], X, wX)[-1]
    b_new = new.hamiltonian.shoot_backward(*s_new[:4], X, wX)[-1]
    assert_states_close(b_old, b_new, rtol=1e-4, atol=1e-5)
    capsys.readouterr()  # xmodmap's grid system prints debug output


# ---------------------------------------------------------------------------
# Closed-form flows versus numerical integration (float64)
# ---------------------------------------------------------------------------

CLOSED_FORM_CASES = [
    # mode, d, dimEff, single
    ("translation", 2, 2, False),
    ("rotation", 2, 2, False),
    ("rotation", 3, 3, False),
    ("scaling", 2, 2, False),
    ("scaling", 3, 3, False),        # anisotropic xy / z
    ("scaling", 3, 2, False),        # z not scaled
    ("rigid", 2, 2, False),
    ("rigid", 3, 3, False),
    ("similarity", 2, 2, False),
    ("similarity", 3, 3, True),      # isotropic scaling commutes with rotations
    ("linear", 2, 2, False),
    ("affine", 2, 2, False),
    ("affine", 3, 3, False),
    ("translation+scaling", 3, 3, False),
]


@pytest.mark.parametrize("mode, d, dimEff, single", CLOSED_FORM_CASES)
def test_closed_form_matches_rk4(mode, d, dimEff, single):
    px, pw, qx, qw, X, wX = make_state(N=60, d=d, dtype=F64, scale=0.5, seed=1)
    D = Deformation(mode, qx, cA=0.7, cT=1.3, cScale=0.9, cL=1.1, dimEff=dimEff, single=single)
    h = D.hamiltonian
    assert D.has_closed_form

    x0 = (px, pw, qx, qw, X, wX)
    exact = h.flow_trajectory([0.5, 1.0], *x0)

    def rel_error(a, b):
        return max(((u - v).abs().max() / v.abs().max()).item() for u, v in zip(a, b))

    num200 = h.integrate(x0, nt=200, integrator="rk4")
    num400 = h.integrate(x0, nt=400, integrator="rk4")
    err200, err400 = rel_error(num200[-1], exact[1]), rel_error(num400[-1], exact[1])
    assert rel_error(num400[200], exact[0]) < 1e-7
    assert err400 < 1e-7, err400
    # RK4 converges at order 4 towards the closed form: error / 16 when nt doubles
    # (unless both are already at round-off level).
    assert err400 < 1e-13 or err200 / err400 > 10.0, (err200, err400)

    # The linear part really moves the points (the test is not trivial).
    assert (exact[1][2] - qx).abs().max() > 1e-2


@pytest.mark.parametrize("mode, d, dimEff, single", CLOSED_FORM_CASES)
def test_closed_form_conserves_H_and_is_invertible(mode, d, dimEff, single):
    px, pw, qx, qw, X, wX = make_state(N=60, d=d, dtype=F64, scale=0.5, seed=2)
    D = Deformation(mode, qx, cA=0.7, cT=1.3, cScale=0.9, cL=1.1, dimEff=dimEff, single=single)
    h = D.hamiltonian

    traj = h.shoot(px, pw, qx, qw, X=X, wX=wX, nt=5)
    H0 = h(px, pw, qx, qw)
    for s in traj:
        assert_close(h(*s[:4]), H0, 1e-10, 1e-12, "H conservation")

    back = h.shoot_backward(*traj[-1][:4], traj[-1][4], traj[-1][5])[-1]
    assert_states_close(back[2:], (qx, qw, X, wX), rtol=1e-10, atol=1e-12, names=("qx", "qw", "X", "wX"))
    assert_states_close(back[:2], (-px, -pw), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", ["rigid", "similarity", "affine"])
def test_closed_form_gradient_matches_numeric(mode):
    px, pw, qx, qw, _, _ = make_state(N=40, d=2, dtype=F64, scale=1.0, seed=3)
    target = torch.rand(40, 2, dtype=F64, generator=torch.Generator().manual_seed(4))
    D = Deformation(mode, qx, cA=0.7, cT=1.3, cScale=0.9, cL=1.1)
    h = D.hamiltonian

    gen = torch.Generator().manual_seed(5)
    weights = [torch.randn(v.shape, dtype=F64, generator=gen) for v in (px, pw, qx, qw)]

    def loss(state):
        # every entry of the final state (momenta included) plus the energy
        lin = sum((w * v).sum() for w, v in zip(weights, state))
        return lin + (state[2] - target).square().sum() + 0.1 * h(px, pw, qx, qw)

    px.requires_grad_(True)
    pw.requires_grad_(True)
    g_exact = torch.autograd.grad(loss(h.endpoint(px, pw, qx, qw)), (px, pw))
    g_num = torch.autograd.grad(loss(h.integrate((px, pw, qx, qw), nt=200, integrator="rk4")[-1]), (px, pw))
    for a, b in zip(g_exact, g_num):
        assert_close(a, b, 1e-7, 1e-9, "gradient")


def test_rotation_plus_anisotropic_scaling_is_integrated_numerically():
    """The controls are not conserved: a constant-L formula would be wrong."""
    px, pw, qx, qw, _, _ = make_state(N=60, d=3, dtype=F64, scale=1.5, seed=5)
    D = Deformation("similarity", qx, dimEff=3, single=False)
    h = D.hamiltonian
    assert not D.has_closed_form

    traj = h.integrate((px, pw, qx, qw), nt=200, integrator="rk4")
    L0, _ = h.controls(*traj[0])
    L1, _ = h.controls(*traj[-1])
    assert (L1 - L0).abs().max() > 1e-4
    assert_close(h(*traj[-1]), h(*traj[0]), 1e-8, 1e-10, "H conservation")


# ---------------------------------------------------------------------------
# Explicit 2D rotation and the role of the coefficients
# ---------------------------------------------------------------------------

def test_2d_rigid_flow_map_is_rotation_about_barycenter():
    px, pw, qx, qw, _, _ = make_state(N=50, d=2, dtype=F64, seed=6)
    D = Deformation("rigid", qx, cA=0.5, cT=2.0)
    A, tau = D.hamiltonian.controls(px, pw, qx, qw)
    theta = A[1, 0]
    Phi, b = D.flow_map(px, pw, qx, qw)
    R = torch.stack([torch.stack([theta.cos(), -theta.sin()]), torch.stack([theta.sin(), theta.cos()])])
    assert_close(Phi, R, 1e-12, 1e-12, "Phi")
    c = (qw * qx).sum(0) / qw.sum()
    assert_close(b, c + tau - R @ c, 1e-12, 1e-12, "b")
    assert_close(D(px, pw, qx, qw)[2], qx @ Phi.T + b, 1e-12, 1e-12, "qx(1)")


def test_common_rescaling_of_coefficients_equals_rescaling_gamma():
    """c_k -> lam c_k and p -> lam p: same deformation, H multiplied by lam."""
    px, pw, qx, qw, _, _ = make_state(N=80, d=2, seed=7)
    lam = 4.0
    coefs = dict(cA=1.0, cT=2.0, cScale=0.5, cNR=10.0)
    D1 = Deformation("all-mixed", qx, sigma=SIGMA, **coefs)
    D2 = Deformation("all-mixed", qx, sigma=SIGMA, **{k: lam * v for k, v in coefs.items()})
    s1 = D1(px, pw, qx, qw)
    s2 = D2(lam * px, lam * pw, qx, qw)
    assert_close(s1[2], s2[2], 1e-4, 1e-6, "qx(1)")
    assert_close(lam * D1.hamiltonian(px, pw, qx, qw), D2.hamiltonian(lam * px, lam * pw, qx, qw), 1e-4, 1e-7, "H")


def test_cNR_is_an_explicit_inverse_gain():
    px, pw, qx, qw, X, _ = make_state(N=80, d=2, seed=8)
    D1 = Deformation("non-rigid", qx, sigma=SIGMA, cNR=1.0)
    D10 = Deformation("non-rigid", qx, sigma=SIGMA, cNR=10.0)
    assert D1.hamiltonian.normalization == D10.hamiltonian.normalization  # data-dependent part only
    v1 = D1.hamiltonian.velocity(px, pw, qx, qw, X)
    v10 = D10.hamiltonian.velocity(px, pw, qx, qw, X)
    assert_close(v1, 10.0 * v10, 1e-5, 1e-7, "velocity")


# ---------------------------------------------------------------------------
# Deformation API and model
# ---------------------------------------------------------------------------

def test_deformation_api():
    px, pw, qx, qw, X, wX = make_state(N=80, d=2, seed=9)
    D = Deformation("all-mixed", qx, sigma=SIGMA, cNR=10.0, weight=0.1, nt=40)
    energies = D.energies(px, pw, qx, qw)
    assert list(energies) == ["non-rigid", "rotation", "translation", "scaling"]
    assert_close(sum(energies.values()), D.hamiltonian(px, pw, qx, qw), 1e-6, 1e-8, "sum of energies")
    assert_close(D.energy(px, pw, qx, qw), 0.1 * D.hamiltonian(px, pw, qx, qw), 1e-6, 1e-8, "weight")

    X1, wX1 = D.transform(px, pw, qx, qw, X, wX)
    final = D(px, pw, qx, qw)
    X0, wX0 = D.inverse_transform(*final, X1, wX1)
    assert_close(X0, X, 1e-3, 1e-4, "inverse transform")
    assert len(D.shoot(px, pw, qx, qw, nt=4)) == 5
    with pytest.raises(NotImplementedError):
        D.flow_map(px, pw, qx, qw)


@pytest.mark.parametrize("mode", ["rigid", "all-mixed"])
def test_single_modality_model_runs(tmp_path, mode, capsys):
    from xmodmap2.preprocess.makePQ_legacy import makePQ

    g = torch.Generator().manual_seed(10)
    S = torch.rand(120, 2, generator=g)
    T = S @ torch.tensor([[0.9, -0.3], [0.3, 0.9]]).T + 0.05
    nu = torch.ones(120, 1)
    w_S, w_T, zeta_S, zeta_T, _, _, _, Stilde, Ttilde, _, _, pi_STinit, _ = makePQ(S, nu, T, nu.clone())

    dataloss = xmodmap2.distance.LossVarifoldNorm([0.2, 0.05], w_T, zeta_T, Ttilde)
    dataloss.normalize_across_scale(Stilde, w_S, zeta_S, pi_STinit)
    D = Deformation(mode, Stilde, sigma=[0.2, 0.1], cNR=10.0, dimEff=2, weight=0.1)

    model = xmodmap2.model.SingleModality(D, dataloss)
    variables = {
        "px": torch.zeros_like(Stilde).requires_grad_(True),
        "pw": torch.zeros_like(w_S).requires_grad_(True),
        "qx": Stilde.clone().detach().requires_grad_(True),
        "qw": w_S.clone().detach().requires_grad_(True),
        "zeta_S": zeta_S,
    }
    model.init(variables, ["px", "pw"], savedir=str(tmp_path))
    model.optimize(2)
    losses = [sum(l).item() for l in model.logs["full"]]
    assert losses[-1] < losses[0]
    capsys.readouterr()
