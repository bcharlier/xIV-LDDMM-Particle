"""
Running-time benchmark of the xmodmap2 deformation modes on the 2D rotation example.

Part A (deformation only): for each mode, time of
    * H        : evaluation of the Hamiltonian,
    * forward  : shooting to t = 1,
    * fwd+bwd  : gamma * H + shooting + backward pass to (px, pw)  (= deformation cost of one loss evaluation),
  with the closed-form flow and with the same mode integrated numerically (Ralston, nt = 10),
  plus the original xmodmap Hamiltonian + Shooting ("all-mixed") as reference.

Part B (full registration): for each mode, run the optimizer of framework_script_2d_rotation_yi.py
  and report wall time, number of loss evaluations, final loss, and the error of the
  recovered rotation / translation against the ground truth.

Usage:
    python benchmark_modes_yi.py --sizes 1000 10000 --npoints-registration 2000 --steps 20
Results are written to output/benchmark/benchmark_results.json and .md
"""

import argparse
import json
import math
import os
import platform
import statistics
import time
from sys import path as sys_path

HERE = os.path.dirname(os.path.abspath(__file__))
sys_path.append(os.path.join(HERE, "..", ".."))

import torch
import xmodmap
import xmodmap2
from xmodmap2.deformation import Deformation
from xmodmap2.preprocess.makePQ_legacy import makePQ

MODES = ["translation", "rotation", "scaling", "rigid", "similarity", "affine",
         "non-rigid", "non-rigid+rigid", "all-mixed"]
SIGMA_RKHS = [0.2, 0.1, 0.05]
SIGMA_VAR = [0.2, 0.1, 0.05, 0.02]
COEFS = dict(cA=1.0, cT=1.0, cScale=1.0, cNR=10.0, cL=1.0)
GAMMA = 0.1
ANGLE = 0.6


def load_source():
    data = torch.load(os.path.join(HERE, "..", "..", "data", "2D_amygdala", "hat_run77_10000points_real.pt"))
    S = data["points"][:, :2].contiguous() if isinstance(data, dict) else data[0][:, :2].contiguous()
    return S


def timeit(fn, repeats):
    fn()  # warm-up (KeOps compilation, caches)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def deformation_inputs(S, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(S.shape[0], generator=g)[:N]
    X = S[idx]
    Stilde = (X - X.min(0).values) / (X.max(0).values - X.min(0).values).max()
    qw = torch.ones(N, 1)
    px = 0.3 * torch.randn(N, 2, generator=g) / N ** 0.5
    pw = 0.03 * torch.randn(N, 1, generator=g) / N ** 0.5
    return px, pw, Stilde, qw


def bench_hamiltonian(energy, endpoint, px, pw, qx, qw, repeats):
    def f_H():
        with torch.no_grad():
            energy(px, pw, qx, qw)

    def f_fwd():  # momenta require grad, as inside the optimizer (graph is built)
        endpoint(px.clone().requires_grad_(True), pw.clone().requires_grad_(True), qx, qw)

    def f_fwdbwd():
        pxv, pwv = px.clone().requires_grad_(True), pw.clone().requires_grad_(True)
        state = endpoint(pxv, pwv, qx, qw)
        loss = GAMMA * energy(pxv, pwv, qx, qw) + state[2].square().sum() + state[3].sum()
        torch.autograd.grad(loss, (pxv, pwv), allow_unused=True)

    return {"H": timeit(f_H, repeats), "forward": timeit(f_fwd, repeats), "fwd+bwd": timeit(f_fwdbwd, repeats)}


def part_a(S, sizes, repeats):
    rows = []
    for N in sizes:
        px, pw, qx, qw = deformation_inputs(S, N)
        reps = repeats if N <= 2000 else max(1, repeats // 2)

        # reference: original xmodmap
        H_old = xmodmap.deformation.Hamiltonian(SIGMA_RKHS, qx, cA=1.0, cS=COEFS["cNR"], cT=1.0, dimEff=2)
        sh_old = xmodmap.deformation.Shooting(SIGMA_RKHS, qx, cA=1.0, cS=COEFS["cNR"], cT=1.0, dimEff=2)
        qxg, qwg = qx.clone().requires_grad_(True), qw.clone().requires_grad_(True)
        t = bench_hamiltonian(H_old, lambda *a: sh_old(*a)[-1], px, pw, qxg, qwg, reps)
        rows.append({"N": N, "mode": "all-mixed (xmodmap)", "flow": "Ralston nt=10", **t, "endpoint_rel_diff": None})
        print(rows[-1], flush=True)

        for mode in MODES:
            D = Deformation(mode, qx, sigma=SIGMA_RKHS, dimEff=2, **COEFS)
            h = D.hamiltonian
            flows = ["closed form", "Ralston nt=10"] if D.has_closed_form else ["Ralston nt=10"]
            ref_end = None
            for flow in flows:
                h.has_closed_form = flow == "closed form"  # force numerical integration for comparison
                t = bench_hamiltonian(h, h.endpoint, px, pw, qx, qw, reps)
                end = h.endpoint(px, pw, qx, qw)[2].detach()
                rel = None
                if ref_end is None:
                    ref_end = end
                else:  # numerical endpoint versus exact closed form
                    rel = ((end - ref_end).abs().max() / (ref_end - qx).abs().max()).item()
                rows.append({"N": N, "mode": mode, "flow": flow, **t, "endpoint_rel_diff": rel})
                print(rows[-1], flush=True)
    return rows


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------

def registration_problem(S, npoints):
    torch.manual_seed(0)
    translation = torch.randn(1, 2)
    if npoints is not None and npoints < S.shape[0]:
        S = S[torch.randperm(S.shape[0])[:npoints]]
    R = torch.tensor([[math.cos(ANGLE), -math.sin(ANGLE)], [math.sin(ANGLE), math.cos(ANGLE)]])
    T = S @ R.T + translation
    nu = torch.ones(S.shape[0], 1)
    w_S, w_T, zeta_S, zeta_T, _, _, _, Stilde, Ttilde, s, m, pi_STinit, _ = makePQ(
        S, nu, T, nu.clone(), Csqpi=10000.0, lambInit=-1, Csqlamb=100.0
    )
    b_true = ((m @ R.T + translation - m) / s).flatten()
    return dict(w_S=w_S, w_T=w_T, zeta_S=zeta_S, zeta_T=zeta_T, Stilde=Stilde, Ttilde=Ttilde,
                pi_STinit=pi_STinit, b_true=b_true)


def rigid_fit(qx0, qx1, qw0):
    w = qw0 / qw0.sum()
    xc, yc = (w * qx0).sum(0), (w * qx1).sum(0)
    U, _, Vt = torch.linalg.svd(((qx1 - yc) * w).T @ (qx0 - xc))
    D = torch.eye(2)
    D[-1, -1] = torch.sign(torch.linalg.det(U @ Vt))
    Phi = U @ D @ Vt
    return Phi, yc - Phi @ xc


def part_b(S, npoints, steps, modes):
    prob = registration_problem(S, npoints)
    rows = []
    for mode in modes:
        dataloss = xmodmap2.distance.LossVarifoldNorm(SIGMA_VAR, prob["w_T"], prob["zeta_T"], prob["Ttilde"])
        dataloss.normalize_across_scale(prob["Stilde"], prob["w_S"], prob["zeta_S"], prob["pi_STinit"])
        D = Deformation(mode, prob["Stilde"], sigma=SIGMA_RKHS, dimEff=2, weight=GAMMA, **COEFS)
        model = xmodmap2.model.SingleModality(D, dataloss)
        variables = {
            "px": torch.zeros_like(prob["Stilde"]).requires_grad_(True),
            "pw": torch.zeros_like(prob["w_S"]).requires_grad_(True),
            "qx": prob["Stilde"].clone().detach().requires_grad_(True),
            "qw": prob["w_S"].clone().detach().requires_grad_(True),
            "zeta_S": prob["zeta_S"],
        }
        model.init(variables, ["px", "pw"], precond={"px": 1.0, "pw": 1.0 / 2 / prob["w_S"]},
                   savedir=os.path.join(HERE, "output", "benchmark", "runs", mode.replace("+", "_")))
        # one loss evaluation (forward + backward), after a warm-up
        model.closure()
        t0 = time.perf_counter()
        model.closure()
        t_eval = time.perf_counter() - t0
        model.logs["full"].clear()

        t0 = time.perf_counter()
        model.optimize(steps)
        wall = time.perf_counter() - t0
        n_evals = len(model.logs["full"])

        var = model.get_variables_optimized()
        px0, pw0, qx0, qw0 = (var[k].detach() for k in ("px", "pw", "qx", "qw"))
        with torch.no_grad():
            final = D(px0, pw0, qx0, qw0)
            qx1 = final[2].detach()
            if D.has_closed_form:
                Phi, b = D.flow_map(px0, pw0, qx0, qw0)
            else:
                Phi, b = rigid_fit(qx0, qx1, qw0)
            losses = model.logs["full"][-1]
        angle = math.atan2(Phi[1, 0].item(), Phi[0, 0].item())
        rows.append({
            "mode": mode,
            "closed_form": D.has_closed_form,
            "N": prob["Stilde"].shape[0],
            "time_per_eval_s": t_eval,
            "optim_wall_s": wall,
            "n_loss_evals": n_evals,
            "optimizer_steps": model.current_step + 1,
            "final_H_loss": losses[0].item(),
            "final_data_loss": losses[1].item(),
            "angle": angle,
            "angle_error": abs(angle - ANGLE),
            "translation_error": (b - prob["b_true"]).norm().item(),
            "rms_to_target": (qx1 - prob["Ttilde"]).square().sum(-1).mean().sqrt().item(),
        })
        print(rows[-1], flush=True)
    return rows


# ---------------------------------------------------------------------------

def to_markdown(meta, rows_a, rows_b):
    lines = ["# Benchmark results", "", "```", json.dumps(meta, indent=1), "```", ""]
    if rows_a:
        lines += ["## Part A: deformation only (median seconds)", "",
                  "| N | mode | flow | H | forward | fwd+bwd | rel. diff. to closed form |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows_a:
            rel = "" if r["endpoint_rel_diff"] is None else f"{r['endpoint_rel_diff']:.1e}"
            lines.append(f"| {r['N']} | {r['mode']} | {r['flow']} | {r['H']:.4f} | {r['forward']:.4f} | "
                         f"{r['fwd+bwd']:.4f} | {rel} |")
        lines.append("")
    if rows_b:
        lines += ["## Part B: registration", "",
                  "| mode | closed form | N | s / loss eval | optim. wall (s) | loss evals | H loss | data loss | "
                  "angle | angle err. | transl. err. | RMS to target |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows_b:
            lines.append(
                f"| {r['mode']} | {r['closed_form']} | {r['N']} | {r['time_per_eval_s']:.3f} | {r['optim_wall_s']:.1f} | "
                f"{r['n_loss_evals']} | {r['final_H_loss']:.3e} | {r['final_data_loss']:.3e} | {r['angle']:.4f} | "
                f"{r['angle_error']:.1e} | {r['translation_error']:.1e} | {r['rms_to_target']:.1e} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=[1000, 10000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--npoints-registration", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--modes", nargs="*", default=MODES)
    parser.add_argument("--skip-a", action="store_true")
    parser.add_argument("--skip-b", action="store_true")
    args = parser.parse_args()

    S = load_source()
    meta = {
        "machine": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "sigmaRKHS": SIGMA_RKHS,
        "sigmaVar": SIGMA_VAR,
        "coefficients": COEFS,
        "gamma": GAMMA,
        "registration_points": args.npoints_registration,
        "registration_steps": args.steps,
    }
    rows_a = [] if args.skip_a else part_a(S, args.sizes, args.repeats)
    rows_b = [] if args.skip_b else part_b(S, args.npoints_registration, args.steps, args.modes)

    outdir = os.path.join(HERE, "output", "benchmark")
    os.makedirs(outdir, exist_ok=True)
    tag = "" if not (args.skip_a or args.skip_b) else ("_A" if args.skip_b else "_B")
    with open(os.path.join(outdir, f"benchmark_results{tag}.json"), "w") as f:
        json.dump({"meta": meta, "part_a": rows_a, "part_b": rows_b}, f, indent=1)
    with open(os.path.join(outdir, f"benchmark_results{tag}.md"), "w") as f:
        f.write(to_markdown(meta, rows_a, rows_b))
    print(to_markdown(meta, rows_a, rows_b))
