"""Figures for the activation experiment (run experiment.py first).
Colour-blind-safe Okabe-Ito palette. No sigma in any label: the sigmoid is called 'logistic'."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from experiment import ACTS, forward, make_xor, make_moons, logistic

OUT = "../figures/"
C = {"identity": "#999999", "logistic": "#E69F00", "tanh": "#0072B2", "relu": "#009E73"}
LBL = {"identity": "identity", "logistic": "logistic (sigmoid)", "tanh": "tanh", "relu": "ReLU"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
S = np.load("experiment_state.npz")

# (1) activation functions and derivatives
z = np.linspace(-4, 4, 801)
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
for a, (name, (f, df)) in zip(ax, ACTS.items()):
    a.axhline(0, color="#dddddd", lw=0.8); a.axvline(0, color="#dddddd", lw=0.8)
    a.plot(z, f(z), color=C[name], lw=2.2, label="φ(z)")
    a.plot(z, df(z), color="#222222", lw=1.4, ls="--", label="φ′(z)")
    a.set_title(LBL[name]); a.set_xlabel("z"); a.set_ylim(-1.5, 2.5)
ax[0].legend(frameon=False, loc="upper left")
fig.suptitle("Activation functions φ and their derivatives φ′ (ReLU′(0) := 0)", y=1.02)
fig.tight_layout(); fig.savefig(OUT + "fig1_activations.png", dpi=150, bbox_inches="tight")

# (2) decision boundaries, seed 0
data = {"xor": make_xor(), "moons": make_moons()}
fig, ax = plt.subplots(2, 4, figsize=(13, 6.4))
for i, (dn, (X, Y)) in enumerate(data.items()):
    pad = 0.5; x0, x1 = X[0].min()-pad, X[0].max()+pad; y0, y1 = X[1].min()-pad, X[1].max()+pad
    g0, g1 = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    G = np.vstack([g0.ravel(), g1.ravel()])            # grid points as COLUMNS, 2 x 90000
    for j, act in enumerate(ACTS):
        P = [S[f"P{k}_{dn}_{act}"] for k in range(4)]
        _, _, Yh = forward(P, G, ACTS[act][0]); Yh = Yh.reshape(g0.shape)
        a = ax[i, j]
        a.contourf(g0, g1, Yh, levels=np.linspace(0, 1, 11), cmap="PuOr_r", alpha=0.55)
        if np.ptp(Yh) > 1e-6:
            a.contour(g0, g1, Yh, levels=[0.5], colors="k", linewidths=1.5)
        else:   # output constant (identity on XOR): no meaningful boundary
            a.text(0.5, 0.5, f"ŷ = {Yh.mean():.3f} everywhere\n(no boundary)", transform=a.transAxes,
                   ha="center", va="center", fontsize=10, bbox=dict(fc="white", ec="#999999"))
        for cls, mk, col in [(0, "o", "#5e3c99"), (1, "s", "#e66101")]:
            msk = Y[0] == cls
            a.scatter(X[0, msk], X[1, msk], marker=mk, c=col, edgecolor="white",
                      s=70 if dn == "xor" else 18, lw=0.6, label=f"y = {cls}")
        acc = float(np.mean((forward(P, X, ACTS[act][0])[2] > 0.5) == (Y > 0.5)))
        tag = "acc: tie, ŷ = 0.5" if np.ptp(Yh) <= 1e-6 else f"acc {acc:.2f}"
        a.set_title(f"{dn.upper() if dn=='xor' else 'two moons'}: {LBL[act]}  ({tag})")
        a.set_xticks([]); a.set_yticks([])
ax[0, 0].legend(frameon=True, fontsize=8, loc="upper center")
fig.suptitle("Decision boundaries (black line: ŷ = 0.5), 2 → 8 → 1 network, seed 0", y=1.0)
fig.tight_layout(); fig.savefig(OUT + "fig2_decision_boundaries.png", dpi=150, bbox_inches="tight")

# (3) loss curves, mean over 5 seeds with min-max band
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, dn in zip(ax, data):
    for act in ACTS:
        c = S[f"curve_{dn}_{act}"]; ep = np.arange(c.shape[1])
        a.fill_between(ep, c.min(0), c.max(0), color=C[act], alpha=0.2, lw=0)
        a.plot(ep, c.mean(0), color=C[act], lw=2, label=LBL[act])
    a.set_yscale("log"); a.set_xlabel("epoch (full-batch gradient-descent step)")
    a.set_ylabel("cross-entropy loss ℓ (log scale)")
    a.set_title(("XOR (m = 4)" if dn == "xor" else "two moons (m = 200)") + ", η = 0.5")
ax[0].legend(frameon=False, loc="upper right")
fig.suptitle("Training loss: mean over 5 seeds, shaded band = min–max", y=1.02)
fig.tight_layout(); fig.savefig(OUT + "fig3_loss_curves.png", dpi=150, bbox_inches="tight")

for dn in data:   # extra convergence stat
    for act in ACTS:
        c = S[f"curve_{dn}_{act}"]
        for thr in (0.1, 0.05):
            e = [int(np.nonzero(r < thr)[0][0]) if (r < thr).any() else None for r in c]
            print(f"{dn:5s} {act:8s} epochs to loss < {thr}: {e}")
