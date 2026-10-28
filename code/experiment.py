"""Activation-function experiment, matrix form, NumPy only (no frameworks).

Network 2 -> H -> 1: Z1 = W1 X + b1 1^T, A1 = phi(Z1), Z2 = W2 A1 + b2 1^T,
Yhat = logistic(Z2), loss l = -(1/m) sum_j [y_j log yhat_j + (1-y_j) log(1-yhat_j)].
Samples are COLUMNS: X is 2 x m.  '@' is matrix multiplication; '*' is ONLY the
element-wise (Hadamard) product.  With logistic output + cross-entropy,
Delta2 = dl/dZ2 = (Yhat - Y)/m  (the logistic' factor cancels).  Here we DO average by 1/m.
"""
import json, numpy as np

ACTS = {  # name: (phi, phi')
    "identity": (lambda z: z,                 lambda z: np.ones_like(z)),
    "logistic": (lambda z: 1/(1+np.exp(-z)),  lambda z: (1/(1+np.exp(-z)))*(1-1/(1+np.exp(-z)))),
    "tanh":     (np.tanh,                     lambda z: 1-np.tanh(z)**2),
    "relu":     (lambda z: np.maximum(z, 0),  lambda z: (z > 0).astype(float)),  # ReLU'(0) := 0
}
H, ETA, EPOCHS, SEEDS = 8, 0.5, 5000, [0, 1, 2, 3, 4]

def logistic(z): return 1/(1+np.exp(-z))

def make_xor():
    return np.array([[0., 0, 1, 1], [0, 1, 0, 1]]), np.array([[0., 1, 1, 0]])

def make_moons(m=200, noise=0.15, seed=0):            # numpy-only two moons
    r = np.random.default_rng(seed); t = np.pi*r.random(m//2); s = np.pi*r.random(m//2)
    X = np.hstack([np.vstack([np.cos(t), np.sin(t)]), np.vstack([1-np.cos(s), 0.5-np.sin(s)])])
    X = X + noise*r.normal(size=X.shape)
    return X, np.hstack([np.zeros((1, m//2)), np.ones((1, m//2))])

def init(seed, n0=2, h=H):                             # same scheme for every activation
    r = np.random.default_rng(seed)                    # W ~ N(0, 1/n_in), b = 0
    return [r.normal(0, 1/np.sqrt(n0), (h, n0)), np.zeros((h, 1)),
            r.normal(0, 1/np.sqrt(h), (1, h)), np.zeros((1, 1))]

def forward(P, X, phi):
    W1, b1, W2, b2 = P; one = np.ones((X.shape[1], 1))
    Z1 = W1 @ X + b1 @ one.T; A1 = phi(Z1)
    return Z1, A1, logistic(W2 @ A1 + b2 @ one.T)

def bce(Yh, Y):
    Yh = np.clip(Yh, 1e-12, 1-1e-12)
    return float(-np.mean(Y*np.log(Yh) + (1-Y)*np.log(1-Yh)))

def train(X, Y, act, seed):
    phi, dphi = ACTS[act]; P = init(seed); m = X.shape[1]; one = np.ones((m, 1)); hist = []
    for _ in range(EPOCHS):
        Z1, A1, Yh = forward(P, X, phi); hist.append(bce(Yh, Y))
        D2 = (Yh - Y)/m                                # 1 x m
        D1 = (P[2].T @ D2) * dphi(Z1)                  # H x m:  [(W2)^T Delta2] (.) phi'(Z1)
        G = [D1 @ X.T, D1 @ one, D2 @ A1.T, D2 @ one]  # dl/dW1, dl/db1, dl/dW2, dl/db2
        P = [p - ETA*g for p, g in zip(P, G)]
    _, _, Yh = forward(P, X, phi)
    return P, np.array(hist + [bce(Yh, Y)]), float(np.mean((Yh > 0.5) == (Y > 0.5)))

def first_epoch_below(h, thr):
    idx = np.nonzero(h < thr)[0]; return int(idx[0]) if idx.size else None

if __name__ == "__main__":
    data = {"xor": make_xor(), "moons": make_moons()}
    results, models, curves = {}, {}, {}
    for dn, (X, Y) in data.items():
        for act in ACTS:
            runs = [train(X, Y, act, s) for s in SEEDS]
            L = np.array([r[1][-1] for r in runs]); A = np.array([r[2] for r in runs])
            thr = 0.1 if dn == "xor" else 0.3
            results[f"{dn}/{act}"] = dict(
                loss_mean=L.mean(), loss_std=L.std(), loss_min=L.min(), loss_max=L.max(),
                acc_mean=A.mean(), acc_min=A.min(), acc_max=A.max(), accs=A.tolist(),
                losses=L.tolist(), thr=thr,
                epochs_to_thr=[first_epoch_below(r[1], thr) for r in runs])
            models[(dn, act)] = runs[0][0]; curves[(dn, act)] = np.array([r[1] for r in runs])
            r = results[f"{dn}/{act}"]
            print(f"{dn:5s} {act:8s} loss {r['loss_mean']:.4f} +- {r['loss_std']:.4f} "
                  f"[{r['loss_min']:.4f}, {r['loss_max']:.4f}]  acc {r['acc_mean']:.3f} "
                  f"[{r['acc_min']:.3f}, {r['acc_max']:.3f}]  epochs to loss<{thr}: {r['epochs_to_thr']}")
    json.dump(results, open("experiment_results.json", "w"), indent=1)
    np.savez("experiment_state.npz", **{f"curve_{d}_{a}": c for (d, a), c in curves.items()},
             **{f"P{i}_{d}_{a}": p for (d, a), P in models.items() for i, p in enumerate(P)})
