"""Running example for CS103 'Neural Networks Expressed in Linear Algebra'.

Network: 2 inputs -> 2 hidden (ReLU) -> 1 output (identity/linear output), XOR data.
Notation (handoff section 2): column vectors, W^(l) in R^{n_l x n_{l-1}},
batch X in R^{n_0 x m} with samples as COLUMNS, Z = W A + b 1^T.
In code, '@' is matrix multiplication; '*' is ONLY used for the element-wise
(Hadamard) product, written as (.) in the notes.
Loss convention (used everywhere in this file): batch loss
    l = 1/2 sum_j (yhat_j - y_j)^2        (a SUM over samples, NO 1/m averaging)
per-sample loss l_j = 1/2 (yhat_j - y_j)^2, so l = sum_j l_j.
ReLU'(0) is DEFINED to be 0.
"""
import numpy as np

np.set_printoptions(precision=6, suppress=True)

def relu(z):  return np.maximum(z, 0.0)
def drelu(z): return (z > 0).astype(float)          # ReLU'(0) := 0

def show(name, M):
    M = np.atleast_2d(M)
    print(f"{name}  shape {M.shape[0]}x{M.shape[1]}\n{M}\n")

# XOR batch: samples as columns, X is 2x4, y is 1x4
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=float)
Y = np.array([[0, 1, 1, 0]], dtype=float)
m = X.shape[1]
ones = np.ones((m, 1))                               # bold 1 in R^m

def forward(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1 @ ones.T                        # bias broadcast = outer product b 1^T
    A1 = relu(Z1)
    Yhat = W2 @ A1 + b2 @ ones.T                     # linear output layer
    return Z1, A1, Yhat

def loss(W1, b1, W2, b2):
    _, _, Yhat = forward(W1, b1, W2, b2, X)
    return 0.5 * np.sum((Yhat - Y) ** 2)

# ---------------------------------------------------------------- Part 1
print("=" * 70, "\nPART 1: Goodfellow et al. (2016) sec. 6.1 exact XOR solution\n")
W1 = np.array([[1., 1.], [1., 1.]]); b1 = np.array([[0.], [-1.]])
W2 = np.array([[1., -2.]]);          b2 = np.array([[0.]])
for j in range(m):                                   # one sample at a time
    x = X[:, [j]]
    z1 = W1 @ x + b1; a1 = relu(z1); yhat = W2 @ a1 + b2
    print(f"x_{j+1} = {x.ravel()}  z1 = {z1.ravel()}  a1 = {a1.ravel()}  yhat = {yhat.ravel()}")
print()
Z1, A1, Yhat = forward(W1, b1, W2, b2, X)
show("b1 1^T", b1 @ ones.T); show("Z1 = W1 X + b1 1^T", Z1); show("A1 = ReLU(Z1)", A1)
show("Yhat = W2 A1 + b2 1^T", Yhat)
print("loss (sum, 1/2) =", loss(W1, b1, W2, b2), "\n")

# ---------------------------------------------------------------- Part 2
print("=" * 70, "\nPART 2: one gradient-descent step from a PERTURBED start (W2 = [1 -1])\n")
W1 = np.array([[1., 1.], [1., 1.]]); b1 = np.array([[0.], [-1.]])
W2 = np.array([[1., -1.]]);          b2 = np.array([[0.]])
eta = 0.1
Z1, A1, Yhat = forward(W1, b1, W2, b2, X)
show("Z1", Z1); show("A1 = ReLU(Z1)", A1); show("Yhat", Yhat); show("Yhat - Y", Yhat - Y)
l0 = loss(W1, b1, W2, b2); print("loss before =", l0, "\n")

D2 = Yhat - Y                                        # Delta^(2) (1x4): dl/dZ2, linear output
dW2 = D2 @ A1.T                                      # 1x2
db2 = D2 @ ones                                      # 1x1  (= sum of columns)
B = W2.T @ D2                                        # (W2)^T Delta^(2), 2x4
D1 = B * drelu(Z1)                                   # Hadamard product, 2x4
dW1 = D1 @ X.T                                       # 2x2
db1 = D1 @ ones                                      # 2x1
show("Delta2 = Yhat - Y", D2); show("dl/dW2 = Delta2 A1^T", dW2); show("dl/db2 = Delta2 1", db2)
show("(W2)^T Delta2", B); show("ReLU'(Z1)", drelu(Z1)); show("Delta1", D1)
show("dl/dW1 = Delta1 X^T", dW1); show("dl/db1 = Delta1 1", db1)

# sum of per-sample rank-1 outer products
outer_sum = np.zeros((2, 2))
for j in range(m):
    Pj = D1[:, [j]] @ X[:, [j]].T
    print(f"delta1_{j+1} x_{j+1}^T =\n{Pj}")
    outer_sum += Pj
print("sum of outer products equals Delta1 X^T:", np.allclose(outer_sum, dW1))
print("rank(dl/dW1) =", np.linalg.matrix_rank(dW1), "\n")

# finite-difference gradient check (central differences)
def fd_grad(P, h=1e-6):
    G = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + h; lp = loss(W1, b1, W2, b2)
        P[idx] = old - h; lm = loss(W1, b1, W2, b2)
        P[idx] = old; G[idx] = (lp - lm) / (2 * h)
    return G
diffs = [np.max(np.abs(fd_grad(P) - G)) for P, G in
         [(W1, dW1), (b1, db1), (W2, dW2), (b2, db2)]]
print("finite-difference check, max |analytic - numeric| for W1,b1,W2,b2:", diffs)
print("overall max abs diff =", max(diffs), "\n")

W1n = W1 - eta * dW1; b1n = b1 - eta * db1; W2n = W2 - eta * dW2; b2n = b2 - eta * db2
show("W1 new", W1n); show("b1 new", b1n); show("W2 new", W2n); show("b2 new", b2n)
Z1n, A1n, Yhatn = forward(W1n, b1n, W2n, b2n, X)
show("Z1 new", Z1n); show("A1 new", A1n); show("Yhat new", Yhatn)
l1 = loss(W1n, b1n, W2n, b2n)
print(f"loss before = {l0:.6f}, after = {l1:.6f}, decreased: {l1 < l0}\n")

# gradient check again at a generic random point (no z exactly at the ReLU kink)
rng = np.random.default_rng(0)
saved = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
W1[:] = rng.normal(size=(2, 2)); b1[:] = rng.normal(size=(2, 1))
W2[:] = rng.normal(size=(1, 2)); b2[:] = rng.normal(size=(1, 1))
Z1, A1, Yhat = forward(W1, b1, W2, b2, X)
print("random point: min |z1| =", np.min(np.abs(Z1)))
D2 = Yhat - Y; D1 = (W2.T @ D2) * drelu(Z1)
diffs = [np.max(np.abs(fd_grad(P) - G)) for P, G in
         [(W1, D1 @ X.T), (b1, D1 @ ones), (W2, D2 @ A1.T), (b2, D2 @ ones)]]
print("random-point finite-difference max abs diff =", max(diffs), "\n")
W1, b1, W2, b2 = saved

# ---------------------------------------------------------------- Part 3
print("=" * 70, "\nPART 3: single-sample slide version, x = [1 1]^T, y = 0\n")
x = np.array([[1.], [1.]]); y = np.array([[0.]])
z1 = W1 @ x + b1; a1 = relu(z1); yhat = W2 @ a1 + b2; ls = 0.5 * ((yhat - y) ** 2).item()
d2 = yhat - y; d1 = (W2.T @ d2) * drelu(z1)
show("z1", z1); show("a1", a1); show("yhat", yhat); print("loss =", ls)
show("delta2", d2); show("dl/dW2 = delta2 a1^T", d2 @ a1.T); show("(W2)^T delta2", W2.T @ d2)
show("delta1", d1); show("dl/dW1 = delta1 x^T", d1 @ x.T)
W1s = W1 - eta * (d1 @ x.T); b1s = b1 - eta * d1; W2s = W2 - eta * (d2 @ a1.T); b2s = b2 - eta * d2
yh = W2s @ relu(W1s @ x + b1s) + b2s
print("after step: yhat =", yh.ravel(), " loss =", 0.5 * ((yh - y) ** 2).item())
print("(for comparison x = [1 0]^T gives yhat =",
      (W2 @ relu(W1 @ np.array([[1.], [0.]]) + b1) + b2).ravel(), "= y, so loss 0)\n")

# ---------------------------------------------------------------- Part 4
print("=" * 70, "\nPART 4: collapse demo (identity activation) and least squares\n")
W1 = np.array([[1., 1.], [1., 1.]]); b1 = np.array([[0.], [-1.]])
W2 = np.array([[1., -2.]]);          b2 = np.array([[0.]])
two_layer = W2 @ (W1 @ X + b1 @ ones.T) + b2 @ ones.T
Wc = W2 @ W1; bc = W2 @ b1 + b2
show("W2 W1", Wc); show("W2 b1 + b2", bc)
show("two-layer identity output", two_layer); show("single affine map output", Wc @ X + bc @ ones.T)
print("identical:", np.allclose(two_layer, Wc @ X + bc @ ones.T))
print("rank(W1) =", np.linalg.matrix_rank(W1), " rank(W2 W1) =", np.linalg.matrix_rank(Wc), "\n")

Xt = np.vstack([X, ones.T])                          # [X; 1^T], 3x4
G = Xt @ Xt.T                                        # Gram matrix, 3x3
rhs = Xt @ Y.T                                       # 3x1
theta = np.linalg.solve(G, rhs)                      # normal equations
show("[X; 1^T]", Xt); show("[X; 1^T][X; 1^T]^T", G); show("[X; 1^T] y", rhs)
show("theta = [w1 w2 b]^T", theta); show("least-squares predictions", theta.T @ Xt)
print("least-squares loss (1/2 sum) =", 0.5 * np.sum((theta.T @ Xt - Y) ** 2))
