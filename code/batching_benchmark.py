"""Batching benchmark: forward pass of a 784 -> 128 -> 10 network (ReLU hidden, linear output)
for m = 1024 samples, as (a) a loop of m matrix-vector products, (b) one matrix-matrix product.
Samples are COLUMNS: X is 784 x m.  '@' is matrix multiplication.  Runs on a CPU sandbox."""
import os, platform, time, numpy as np

n0, n1, n2, m, REPS = 784, 128, 10, 1024, 7
r = np.random.default_rng(0)
W1 = r.normal(0, 1/np.sqrt(n0), (n1, n0)); b1 = np.zeros((n1, 1))
W2 = r.normal(0, 1/np.sqrt(n1), (n2, n1)); b2 = np.zeros((n2, 1))
X = r.normal(size=(n0, m)); one = np.ones((m, 1))

def loop_forward(X):
    out = np.empty((n2, X.shape[1]))
    for j in range(X.shape[1]):                     # m separate matrix-vector products per layer
        x = X[:, j:j+1]
        out[:, j:j+1] = W2 @ np.maximum(W1 @ x + b1, 0) + b2
    return out

def batch_forward(X):                               # one matrix-matrix product per layer
    return W2 @ np.maximum(W1 @ X + b1 @ one.T, 0) + b2 @ one.T

def best_time(f):
    f(X); ts = []
    for _ in range(REPS):
        t = time.perf_counter(); f(X); ts.append(time.perf_counter() - t)
    return min(ts), float(np.median(ts))

assert np.allclose(loop_forward(X), batch_forward(X))
flops = 2*n1*n0*m + 2*n2*n1*m                       # multiply-adds counted as 2 FLOPs, weights only
tl, tlm = best_time(loop_forward); tb, tbm = best_time(batch_forward)
print(f"CPU: {platform.processor() or platform.machine()}, os.cpu_count() = {os.cpu_count()}, "
      f"NumPy {np.__version__}")
print(f"network {n0}->{n1}->{n2}, m = {m}, best of {REPS} (median in brackets)")
print(f"FLOPs (2 n_out n_in m per layer): layer1 {2*n1*n0*m:,}, layer2 {2*n2*n1*m:,}, total {flops:,}")
print(f"loop of mat-vec : {tl*1e3:8.2f} ms ({tlm*1e3:.2f})  -> {flops/tl/1e9:6.2f} GFLOP/s")
print(f"one mat-mat     : {tb*1e3:8.2f} ms ({tbm*1e3:.2f})  -> {flops/tb/1e9:6.2f} GFLOP/s")
print(f"speed-up (best/best): {tl/tb:.1f}x   (median/median): {tlm/tbm:.1f}x")
print("outputs identical (allclose):", True)
