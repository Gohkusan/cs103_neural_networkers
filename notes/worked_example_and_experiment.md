# Worked running example and activation experiment (computed, not researched)

All numbers in this file were produced by running the scripts in `code/` on 2026-10-28 in a CPU sandbox (Intel Xeon @ 2.10 GHz, 2 logical cores, Python 3, NumPy 2.5.3, Matplotlib 3.11.2, OpenBLAS 0.3.34). Raw console output is saved next to the code (`*_output.txt`). Nothing below is estimated or typed in by hand without being checked by the scripts.

Notation follows handoff §2: column vectors, $\mathbf{W}^{(l)}\in\mathbb{R}^{n_l\times n_{l-1}}$, activation $\varphi$, loss $\ell$, error $\boldsymbol{\delta}^{(l)}=\partial\ell/\partial\mathbf{z}^{(l)}$, batch $\mathbf{X}\in\mathbb{R}^{n_0\times m}$ with samples as **columns**, $\mathbf{Z}^{(l)}=\mathbf{W}^{(l)}\mathbf{A}^{(l-1)}+\mathbf{b}^{(l)}\mathbf{1}^\mathsf{T}$. Capital $\boldsymbol{\Delta}^{(l)}=[\boldsymbol{\delta}^{(l)}_1\ \cdots\ \boldsymbol{\delta}^{(l)}_m]$ is the batch of error columns. In the code, `@` is matrix multiplication and `*` is used **only** for the Hadamard product $\odot$.

## Commands run

```bash
cd code
python3 worked_example.py   | tee worked_example_output.txt
python3 experiment.py       | tee experiment_output.txt      # about 10 s
python3 make_figures.py     > figures_output.txt             # writes ../figures/*.png
python3 batching_benchmark.py | tee batching_output.txt       # run 3 times, see Part B3
OPENBLAS_NUM_THREADS=1 python3 batching_benchmark.py | tee batching_output_1thread.txt
```

Files: `code/worked_example.py`, `code/experiment.py` (78 lines incl. docstring, 67 non-blank), `code/make_figures.py`, `code/batching_benchmark.py`, `code/experiment_results.json`, `figures/fig1_activations.png`, `figures/fig2_decision_boundaries.png`, `figures/fig3_loss_curves.png`.

---

# Part A: running example, 2 → 2 (ReLU) → 1 (linear output) on XOR

Data (samples as columns):

$$
\underset{2\times 4}{\mathbf{X}}=\begin{bmatrix}0&0&1&1\\0&1&0&1\end{bmatrix},\qquad
\underset{1\times 4}{\mathbf{y}^\mathsf{T}}=\begin{bmatrix}0&1&1&0\end{bmatrix},\qquad
\underset{4\times 1}{\mathbf{1}}=\begin{bmatrix}1&1&1&1\end{bmatrix}^\mathsf{T}.
$$

Conventions fixed for all of Part A:

* Output layer is linear: $\hat{y}=z^{(2)}$, so $\varphi'=1$ at the output.
* **Batch loss is a sum, no $1/m$ averaging:** $\ell=\tfrac12\sum_{j=1}^{4}(\hat y_j-y_j)^2$. Per-sample loss $\ell_j=\tfrac12(\hat y_j-y_j)^2$, so $\ell=\sum_j \ell_j$ and every batch gradient is the sum of the per-sample gradients.
* $\mathrm{ReLU}'(0):=0$ (the convention used by the code; it is a choice, since ReLU has no derivative at 0).

## A1. Goodfellow et al. exact solution (Deep Learning, §6.1)

$$
\underset{2\times2}{\mathbf{W}^{(1)}}=\begin{bmatrix}1&1\\1&1\end{bmatrix},\quad
\underset{2\times1}{\mathbf{b}^{(1)}}=\begin{bmatrix}0\\-1\end{bmatrix},\quad
\underset{1\times2}{\mathbf{W}^{(2)}}=\begin{bmatrix}1&-2\end{bmatrix},\quad
\underset{1\times1}{b^{(2)}}=0 .
$$

One sample at a time, $\mathbf{z}^{(1)}=\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)}$ (2×1), $\mathbf{a}^{(1)}=\mathrm{ReLU}(\mathbf{z}^{(1)})$ (2×1), $\hat y=\mathbf{W}^{(2)}\mathbf{a}^{(1)}+b^{(2)}$ (1×1):

| $\mathbf{x}$ | $\mathbf{z}^{(1)}$ | $\mathbf{a}^{(1)}$ | $\hat y$ | $y$ |
|---|---|---|---|---|
| $[0\ 0]^\mathsf{T}$ | $[0\ \ {-1}]^\mathsf{T}$ | $[0\ 0]^\mathsf{T}$ | 0 | 0 |
| $[0\ 1]^\mathsf{T}$ | $[1\ 0]^\mathsf{T}$ | $[1\ 0]^\mathsf{T}$ | 1 | 1 |
| $[1\ 0]^\mathsf{T}$ | $[1\ 0]^\mathsf{T}$ | $[1\ 0]^\mathsf{T}$ | 1 | 1 |
| $[1\ 1]^\mathsf{T}$ | $[2\ 1]^\mathsf{T}$ | $[2\ 1]^\mathsf{T}$ | 0 | 0 |

As one batch (bias broadcast is the outer product $\mathbf{b}^{(1)}\mathbf{1}^\mathsf{T}$):

$$
\underset{2\times4}{\mathbf{b}^{(1)}\mathbf{1}^\mathsf{T}}=\begin{bmatrix}0&0&0&0\\-1&-1&-1&-1\end{bmatrix},\qquad
\underset{2\times4}{\mathbf{Z}^{(1)}}=\underset{2\times2}{\mathbf{W}^{(1)}}\underset{2\times4}{\mathbf{X}}+\mathbf{b}^{(1)}\mathbf{1}^\mathsf{T}=\begin{bmatrix}0&1&1&2\\-1&0&0&1\end{bmatrix},
$$

$$
\underset{2\times4}{\mathbf{A}^{(1)}}=\mathrm{ReLU}(\mathbf{Z}^{(1)})=\begin{bmatrix}0&1&1&2\\0&0&0&1\end{bmatrix},\qquad
\underset{1\times4}{\hat{\mathbf{y}}^\mathsf{T}}=\underset{1\times2}{\mathbf{W}^{(2)}}\underset{2\times4}{\mathbf{A}^{(1)}}+b^{(2)}\mathbf{1}^\mathsf{T}=\begin{bmatrix}0&1&1&0\end{bmatrix}.
$$

Observed loss: $\ell=0$ exactly. The four columns of the batch result equal the four single-sample results.

## A2. One gradient-descent step from a perturbed start

Perturbation: only the output weights change, $\mathbf{W}^{(2)}=[1\ \ {-1}]$ (instead of $[1\ \ {-2}]$); $\mathbf{W}^{(1)},\mathbf{b}^{(1)},b^{(2)}$ as in A1. Learning rate $\eta=0.1$.
Both hidden units are active (positive $z$) on at least one sample: unit 1 on samples 2, 3, 4; unit 2 on sample 4.

**Forward pass.** $\mathbf{Z}^{(1)}$ and $\mathbf{A}^{(1)}$ are unchanged from A1 (layer 1 is unchanged), and

$$
\hat{\mathbf{y}}^\mathsf{T}=\begin{bmatrix}1&-1\end{bmatrix}\begin{bmatrix}0&1&1&2\\0&0&0&1\end{bmatrix}=\begin{bmatrix}0&1&1&1\end{bmatrix},\qquad
\hat{\mathbf{y}}^\mathsf{T}-\mathbf{y}^\mathsf{T}=\begin{bmatrix}0&0&0&1\end{bmatrix},\qquad
\ell=\tfrac12\,(1)^2=0.5 .
$$

Only sample 4 ($\mathbf{x}=[1\ 1]^\mathsf{T}$, $y=0$, $\hat y=1$) is wrong.

**Output error** (linear output, squared loss, so $\boldsymbol\delta^{(2)}=\hat y-y$ per sample):

$$
\underset{1\times4}{\boldsymbol{\Delta}^{(2)}}=\hat{\mathbf{y}}^\mathsf{T}-\mathbf{y}^\mathsf{T}=\begin{bmatrix}0&0&0&1\end{bmatrix}.
$$

**Output-layer gradients:**

$$
\underset{1\times2}{\frac{\partial\ell}{\partial\mathbf{W}^{(2)}}}=\underset{1\times4}{\boldsymbol{\Delta}^{(2)}}\underset{4\times2}{\big(\mathbf{A}^{(1)}\big)^\mathsf{T}}=\begin{bmatrix}2&1\end{bmatrix},\qquad
\underset{1\times1}{\frac{\partial\ell}{\partial b^{(2)}}}=\boldsymbol{\Delta}^{(2)}\mathbf{1}=1 .
$$

**Hidden error** $\boldsymbol{\Delta}^{(1)}=\big[(\mathbf{W}^{(2)})^\mathsf{T}\boldsymbol{\Delta}^{(2)}\big]\odot\mathrm{ReLU}'(\mathbf{Z}^{(1)})$:

$$
\underset{2\times4}{\big(\mathbf{W}^{(2)}\big)^\mathsf{T}\boldsymbol{\Delta}^{(2)}}=\begin{bmatrix}1\\-1\end{bmatrix}\begin{bmatrix}0&0&0&1\end{bmatrix}=\begin{bmatrix}0&0&0&1\\0&0&0&-1\end{bmatrix},\qquad
\underset{2\times4}{\mathrm{ReLU}'(\mathbf{Z}^{(1)})}=\begin{bmatrix}0&1&1&1\\0&0&0&1\end{bmatrix},
$$

$$
\underset{2\times4}{\boldsymbol{\Delta}^{(1)}}=\begin{bmatrix}0&0&0&1\\0&0&0&-1\end{bmatrix}.
$$

Note on $\mathrm{ReLU}'(0)$: hidden unit 2 has $z=0$ exactly on samples 2 and 3 (and unit 1 on sample 1). With the convention $\mathrm{ReLU}'(0)=0$ those entries of $\mathrm{ReLU}'(\mathbf{Z}^{(1)})$ are 0. In this particular example the choice does not change any gradient, because $\boldsymbol\Delta^{(2)}$ is 0 on samples 1–3, so the Hadamard product is 0 there either way.

**Hidden-layer gradients:**

$$
\underset{2\times2}{\frac{\partial\ell}{\partial\mathbf{W}^{(1)}}}=\underset{2\times4}{\boldsymbol{\Delta}^{(1)}}\underset{4\times2}{\mathbf{X}^\mathsf{T}}=\begin{bmatrix}1&1\\-1&-1\end{bmatrix},\qquad
\underset{2\times1}{\frac{\partial\ell}{\partial\mathbf{b}^{(1)}}}=\boldsymbol{\Delta}^{(1)}\mathbf{1}=\begin{bmatrix}1\\-1\end{bmatrix}.
$$

As a sum of per-sample rank-1 outer products, $\boldsymbol\Delta^{(1)}\mathbf{X}^\mathsf{T}=\sum_{j=1}^{4}\boldsymbol\delta^{(1)}_j\mathbf{x}_j^\mathsf{T}$:

$$
\boldsymbol\delta^{(1)}_1\mathbf{x}_1^\mathsf{T}=\boldsymbol\delta^{(1)}_2\mathbf{x}_2^\mathsf{T}=\boldsymbol\delta^{(1)}_3\mathbf{x}_3^\mathsf{T}=\begin{bmatrix}0&0\\0&0\end{bmatrix},\qquad
\boldsymbol\delta^{(1)}_4\mathbf{x}_4^\mathsf{T}=\begin{bmatrix}1\\-1\end{bmatrix}\begin{bmatrix}1&1\end{bmatrix}=\begin{bmatrix}1&1\\-1&-1\end{bmatrix}.
$$

The script confirms the sum equals $\boldsymbol\Delta^{(1)}\mathbf{X}^\mathsf{T}$ (`True`) and that $\mathrm{rank}(\partial\ell/\partial\mathbf{W}^{(1)})=1$ (only one sample contributes here; in general a batch gradient is a sum of $m$ rank-1 terms and can have higher rank).

**Gradient check (central finite differences, $h=10^{-6}$).** Max $|\text{analytic}-\text{numeric}|$ observed:

| parameter | $\mathbf{W}^{(1)}$ | $\mathbf{b}^{(1)}$ | $\mathbf{W}^{(2)}$ | $b^{(2)}$ |
|---|---|---|---|---|
| max abs diff | $2.50\times10^{-7}$ | $5.00\times10^{-7}$ | $8.1\times10^{-11}$ | $1.0\times10^{-12}$ |

Because some $z$ values sit exactly at the ReLU kink, the check was repeated at a random parameter point (seed 0, smallest $|z^{(1)}|=0.36$, so no kink is crossed): max abs diff $=1.24\times10^{-10}$. (The $\sim10^{-7}$ differences at the kink point are consistent with a finite step crossing $z=0$ on samples where the residual is zero; they are still small.)

**Gradient-descent update** $\boldsymbol\theta\leftarrow\boldsymbol\theta-\eta\,\partial\ell/\partial\boldsymbol\theta$, $\eta=0.1$:

$$
\mathbf{W}^{(1)}\leftarrow\begin{bmatrix}1&1\\1&1\end{bmatrix}-0.1\begin{bmatrix}1&1\\-1&-1\end{bmatrix}=\begin{bmatrix}0.9&0.9\\1.1&1.1\end{bmatrix},\qquad
\mathbf{b}^{(1)}\leftarrow\begin{bmatrix}0\\-1\end{bmatrix}-0.1\begin{bmatrix}1\\-1\end{bmatrix}=\begin{bmatrix}-0.1\\-0.9\end{bmatrix},
$$

$$
\mathbf{W}^{(2)}\leftarrow\begin{bmatrix}1&-1\end{bmatrix}-0.1\begin{bmatrix}2&1\end{bmatrix}=\begin{bmatrix}0.8&-1.1\end{bmatrix},\qquad
b^{(2)}\leftarrow 0-0.1\,(1)=-0.1 .
$$

**Forward pass after the step:**

$$
\mathbf{Z}^{(1)}=\begin{bmatrix}-0.1&0.8&0.8&1.7\\-0.9&0.2&0.2&1.3\end{bmatrix},\quad
\mathbf{A}^{(1)}=\begin{bmatrix}0&0.8&0.8&1.7\\0&0.2&0.2&1.3\end{bmatrix},\quad
\hat{\mathbf{y}}^\mathsf{T}=\begin{bmatrix}-0.1&0.32&0.32&-0.17\end{bmatrix}.
$$

**Loss before vs after:** $\ell=0.500000\ \to\ 0.481850$ (decreased, verified by the script). By hand: residuals $-0.1,\,-0.68,\,-0.68,\,-0.17$; $\tfrac12(0.01+0.4624+0.4624+0.0289)=0.48185$.

Honest observation for the slides: the total loss went down only slightly. Sample 4 improved a lot ($\hat y$: $1\to-0.17$), but samples 2 and 3, which were perfect, got worse ($\hat y$: $1\to0.32$), because they share the weights that the step changed. One step of gradient descent decreases the **total** loss, not every per-sample loss.

## A3. Single-sample version for slides ($\mathbf{x}=[1\ 1]^\mathsf{T}$, $y=0$)

($\mathbf{x}=[1\ 0]^\mathsf{T}$ was considered but gives $\hat y=1=y$, i.e. zero loss and all-zero gradients at this start, so it is useless for showing a step. Sample 4 gives integer numbers throughout.) Same start as A2, $\eta=0.1$, $\ell=\tfrac12(\hat y-y)^2$.

| quantity | value | shape |
|---|---|---|
| $\mathbf{z}^{(1)}=\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)}$ | $[2\ 1]^\mathsf{T}$ | 2×1 |
| $\mathbf{a}^{(1)}=\mathrm{ReLU}(\mathbf{z}^{(1)})$ | $[2\ 1]^\mathsf{T}$ | 2×1 |
| $\hat y=\mathbf{W}^{(2)}\mathbf{a}^{(1)}+b^{(2)}$ | $1$ | 1×1 |
| $\ell$ | $0.5$ | scalar |
| $\delta^{(2)}=\hat y-y$ | $1$ | 1×1 |
| $\partial\ell/\partial\mathbf{W}^{(2)}=\delta^{(2)}(\mathbf{a}^{(1)})^\mathsf{T}$ | $[2\ 1]$ | 1×2 |
| $(\mathbf{W}^{(2)})^\mathsf{T}\delta^{(2)}$ | $[1\ \ {-1}]^\mathsf{T}$ | 2×1 |
| $\mathrm{ReLU}'(\mathbf{z}^{(1)})$ | $[1\ 1]^\mathsf{T}$ | 2×1 |
| $\boldsymbol\delta^{(1)}$ | $[1\ \ {-1}]^\mathsf{T}$ | 2×1 |
| $\partial\ell/\partial\mathbf{W}^{(1)}=\boldsymbol\delta^{(1)}\mathbf{x}^\mathsf{T}$ | $\begin{bmatrix}1&1\\-1&-1\end{bmatrix}$ | 2×2 (rank 1) |
| $\partial\ell/\partial\mathbf{b}^{(1)}=\boldsymbol\delta^{(1)}$, $\partial\ell/\partial b^{(2)}=\delta^{(2)}$ | $[1\ \ {-1}]^\mathsf{T}$, $1$ | |

Because only sample 4 had nonzero error in A2, the single-sample gradients here are **identical** to the batch gradients of A2, so the updated parameters are the same as in A2. After the step, for this sample: $\hat y=-0.17$, $\ell=0.01445$ (from 0.5).

## A4. Collapse demo (identity activation) and best linear fit

With $\varphi=$ identity and the A1 weights:

$$
\underset{1\times2}{\mathbf{W}^{(2)}\mathbf{W}^{(1)}}=\begin{bmatrix}1&-2\end{bmatrix}\begin{bmatrix}1&1\\1&1\end{bmatrix}=\begin{bmatrix}-1&-1\end{bmatrix},\qquad
\underset{1\times1}{\mathbf{W}^{(2)}\mathbf{b}^{(1)}+b^{(2)}}=\begin{bmatrix}1&-2\end{bmatrix}\begin{bmatrix}0\\-1\end{bmatrix}+0=2 .
$$

Two-layer output $\mathbf{W}^{(2)}\big(\mathbf{W}^{(1)}\mathbf{X}+\mathbf{b}^{(1)}\mathbf{1}^\mathsf{T}\big)+b^{(2)}\mathbf{1}^\mathsf{T}=[2\ 1\ 1\ 0]$, and the single affine map $(\mathbf{W}^{(2)}\mathbf{W}^{(1)})\mathbf{X}+(\mathbf{W}^{(2)}\mathbf{b}^{(1)}+b^{(2)})\mathbf{1}^\mathsf{T}=[2\ 1\ 1\ 0]$: identical (script: `True`). Side fact the script printed: $\mathrm{rank}(\mathbf{W}^{(1)})=1$ and $\mathrm{rank}(\mathbf{W}^{(2)}\mathbf{W}^{(1)})=1$ (the Goodfellow $\mathbf{W}^{(1)}$ is itself rank 1; ReLU is what lets the two identical rows do different jobs, via the different biases).

**Best affine fit to XOR (least squares, Week 10).** Augment with a row of ones, $\tilde{\mathbf{X}}=\begin{bmatrix}\mathbf{X}\\\mathbf{1}^\mathsf{T}\end{bmatrix}$ (3×4), unknown $\boldsymbol\theta=[w_1\ w_2\ b]^\mathsf{T}$, predictions $\boldsymbol\theta^\mathsf{T}\tilde{\mathbf{X}}$. Normal equations $\tilde{\mathbf{X}}\tilde{\mathbf{X}}^\mathsf{T}\boldsymbol\theta=\tilde{\mathbf{X}}\mathbf{y}$:

$$
\underset{3\times3}{\begin{bmatrix}2&1&2\\1&2&2\\2&2&4\end{bmatrix}}\boldsymbol\theta=\underset{3\times1}{\begin{bmatrix}1\\1\\2\end{bmatrix}}
\;\Longrightarrow\;
\boldsymbol\theta=\begin{bmatrix}0\\0\\0.5\end{bmatrix},\qquad
\boldsymbol\theta^\mathsf{T}\tilde{\mathbf{X}}=\begin{bmatrix}0.5&0.5&0.5&0.5\end{bmatrix},\qquad \ell=0.5 .
$$

So the best affine map predicts 0.5 for every XOR point (verified numerically). Any network with identity activations collapses to such a map, so it can do no better than this.

---

# Part B: activation-function experiment

## B1. Setup (identical for every activation)

* Network $2\to H\to1$ with $H=8$: $\mathbf{Z}^{(1)}=\mathbf{W}^{(1)}\mathbf{X}+\mathbf{b}^{(1)}\mathbf{1}^\mathsf{T}$ (8×m), $\mathbf{A}^{(1)}=\varphi(\mathbf{Z}^{(1)})$, $\mathbf{z}^{(2)\mathsf{T}}=\mathbf{W}^{(2)}\mathbf{A}^{(1)}+b^{(2)}\mathbf{1}^\mathsf{T}$ (1×m), $\hat{\mathbf{y}}^\mathsf{T}=\mathrm{logistic}(\mathbf{z}^{(2)\mathsf{T}})$, where $\mathrm{logistic}(z)=1/(1+e^{-z})$.
* Hidden $\varphi\in\{\text{identity},\ \text{logistic (sigmoid)},\ \tanh,\ \mathrm{ReLU}\}$.
* Loss: binary cross-entropy **averaged over the batch** (here we do use $1/m$): $\ell=-\tfrac1m\sum_j\big[y_j\log\hat y_j+(1-y_j)\log(1-\hat y_j)\big]$.
* Simplification used: with logistic output and cross-entropy, $\partial\ell/\partial z^{(2)}_j=\tfrac1m(\hat y_j-y_j)$, i.e. $\boldsymbol\Delta^{(2)}=\tfrac1m(\hat{\mathbf{y}}^\mathsf{T}-\mathbf{y}^\mathsf{T})$; the logistic derivative cancels. Then $\boldsymbol\Delta^{(1)}=[(\mathbf{W}^{(2)})^\mathsf{T}\boldsymbol\Delta^{(2)}]\odot\varphi'(\mathbf{Z}^{(1)})$, $\partial\ell/\partial\mathbf{W}^{(1)}=\boldsymbol\Delta^{(1)}\mathbf{X}^\mathsf{T}$, $\partial\ell/\partial\mathbf{W}^{(2)}=\boldsymbol\Delta^{(2)}(\mathbf{A}^{(1)})^\mathsf{T}$, bias gradients $=\boldsymbol\Delta\mathbf{1}$.
* Backprop verified against finite differences on the moons data for all four activations: max abs diff $6.9\times10^{-11}$ (identity), $1.1\times10^{-10}$ (logistic), $7.9\times10^{-11}$ (tanh), $9.6\times10^{-11}$ (ReLU).
* Init: $w_{ij}\sim N(0,1/n_{\text{in}})$ (std $1/\sqrt{n_{\text{in}}}$), biases 0; same seed gives the same initial weights for every activation. Full-batch gradient descent, $\eta=0.5$, 5000 epochs (each epoch = one GD step on the whole batch). Seeds 0–4.
* Data: XOR ($m=4$); two moons ($m=200$, noise std 0.15, generated with NumPy only, data seed 0; same dataset for every run). Accuracy = fraction with $(\hat y>0.5)=(y=1)$, measured on the training data (no test split; this is a demonstration of what each model *can fit*, not of generalization).

## B2. Results (observed; mean ± population std over 5 seeds, [min, max])

| dataset | hidden φ | final loss ℓ | final accuracy | epochs until ℓ < threshold, per seed 0–4 |
|---|---|---|---|---|
| XOR | identity | 0.6931 ± 0.0000 [0.6931, 0.6931] | 0.500 [0.25, 0.75] | never (thr 0.1) |
| XOR | logistic | 0.0026 ± 0.0004 [0.0021, 0.0033] | 1.000 [1.00, 1.00] | 828, 1416, 804, 655, 651 (thr 0.1) |
| XOR | tanh | 0.0005 ± 0.0001 [0.0004, 0.0006] | 1.000 [1.00, 1.00] | 167, 160, 129, 106, 105 (thr 0.1) |
| XOR | ReLU | 0.0004 ± 0.0002 [0.0002, 0.0006] | 1.000 [1.00, 1.00] | 133, 95, 123, 66, 101 (thr 0.1) |
| moons | identity | 0.2714 ± 0.0000 [0.2714, 0.2714] | 0.880 [0.880, 0.880] | never (thr 0.1) |
| moons | logistic | 0.2234 ± 0.0949 [0.0337, 0.2729] | 0.903 [0.880, 0.995] | only seed 2: 2808 (thr 0.1) |
| moons | tanh | 0.0052 ± 0.0006 [0.0048, 0.0065] | 1.000 [1.000, 1.000] | 407, 1308, 599, 751, 549 (thr 0.1) |
| moons | ReLU | 0.0972 ± 0.1091 [0.0061, 0.2422] | 0.958 [0.890, 1.000] | 510, never, 1545, 376, never (thr 0.1) |

Per-seed final values (from `experiment_results.json`):

* moons/logistic losses 0.2704, 0.2729, 0.0337, 0.2724, 0.2675; accuracies 0.88, 0.88, 0.995, 0.88, 0.88.
* moons/ReLU losses 0.0064, 0.2422, 0.0124, 0.0061, 0.2187; accuracies 1.00, 0.89, 1.00, 1.00, 0.90.
* XOR/identity accuracies 0.50, 0.50, 0.25, 0.75, 0.50 (see note below).

## B3. What the data show (and only that)

1. **Identity cannot solve XOR.** On every seed, the trained identity network output $\hat y=0.5$ for all four points and the loss sat at $0.6931\approx\ln 2$ (the cross-entropy of predicting 0.5). The collapsed weight $\mathbf{W}^{(2)}\mathbf{W}^{(1)}$ was numerically zero (entries of order $10^{-16}$). The reported "accuracy" of 0.25–0.75 is an artefact: the outputs are 0.5 up to rounding noise, so the $>0.5$ threshold breaks ties at random. The correct reading is "no decision boundary", which is how Figure 2 shows it. This matches Part A4 (best affine fit = 0.5 everywhere).
2. **Identity on two moons** reached the same loss (0.2714) and accuracy (0.88) on all seeds: a straight-line boundary, i.e. logistic regression. Moons is roughly linearly separable, so a linear model already gets 88%; the non-linear models are needed for the last 12%.
3. **All three non-linear activations solved XOR on every seed** (accuracy 1.0). Speed to ℓ < 0.1 on XOR: ReLU (66–133 epochs) and tanh (105–167) were similar; logistic was clearly slowest (651–1416). Figure 3 (left) shows logistic also has an initial plateau of a few hundred epochs.
4. **On two moons with this budget**, tanh was the only activation that reached 100% training accuracy on all 5 seeds (final loss 0.0048–0.0065). ReLU reached 100% on 3 of 5 seeds; on seeds 1 and 4 it stalled at ℓ ≈ 0.22–0.24, accuracy 0.89–0.90. Logistic stayed close to the linear solution (ℓ ≈ 0.27, acc 0.88) on 4 of 5 seeds within 5000 epochs; only seed 2 escaped (acc 0.995). I checked whether the two stalled ReLU runs had "dead" hidden units (inactive on all 200 samples): seed 1 had 1 of 8, seed 4 had 0 of 8, and seed 0 (which succeeded) also had 1 of 8, so dead units do **not** explain the stalls in this data. The cause is not established here.
5. Caveats: one learning rate (0.5), one width (H = 8), one epoch budget, one init scheme, 5 seeds. The ranking could change with other settings (e.g. logistic would probably benefit from a larger η, but that was not run). Do not present these as general claims about the activations, only as what this experiment showed.

## B4. Figures (own design, 150 dpi, Okabe–Ito / PuOr colour-blind-safe palettes, no σ anywhere)

* `figures/fig1_activations.png`: four panels (identity, logistic (sigmoid), tanh, ReLU), each with $\varphi(z)$ solid and $\varphi'(z)$ dashed on $z\in[-4,4]$. Visible: logistic′ peaks at 0.25; tanh′ peaks at 1; ReLU′ is a 0/1 step (ReLU′(0) := 0).
* `figures/fig2_decision_boundaries.png`: 2×4 grid (rows XOR, two moons; columns the four activations), seed 0, shaded by $\hat y$, black line at $\hat y=0.5$. XOR/identity panel is labelled "ŷ = 0.500 everywhere (no boundary)". Visible: logistic on XOR draws a smooth band, tanh a curved saddle shape, ReLU a piecewise-linear (polygonal) boundary; identity and logistic on moons give a straight line; tanh and ReLU wrap around the moons (ReLU with straight segments and corners).
* `figures/fig3_loss_curves.png`: two panels (XOR, moons), log y-axis, mean over 5 seeds with min–max band.

Script `make_figures.py` regenerates all three from `experiment_state.npz` saved by `experiment.py`.

## B5. Batching benchmark (CPU sandbox, not a GPU)

`batching_benchmark.py`: network 784 → 128 (ReLU) → 10 (linear), $m=1024$ random inputs as columns of $\mathbf{X}\in\mathbb{R}^{784\times1024}$. (a) Python loop of $m$ matrix–vector products per layer, $\mathbf{W}^{(1)}\mathbf{x}_j$; (b) one matrix–matrix product per layer, $\mathbf{W}^{(1)}\mathbf{X}+\mathbf{b}^{(1)}\mathbf{1}^\mathsf{T}$. Outputs checked identical (`allclose`). Time = best of 7 repetitions after a warm-up (median in brackets).

FLOP count, $2\,n_l\,n_{l-1}\,m$ per layer (each multiply–add = 2 FLOPs; bias/ReLU ignored):
layer 1: $2\cdot128\cdot784\cdot1024=205{,}520{,}896$; layer 2: $2\cdot10\cdot128\cdot1024=2{,}621{,}440$; total $\approx 2.08\times10^{8}$ FLOPs. Both methods do exactly the same arithmetic; only the organisation differs.

| run | loop of mat–vec | one mat–mat | speed-up best/best (median/median) |
|---|---|---|---|
| 1 (default threads) | 21.18 ms (27.15), 9.8 GFLOP/s | 3.52 ms (3.58), 59.1 GFLOP/s | 6.0× (7.6×) |
| 2 (default threads) | 18.03 ms (23.17), 11.5 GFLOP/s | 3.11 ms (3.16), 66.8 GFLOP/s | 5.8× (7.3×) |
| 3 (`OPENBLAS_NUM_THREADS=1`) | 19.85 ms (23.87), 10.5 GFLOP/s | 3.19 ms (3.25), 65.2 GFLOP/s | 6.2× (7.3×) |

Observed: batching gave about **6× (best-time) to 7.5× (median)** speed-up on this 2-core CPU, with the same 208 MFLOPs of work. The speed-up comes from one BLAS GEMM call reusing each weight from cache across many columns, instead of $m$ separate calls that each re-read the whole $\mathbf{W}$ (plus Python loop overhead). On a GPU the gap is typically much larger, but **that was not measured here**; say "CPU sandbox" if this number is shown.

---

## Checklist items covered by this file

- [x] Every product conformant; shapes written under/next to each matrix.
- [x] No σ for an activation (code and figures use "logistic"); no 𝓛; layer indices in parentheses.
- [x] Samples as columns throughout, including the benchmark and the decision-boundary grid.
- [x] Loss scaling stated: Part A sums (no 1/m), Part B averages (1/m).
- [x] ReLU′(0) := 0 stated, and shown not to affect the A2 numbers.
- [x] All numbers recomputed by script and consistent across A2/A3.
- [x] Experiment results are real (no placeholders).
