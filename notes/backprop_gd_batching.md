# Backpropagation as Matrix Calculus, Gradient Descent, and Batched Computation (handoff §5 P1 items 5–7)

All math is in the handoff §2 notation: column vectors; **W**^{(l)} ∈ ℝ^{n_l × n_{l-1}}; **z**^{(l)} = **W**^{(l)}**a**^{(l-1)} + **b**^{(l)}; **a**^{(l)} = φ(**z**^{(l)}); loss ℓ; **δ**^{(l)} = ∂ℓ/∂**z**^{(l)}; ⊙ Hadamard (element-wise) product; batch **X** = [**x**_1 ⋯ **x**_m] ∈ ℝ^{n_0 × m} (samples as columns). Products are written by juxtaposition only. Where a source uses other symbols (Nielsen's σ, C, w^l_{jk}; Goodfellow's J, ε, **h**), they have been converted.

**Numerical verification done for these notes.** A NumPy script (2-layer net 3→4→2, tanh hidden, sigmoid output, batch m = 5, samples as columns) compared the matrix-form gradients below with central finite differences (ε = 10^{-6}). Max abs error for ∂ℓ/∂**W**^{(1)} and ∂ℓ/∂**b**^{(1)}: about 3e-10 (BCE loss) and 6e-11 (squared loss). The script also checked that **Δ**(**A**)^T equals the sum of m per-sample outer products, and that a single-sample **δa**^T has rank 1. (These are our own checks, not citations. The script is code/backprop_gradcheck.py.)

---

## Q1. Deriving BP1–BP4 from the chain rule: why the transpose, ⊙ and the outer product appear (with a shape-check table)

### Takeaway
All four backprop equations follow from one rule: for a scalar ℓ and **u** = g(**v**), ∂ℓ/∂**v** = (J_g)^T ∂ℓ/∂**u**, where J_g is the Jacobian (the matrix of all partial derivatives ∂u_i/∂v_j). The transpose appears because the affine map's Jacobian is **W**. The ⊙ appears because an element-wise φ has a diagonal Jacobian. The outer product appears because ∂z_i/∂w_{ij} = a_j. Nielsen's matrix-form equations match ours symbol for symbol once σ → φ and C → ℓ.

### Cited Findings
- Nielsen defines the error of neuron j in layer l as δ^l_j ≡ ∂C/∂z^l_j (our **δ**^{(l)} = ∂ℓ/∂**z**^{(l)}). His weight w^l_{jk} connects neuron k in layer l−1 to neuron j in layer l, so the row index is the output neuron, the same as our **W**^{(l)} ∈ ℝ^{n_l × n_{l-1}}. — [Nielsen, NNDL Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- Nielsen's matrix forms: BP1 δ^L = ∇_a C ⊙ σ′(z^L); BP2 δ^l = ((w^{l+1})^T δ^{l+1}) ⊙ σ′(z^l); BP3 ∂C/∂b^l_j = δ^l_j; BP4 ∂C/∂w^l_{jk} = a^{l-1}_k δ^l_j, abbreviated "∂C/∂w = a_in δ_out". — [Nielsen, NNDL Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- Nielsen's two assumptions on the cost: it is an average C = (1/n) Σ_x C_x of per-example costs, and it is a function of the network outputs. — [Nielsen, NNDL Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- Goodfellow et al. (Eq. 6.46) write the vector chain rule as ∇_**x** z = (∂**y**/∂**x**)^T ∇_**y** z, where ∂**y**/∂**x** is the Jacobian of g. They add: "The back-propagation algorithm consists of performing such a Jacobian-gradient product for each operation in the graph." — [Goodfellow, Bengio & Courville, Deep Learning §6.5.2](https://www.deeplearningbook.org/contents/mlp.html). (Note: the transpose on the Jacobian was lost in the web text extraction. The form with ^T is the book's printed Eq. 6.46 and agrees with the derivation below.)
- Goodfellow et al. also state that "back-propagation refers only to the method for computing the gradient", while a separate algorithm such as SGD does the learning. — [Deep Learning §6.5](https://www.deeplearningbook.org/contents/mlp.html)
- Parr & Howard: element-wise binary operations give a Jacobian that is "very often a diagonal matrix", namely whenever f_i depends only on w_i and x_i (their "element-wise diagonal condition"). — [Parr & Howard, The Matrix Calculus You Need for Deep Learning](https://explained.ai/matrix-calculus/)
- CS231n dimension-analysis tip: "the gradient on the weights dW must be of the same size as W". For D = W X with W 5×10 and X 10×3, the backward pass is dW = dD X^T and dX = W^T dD. This uses columns as samples, exactly our convention. — [CS231n, Backprop notes (optimization-2)](https://cs231n.github.io/optimization-2/)
- J. Johnson's CS231n handout: the Jacobian of a linear layer is "typically far too large to fit in memory" (his example is about 68 billion entries, 256 GB at fp32), so implementations never form it explicitly. — [Johnson, Derivatives, Backpropagation, and Vectorization (CS231n handout)](https://cs231n.stanford.edu/handouts/derivatives.pdf)

### Derivation in our notation (verified symbolically and numerically)

**Tool: chain rule for a scalar output.** Let ℓ depend on **v** ∈ ℝ^p only through **u** = g(**v**) ∈ ℝ^q. The Jacobian J = ∂**u**/∂**v** ∈ ℝ^{q × p} has entries J_{ij} = ∂u_i/∂v_j. The scalar chain rule gives ∂ℓ/∂v_j = Σ_i (∂ℓ/∂u_i)(∂u_i/∂v_j) = Σ_i J_{ij} (∂ℓ/∂u_i). In matrix form, with gradients as column vectors:

  ∂ℓ/∂**v** = J^T ∂ℓ/∂**u**  (p×q times q×1 gives p×1)

The transpose is needed because the sum runs down the *rows* index i of J (a column of J), and a column of J is a row of J^T. This is Goodfellow's Eq. 6.46.

**Step A: Jacobians of the two building blocks.**
1. Affine map **z** = **W a** + **b**: z_i = Σ_j w_{ij} a_j + b_i, so ∂z_i/∂a_j = w_{ij}. Hence ∂**z**/∂**a** = **W** (the Jacobian of a linear map is its own matrix, Week 1).
2. Element-wise map **a** = φ(**z**): a_i = φ(z_i) depends only on z_i, so ∂a_i/∂z_j = φ′(z_i) when i = j and 0 otherwise. Hence ∂**a**/∂**z** = diag(φ′(**z**)). For any vector **v**, diag(φ′(**z**)) **v** = φ′(**z**) ⊙ **v** (a diagonal matrix scales each entry). A diagonal matrix is symmetric, so its transpose changes nothing.

**BP1 (output layer).** ℓ depends on **z**^{(L)} through **a**^{(L)} = φ(**z**^{(L)}):
  **δ**^{(L)} = (diag(φ′(**z**^{(L)})))^T ∂ℓ/∂**a**^{(L)} = ∂ℓ/∂**a**^{(L)} ⊙ φ′(**z**^{(L)}). ✔ (matches handout §2.3 and Nielsen BP1)

**BP2 (hidden layers).** ℓ depends on **z**^{(l)} through **a**^{(l)} = φ(**z**^{(l)}), then **z**^{(l+1)} = **W**^{(l+1)}**a**^{(l)} + **b**^{(l+1)}. Apply the rule twice:
  ∂ℓ/∂**a**^{(l)} = (**W**^{(l+1)})^T **δ**^{(l+1)}  (Jacobian **W**^{(l+1)}, transposed)
  **δ**^{(l)} = diag(φ′(**z**^{(l)})) ∂ℓ/∂**a**^{(l)} = [(**W**^{(l+1)})^T **δ**^{(l+1)}] ⊙ φ′(**z**^{(l)}). ✔
  Interpretation: forward, **W** maps n_{l} → n_{l+1}. Backward, **W**^T maps errors n_{l+1} → n_l. This is the mirror image from slide rule 8 of the handout.

**BP3 (bias).** z^{(l)}_i = … + b^{(l)}_i, so ∂**z**^{(l)}/∂**b**^{(l)} = **I**, and ∂ℓ/∂**b**^{(l)} = **I**^T **δ**^{(l)} = **δ**^{(l)}. ✔

**BP4 (weights): an outer product.** The weight w^{(l)}_{ij} appears only in z^{(l)}_i = Σ_k w^{(l)}_{ik} a^{(l-1)}_k + b^{(l)}_i, so ∂z^{(l)}_i/∂w^{(l)}_{ij} = a^{(l-1)}_j. Therefore
  ∂ℓ/∂w^{(l)}_{ij} = δ^{(l)}_i a^{(l-1)}_j.
The matrix whose (i, j) entry is δ_i a_j is exactly the outer product **δ**^{(l)}(**a**^{(l-1)})^T (Week 2). So ∂ℓ/∂**W**^{(l)} = **δ**^{(l)}(**a**^{(l-1)})^T. ✔ (Nielsen's "a_in δ_out" is the same entry-wise statement.)

**Rank.** Every column of **δa**^T is a multiple of **δ** (column j = a_j **δ**), so 𝒞(**δa**^T) ⊆ span{**δ**}. Hence rank ≤ 1, with equality iff **δ** ≠ **0** and **a** ≠ **0**. With ReLU, entries of **a**^{(l-1)} that are 0 give whole zero columns: weights fed by inactive neurons get no update from that sample.

**Why **δ** is used rather than ∂ℓ/∂**a**.** Defining the error at **z** turns every layer into the same two-step pattern (multiply by **W**^T, then ⊙ φ′). Every quantity it needs (**z**^{(l)}, **a**^{(l-1)}) was already computed and stored during the forward pass. This reuse is the source of the cost advantage in Q5.

### Shape-check table (single sample)

| Term | Shape | Conformance check |
|---|---|---|
| **a**^{(l-1)} | n_{l-1} × 1 | |
| **W**^{(l)} | n_l × n_{l-1} | |
| **W**^{(l)}**a**^{(l-1)} | n_l × 1 | (n_l × n_{l-1})(n_{l-1} × 1) |
| **b**^{(l)}, **z**^{(l)}, **a**^{(l)}, φ′(**z**^{(l)}) | n_l × 1 | |
| ∂**z**^{(l)}/∂**a**^{(l-1)} = **W**^{(l)} | n_l × n_{l-1} | Jacobian: rows = outputs |
| ∂**a**^{(l)}/∂**z**^{(l)} = diag(φ′(**z**^{(l)})) | n_l × n_l | diagonal, never formed in practice |
| ∂ℓ/∂**a**^{(L)} | n_L × 1 | same shape as **a**^{(L)} |
| **δ**^{(L)} = ∂ℓ/∂**a**^{(L)} ⊙ φ′(**z**^{(L)}) | n_L × 1 | ⊙ needs equal shapes: (n_L×1) ⊙ (n_L×1) |
| **δ**^{(l+1)} | n_{l+1} × 1 | |
| (**W**^{(l+1)})^T | n_l × n_{l+1} | transpose of n_{l+1} × n_l |
| (**W**^{(l+1)})^T **δ**^{(l+1)} | n_l × 1 | (n_l × n_{l+1})(n_{l+1} × 1) |
| **δ**^{(l)} = [⋯] ⊙ φ′(**z**^{(l)}) | n_l × 1 | (n_l×1) ⊙ (n_l×1) |
| (**a**^{(l-1)})^T | 1 × n_{l-1} | |
| ∂ℓ/∂**W**^{(l)} = **δ**^{(l)}(**a**^{(l-1)})^T | n_l × n_{l-1} | (n_l × 1)(1 × n_{l-1}); same shape as **W**^{(l)} ✔ |
| ∂ℓ/∂**b**^{(l)} = **δ**^{(l)} | n_l × 1 | same shape as **b**^{(l)} ✔ |
| **W**^{(l)} − η ∂ℓ/∂**W**^{(l)} | n_l × n_{l-1} | scalar times matrix, then subtract equal shapes |

Batch versions (from Q4): **A**^{(l-1)} n_{l-1}×m; **Z**^{(l)}, **Δ**^{(l)}, φ′(**Z**^{(l)}) n_l×m; **b**^{(l)}**1**^T n_l×m ((n_l×1)(1×m)); **Δ**^{(l)}(**A**^{(l-1)})^T n_l×n_{l-1} ((n_l×m)(m×n_{l-1})); **Δ**^{(l)}**1** n_l×1 ((n_l×m)(m×1)); (**W**^{(l+1)})^T**Δ**^{(l+1)} n_l×m.

### Inferences
- The "shape-only" heuristic (CS231n) is a good Q&A defence: ∂ℓ/∂**W**^{(l)} must be n_l × n_{l-1}, and the only way to build it from **δ** (n_l×1) and **a** (n_{l-1}×1) is **δa**^T. Shapes do not, however, prove the formula; the entry-wise derivation does.
- The mirror between forward and backward passes is exact: forward multiplies by **W**^{(l)} then applies φ; backward applies ⊙φ′ then multiplies by (**W**^{(l)})^T. This connects to (**AB**)^T = **B**^T**A**^T (Week 3): the Jacobian of the whole network is a product of layer Jacobians, and backprop multiplies its transpose by a vector from the left end, in reverse order.

### Gaps
- I could not retrieve the text of Goodfellow Algorithm 6.4 (the MLP backward pass: g ← g ⊙ f′(a^{(k)}), ∇_{W^{(k)}} J = g **h**^{(k-1)T}, g ← **W**^{(k)T} g) through the web extraction. Check it in the printed book (p. 206 in the MIT Press edition) before quoting it. From memory it matches BP1–BP4 plus a weight-decay term λ∇Ω; this is not verified here.

---

## Q2. Layout convention: numerator vs denominator, and the pitfall

### Takeaway
We use the **denominator (gradient) layout** for derivatives of the scalar ℓ: ∂ℓ/∂**W** has the same shape as **W**, and ∂ℓ/∂**z** is a column like **z**. Jacobians of vector functions keep the standard form (rows = outputs). Parr & Howard use **numerator layout**, where the gradient of a scalar is a *row* vector. Their gradients are therefore the transpose of ours. Do not mix formulas between the two conventions without transposing.

### Cited Findings
- Parr & Howard: "We are using the so-called numerator layout but many papers and software will use the denominator layout." Denominator layout is "just transpose of the numerator layout Jacobian (flip it around its diagonal)." — [Parr & Howard](https://explained.ai/matrix-calculus/)
- In Parr & Howard the Jacobian is built "where the gradients are rows", and "the Jacobian is always m rows for m equations". Partials of a scalar function are organised "into a horizontal vector". In the sum-reduction example they stress: "the result is a horizontal vector full of 1s, not a vertical vector." — [Parr & Howard](https://explained.ai/matrix-calculus/)
- Parr & Howard: "It's very important to keep the shape of all of your vectors and matrices in order otherwise it's impossible to compute the derivatives of complex functions." — [Parr & Howard](https://explained.ai/matrix-calculus/)
- Johnson: a gradient ∂L/∂x formally has shape (1)×(N×D), but "as is typical for representing gradients we instead view ∂L/∂x as a matrix of shape N × D". This is the same-shape-as-variable convention we adopt. — [Johnson, CS231n handout](https://cs231n.stanford.edu/handouts/derivatives.pdf)
- CS231n: dW "must be of the same size as W". — [CS231n optimization-2](https://cs231n.github.io/optimization-2/)

### Inferences
- **Our convention, stated for the slide:** "∂ℓ/∂**W**^{(l)} is the matrix of partials [∂ℓ/∂w_{ij}], the same shape as **W**^{(l)}; Jacobians ∂**u**/∂**v** have one row per output." Under this mixed convention the scalar chain rule carries an explicit transpose (∂ℓ/∂**v** = J^T ∂ℓ/∂**u**). In pure numerator layout it reads ∂ℓ/∂**v** = (∂ℓ/∂**u**) J, with row vectors and no transpose. The two are transposes of each other and fully consistent.
- **Pitfall 1:** taking a Parr & Howard result such as ∂(**w**^T**x**)/∂**w** = **x**^T (a row) and using it as a column gradient gives a shape mismatch. Transpose it first.
- **Pitfall 2 (rows-as-samples sources):** Johnson's handout writes the layer as y = xW with x ∈ ℝ^{N×D}, rows as samples. Its results are ∂L/∂x = (∂L/∂y)W^T and ∂L/∂W = x^T(∂L/∂y). Our columns-as-samples forms are (**W**)^T**Δ** and **Δ**(**A**)^T: transpose every term and reverse the product order. Do not mix the two.
- Johnson's printed Eq. 12 reads "∂L/∂x = ∂L/∂y x^T", which the fetched text's own element-wise derivation contradicts. The derivation gives ∂L/∂y W^T. This looks like a typo in the handout, so do not cite Eq. 12 verbatim. — [Johnson handout](https://cs231n.stanford.edu/handouts/derivatives.pdf) (flagged by my reading of the extracted text; check the PDF directly).

### Gaps
- Parr & Howard's own linear-layer derivations (activation(**x**) = max(0, **w**·**x** + b)) use the dot-product symbol; any formula taken from them must be rewritten as **w**^T**x** to meet handout §1.

---

## Q3. Loss choices: squared error and binary cross-entropy with a logistic output

### Takeaway
For squared error, ∂ℓ/∂**ŷ** = **ŷ** − **y**, so **δ**^{(L)} = (**ŷ** − **y**) ⊙ φ′(**z**^{(L)}). For BCE with a sigmoid output, the φ′ factor cancels exactly and **δ**^{(L)} = **ŷ** − **y**. Both were verified by hand and by the numerical gradient check.

### Cited Findings
- Nielsen: quadratic cost C_x = ½‖**y** − **a**^L‖², with "∇_a C = (a^L − y)". — [Nielsen Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- Nielsen gives the full cost as C = (1/2n) Σ_x ‖**y**(**x**) − **a**^L(**x**)‖², an average over training examples. — [Nielsen Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)

### Derivations (ours; verified)
- **Squared error.** ℓ = ½‖**ŷ** − **y**‖² = ½(**ŷ** − **y**)^T(**ŷ** − **y**). Then ∂ℓ/∂ŷ_i = ŷ_i − y_i, so ∂ℓ/∂**ŷ** = **ŷ** − **y** and **δ**^{(L)} = (**ŷ** − **y**) ⊙ φ′(**z**^{(L)}). With identity output (φ′ = 1) this is just **ŷ** − **y**. That is the regression bridge: for a single linear layer **ŷ** = **W x**, the gradient is (**Wx** − **y**)**x**^T, which is the per-sample form of the normal-equation gradient (Ch. 10).
- **BCE + sigmoid.** Write s(z) = 1/(1 + e^{−z}) and avoid σ, per the handout. Per output unit, ℓ = −[y ln ŷ + (1 − y) ln(1 − ŷ)] with ŷ = s(z). Then ∂ℓ/∂ŷ = −y/ŷ + (1 − y)/(1 − ŷ) = (ŷ − y)/(ŷ(1 − ŷ)) and s′(z) = ŷ(1 − ŷ). Their product is δ = ŷ − y. For a vector of independent sigmoid outputs, **δ**^{(L)} = **ŷ** − **y**. ✔
- Consequence: with squared error plus a sigmoid output, δ carries the factor s′(z) ≤ 1/4. Learning slows when the output saturates (ŷ near 0 or 1) even if it is badly wrong. With BCE that factor is gone. (The same conclusion is in Nielsen Ch. 3 on cross-entropy; that chapter was not fetched here, so cite it only after checking.)
- The same cancellation holds for softmax plus categorical cross-entropy (**δ**^{(L)} = **ŷ** − **y** for one-hot **y**). This is standard, but no source was fetched for it here, so treat it as uncited.

### Inferences
- In code, δ^{(L)} = **ŷ** − **y** is the same expression for "identity output + squared error" and "sigmoid output + BCE". This is a good Q&A point: they are different models with the same output error signal.

### Gaps
- Nielsen Ch. 3 (cross-entropy) was not fetched, so there is no direct source quote for the BCE cancellation. The algebra is above and the numerical check passed.

---

## Q4. Batch version: Z = W A + b 1^T, ∂ℓ/∂W = Δ A^T as a sum of m rank-1 outer products, ∂ℓ/∂b = Δ 1, and averaging

### Takeaway
Putting the m samples side by side as columns turns every matrix–vector product into one matrix–matrix product. The weight gradient **Δ**^{(l)}(**A**^{(l-1)})^T is exactly the sum of the m per-sample outer products (the column-row expansion of a matrix product). The bias gradient is **Δ1** (row sums). With the mean loss ℓ = (1/m)Σ_j ℓ_j, the 1/m is applied once, conventionally inside **Δ**^{(L)}.

### Cited Findings
- Nielsen's mini-batch update: **W**^l → **W**^l − (η/m) Σ_x **δ**^{x,l}(**a**^{x,l−1})^T and **b**^l → **b**^l − (η/m) Σ_x **δ**^{x,l}: the average of per-sample outer products. — [Nielsen Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- Nielsen: per-example gradients are averaged, "We then recover ∂C/∂w and ∂C/∂b by averaging over training examples." — [Nielsen Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- CS231n gives the batched backward rule in our columns-as-samples layout: for D = W X, dW = dD X^T and dX = W^T dD. — [CS231n optimization-2](https://cs231n.github.io/optimization-2/)
- NVIDIA maps the three training GEMMs (general matrix-matrix multiply) of a fully connected layer as follows. Forward: M = outputs, N = batch size, K = inputs. Activation gradient: M = inputs, N = batch, K = outputs. Weight gradient: M = inputs, N = outputs, K = batch size. The batch size is the *inner* (summed) dimension of the weight-gradient GEMM, which is the sum over m outer products. — [NVIDIA, Linear/Fully-Connected Layers Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html)

### Batched equations (ours; verified numerically)
With **A**^{(0)} = **X** ∈ ℝ^{n_0×m} and **1** ∈ ℝ^m:
- Forward: **Z**^{(l)} = **W**^{(l)}**A**^{(l-1)} + **b**^{(l)}**1**^T, **A**^{(l)} = φ(**Z**^{(l)}). Here **b1**^T = [**b** **b** ⋯ **b**] is an outer product that copies **b** into every column. This is the matrix form of NumPy "broadcasting". Column j of **Z** is **W a**_j + **b**, the single-sample forward pass (column picture of matrix multiplication).
- Loss: ℓ = (1/m)Σ_j ℓ_j. Define **Δ**^{(l)} = ∂ℓ/∂**Z**^{(l)} ∈ ℝ^{n_l×m}. Column j is (1/m)**δ**_j^{(l)}.
- Output: **Δ**^{(L)} = (1/m)(**Ŷ** − **Y**) for BCE+sigmoid. For squared error with any φ it is (1/m)(**Ŷ** − **Y**) ⊙ φ′(**Z**^{(L)}).
- Hidden: **Δ**^{(l)} = [(**W**^{(l+1)})^T**Δ**^{(l+1)}] ⊙ φ′(**Z**^{(l)}). This is column-wise BP2, since ⊙ and left-multiplication both act column by column.
- Weights: ∂ℓ/∂**W**^{(l)} = **Δ**^{(l)}(**A**^{(l-1)})^T = Σ_{j=1}^{m} **Δ**^{(l)}_{:,j}(**a**_j^{(l-1)})^T = (1/m)Σ_j **δ**_j^{(l)}(**a**_j^{(l-1)})^T. ✔ Here **Δ**_{:,j} is column j of **Δ**. The identity **PQ**^T = Σ_j **p**_j**q**_j^T is the column-row (outer-product) expansion of a matrix product from Ch. 2. A sum of m rank-1 matrices has rank ≤ min(m, n_l, n_{l-1}).
- Bias: ∂ℓ/∂**b**^{(l)} = **Δ**^{(l)}**1** = Σ_j **Δ**^{(l)}_{:,j} (row sums). ✔ Why: **b** enters through **b1**^T, and the transpose of "copy into m columns" is "add up the m columns".
- **Averaging decision for the team:** use the mean loss and put the 1/m in **Δ**^{(L)}. Then **W** ← **W** − η**Δ**(**A**)^T matches Nielsen's (η/m)Σ form exactly. If the sum loss is used instead, η must shrink by a factor of m to give the same step. Mean is preferred because the step size then does not depend on batch size.

### Inferences
- The fact that "the batch dimension is summed out in the weight gradient" explains NVIDIA's table: m is the K dimension of the weight-gradient GEMM.
- The batched equations are not an approximation. They give exactly the mean of the per-sample gradients, rearranged into level-3 BLAS form (see Q6).

### Gaps
- None material. All batch identities were verified numerically (max error about 1e-10).

---

## Q5. Gradient descent: update rule, why −∇ is the steepest-descent direction, the role of η, mini-batch SGD, and the cost of computing the gradient

### Takeaway
**W** ← **W** − η ∂ℓ/∂**W**. The negative gradient is the steepest-descent direction because, to first order, the change in ℓ for a unit step **u** is **u**^T**g**. By Cauchy–Schwarz this is ≥ −‖**g**‖, with equality only at **u** = −**g**/‖**g**‖. η must be small enough that the first-order Taylor model stays valid; the curvature (Hessian eigenvalues) sets the scale. Reverse mode (backprop) gives the full gradient of a scalar loss for a small constant multiple of one forward pass. Baydin et al., citing Griewank & Walther, give c < 6, typically 2–3. Finite differences need about one extra forward pass per parameter.

### Cited Findings
- Goodfellow et al. §4.3: "To minimize f, we would like to find the direction in which f decreases the fastest." Minimising the directional derivative reduces to minimising cos θ, which happens when "**u** points in the opposite direction as the gradient". So "the gradient points directly uphill, and the negative gradient points directly downhill." — [Deep Learning Ch. 4 (Numerical Computation) §4.3](https://www.deeplearningbook.org/contents/numerical.html)
- Update **x**′ = **x** − ε∇_**x** f(**x**), "where ε is the learning rate, a positive scalar determining the size of the step". "A popular approach is to set ε to a small constant." Line search instead evaluates f(**x** − ε∇f) for several ε and keeps the best one. — [Deep Learning §4.3](https://www.deeplearningbook.org/contents/numerical.html)
- Second-order Taylor analysis: when the curvature term **g**^T**Hg** is large, "the gradient descent step can actually move uphill". For **g**^T**Hg** > 0 the optimal step is ε* = **g**^T**g** / **g**^T**Hg**, which in the worst case is 1/λ_max. The Hessian's eigenvalues therefore set the learning-rate scale. — [Deep Learning §4.3](https://www.deeplearningbook.org/contents/numerical.html) (bridge to handout P2 item 8 and Ch. 11)
- Mini-batch SGD: Nielsen updates with the mini-batch average, **W** → **W** − (η/m)Σ_x **δ**^x(**a**^x)^T. — [Nielsen Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- **Cost: reverse mode vs forward mode.** Baydin et al. 2018 (JMLR 18:153), §3.2: "if we denote the operation count to evaluate the original function by ops(f), the time it takes to calculate the m × n Jacobian by the forward mode is n c ops(f), whereas the same computation can be done via reverse mode in m c ops(f), where c is a constant guaranteed to be c < 6 and typically c ∼ [2, 3] (Griewank and Walther, 2008)." For a scalar loss (m = 1 output), the gradient over all n parameters costs c ops(f). — [Baydin, Pearlmutter, Radul & Siskind, "Automatic Differentiation in Machine Learning: a Survey", JMLR 2018](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)
- Baydin et al. §3.1: for f: ℝ^n → ℝ, "forward mode AD requires n evaluations to compute the gradient". §2.1: finite differences need "O(n) evaluations of f for a gradient in n dimensions". §4.2: "the backpropagation algorithm is only a special case of AD", and reverse mode "generalizes the backpropagation idea" (§1). — [Baydin et al. 2018](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)
- Griewank (2012) states the cheap gradient result as OPS{∇f(**x**)} ≤ ω OPS{f(**x**)}, where ω depends on the complexity measure and platform (e.g. ω = 3 when counting multiplications for polynomial operations). He cites Wolfe's observation that the ratio is "usually 1.5" with care. The gradient costs "a small multiple of the cost of computing the underlying function" regardless of the number of variables. Caveat: the bound ignores memory, since reverse mode stores forward-pass intermediates (checkpointing trades memory for recomputation). — [Griewank, "Who Invented the Reverse Mode of Differentiation?", Documenta Mathematica Extra Vol. ISMP (2012) 389–400](https://papers.baulab.info/papers/also/Griewank-2012.pdf)
- Baydin et al. use the bound c < 6 rather than naming the "cheap gradient principle". A later paper describes both AD modes' cost as "bounded by a constant multiple of the time it takes to run the function f" (citing Griewank & Walther 2008), and notes that in practice the reverse-mode factor R_b "is typically between 5 and 10" (citing Hascoët 2014), and is "highly program dependent". — [Baydin et al., "Gradients without Backpropagation", 2022, §2.3](https://arxiv.org/pdf/2202.08587)
- Nielsen's version for neural networks: with "a million weights", finite differences need "a million forward passes through the network (per training example)", while "the computational cost of the backward pass is about the same as the forward pass", so backprop costs about two forward passes in total. — [Nielsen Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- Goodfellow et al. §6.5.3: back-propagation computation "scales linearly with the number of edges in G", with one partial derivative, one multiplication and one addition per edge. — [Deep Learning §6.5](https://www.deeplearningbook.org/contents/mlp.html)

### Steepest descent proof (ours, Week 2 tools only)
First-order Taylor: ℓ(**θ** + t**u**) ≈ ℓ(**θ**) + t **u**^T**g**, with **g** = ∇ℓ(**θ**) and ‖**u**‖ = 1. By Cauchy–Schwarz, **u**^T**g** ≥ −‖**u**‖‖**g**‖ = −‖**g**‖, with equality iff **u** = −**g**/‖**g**‖. For a matrix parameter, apply the same argument with the Frobenius inner product trace(**U**^T**G**) = Σ_{ij} u_{ij}g_{ij}. That is why the update subtracts a matrix of the same shape as **W**. For a step η**g** the predicted decrease is η‖**g**‖² > 0 when **g** ≠ **0**, but only for η small enough that the higher-order terms are negligible (Goodfellow's "move uphill" warning).

### Recommended wording for the cost claim (handout §8 checklist)
"For a scalar loss, backpropagation (reverse-mode differentiation) computes the gradient with respect to *all* parameters at a cost that is a small constant multiple of one forward pass. Baydin et al. (2018), citing Griewank & Walther, give a guaranteed constant below 6, typically 2–3. Finite differences or forward-mode differentiation need on the order of one forward pass *per parameter*. For our 784→128→10 example that is 101,770 passes versus about 3." The memory caveat should go in the speaker notes.

### Inferences
- The c ≈ 2–3 figure is consistent with our FLOP count (Q6): backward GEMMs ≈ 1–2× forward GEMMs, so forward + backward ≈ 2–3× forward.
- Without computing dX (the gradient w.r.t. the input, which is not needed for training) the backward pass of our MLP is ≈ 1.01× the forward FLOPs. With dX it would be ≈ 2×.

### Gaps
- I did not get a primary page reference for the Griewank & Walther (2008, SIAM, 2nd ed.) book itself. The c < 6 constant is cited through Baydin et al. Griewank & Walther's own statement (often quoted as the gradient costing at most about 4–5 times OPS(f) under their complexity model) could not be verified from accessible text, so do not quote a 4 or 5 bound. Use Baydin's "c < 6, typically 2–3".
- The convergence condition η < 2/λ_max for quadratic losses is not in the fetched text (only ε* = 1/λ_max worst case). Source it separately if used for P2 item 8.

---

## Q6. FLOP counts for 784→128→10 with m = 64, and why one GEMM beats m GEMVs (arithmetic intensity, BLAS-3, GPUs)

### Takeaway
A product of (M×K)(K×N) costs 2MKN FLOPs. For the example MLP on a batch of 64, the forward pass is ≈ 13.0 MFLOP (million floating-point operations) and the backward pass (weights only) ≈ 13.2 MFLOP. Batching does not change the FLOP count. What it changes is arithmetic intensity (FLOPs per byte moved): each weight loaded from memory is reused m times. This turns memory-bound matrix–vector work (BLAS level 2, intensity < 1 FLOP/B on GPUs) into compute-bound matrix–matrix work (level 3).

### Cited Findings
- NVIDIA: "Each FMA is 2 operations, a multiply and an add, so a total of 2 * M * N * K FLOPS are required" for an M×K by K×N GEMM. FMA means fused multiply-add. — [NVIDIA, Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- NVIDIA arithmetic intensity: FLOPs / bytes accessed, proportional to MNK / (MK + NK + MN) (times 2/bytes-per-element). "Matrix-vector products (general matrix-vector product or GEMV), where either M=1 or N=1, are always memory limited; their arithmetic intensity is less than 1." On a V100 (FP16 with FP32 accumulation) the FLOPS:B ratio is 138.9. An 8192×128×8192 GEMM (intensity 124.1) is still memory-limited, while 8192³ (intensity 2730) is math-limited. — [NVIDIA Matrix Multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- NVIDIA, fully connected layers: for a 4096→4096 layer on an A100-SXM4-80GB, all three passes are "estimated to be memory-bound for batch sizes 128 and below". "The computation will always be bandwidth-limited for very small batch sizes, for example, 8 and below". The guide suggests "choose batch sizes and neuron counts greater than 128" and notes "Training with larger batch sizes is one option to extract more performance". — [NVIDIA Fully-Connected Layers guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html)
- Netlib BLAS: Level 1 performs "scalar, vector and vector-vector operations", Level 2 "matrix-vector operations", Level 3 "matrix-matrix operations". — [Netlib BLAS](https://www.netlib.org/blas/)
- Demmel (Berkeley CS267) counts slow-memory references m, flops f and ratio q = f/m for dimension n. saxpy (BLAS1): 3n refs, 2n flops, q = 2/3. sgemv (BLAS2): n² + O(n) refs, 2n² flops, q ≈ 2. sgemm (BLAS3): 4n² refs, 2n³ flops, q = n/2. "There is a clear difference in q for each level of BLAS, with BLAS3 the highest." The miss ratio is bounded below by 1/(2q), and blocked sgemm can reach q ≈ √(M_fast/3) (Hong & Kung lower bound). — [Demmel, CS267 Lecture 2, "Designing fast linear algebra kernels in the presence of memory hierarchies"](https://people.eecs.berkeley.edu/~demmel/cs267-1995/lecture02.html)

### Worked FLOP numbers (our computation, 2MKN rule; verified by script)
Network 784 → 128 → 10, batch m = 64. **W**^{(1)} ∈ ℝ^{128×784}, **W**^{(2)} ∈ ℝ^{10×128}, **X** ∈ ℝ^{784×64}. Parameters: 128·784 + 128 + 10·128 + 10 = **101,770**.

| Pass | Product | Shapes | FLOPs |
|---|---|---|---|
| Fwd L1 | **W**^{(1)}**X** | (128×784)(784×64) | 2·128·784·64 = 12,845,056 |
| Fwd L2 | **W**^{(2)}**A**^{(1)} | (10×128)(128×64) | 2·10·128·64 = 163,840 |
| Fwd bias + activation | **b1**^T adds, φ | element-wise | ≈ 128·64 + 10·64 adds + φ ≈ 9–17 k (negligible) |
| **Forward total (GEMMs)** | | | **≈ 13.01 MFLOP** |
| Bwd ∂ℓ/∂**W**^{(2)} | **Δ**^{(2)}(**A**^{(1)})^T | (10×64)(64×128) | 163,840 |
| Bwd error to layer 1 | (**W**^{(2)})^T**Δ**^{(2)} | (128×10)(10×64) | 163,840 |
| Bwd ∂ℓ/∂**W**^{(1)} | **Δ**^{(1)}(**X**)^T | (128×64)(64×784) | 12,845,056 |
| (not needed) ∂ℓ/∂**X** | (**W**^{(1)})^T**Δ**^{(1)} | (784×128)(128×64) | 12,845,056 |
| **Backward total (training)** | | | **≈ 13.17 MFLOP** (≈ 26.0 M if ∂ℓ/∂**X** is also computed) |

Forward + backward ≈ 26.2 MFLOP per mini-batch, about 2.0× the forward pass. That is ≈ 409 kFLOP per sample, consistent with the "small constant multiple" claim (Q5).

**Finite-difference comparison:** one-sided differences need 101,770 + 1 forward passes per batch, ≈ 101,771 × 13.0 MFLOP ≈ 1.3 × 10^{12} FLOP, versus ≈ 2.6 × 10^7 for backprop. The ratio is about 5 × 10^4.

**Arithmetic intensity of layer 1 at fp32 (4 bytes), minimum traffic = read **W**, read **X**, write **Z**:**

| m | FLOPs | Bytes | FLOP/B |
|---|---|---|---|
| 1 (GEMV) | 200,704 | 405,056 | ≈ 0.50 |
| 64 | 12,845,056 | 634,880 | ≈ 20.2 |
| 256 | 51,380,224 | 1,335,296 | ≈ 38.5 |

With m = 1, the whole 401 kB weight matrix is read to do only about 0.2 MFLOP. With m = 64, the same weights are read once and used 64 times: the FLOPs are 64× larger but the traffic only about 1.6× larger.

### Inferences
- **Why one GEMM beats m GEMVs:** the FLOPs are identical (m × 2n_l n_{l-1}). But m separate GEMVs reload **W** from memory m times (intensity < 1 FLOP/B, memory-bound by NVIDIA's statement), while one GEMM loads **W** once and reuses each loaded block across all m columns from cache/shared memory (intensity grows with m, Demmel's q = n/2 for square sgemm). GPUs have very high FLOP:byte ratios (e.g. 138.9 for V100 FP16), so only level-3 work can keep their arithmetic units busy. This is the "why GPUs" point for slide 7.
- Even m = 64 for our small MLP gives about 20 FLOP/B. By NVIDIA's ratios that is still memory-bound on a big GPU (their guide says ≤ 128 is memory-bound for a 4096-wide layer on A100). Present it honestly: batching moves the work toward the compute-bound regime, and larger layers and batches get closer. Do not claim m = 64 saturates a GPU.
- The batch dimension plays three different GEMM roles: N in the forward and error-propagation GEMMs, and K (the summed dimension) in the weight-gradient GEMM. This matches NVIDIA's Table 1 and the outer-product-sum view in Q4.

### Gaps
- No wall-clock benchmark was run (loop of 64 GEMVs vs one GEMM in NumPy). If the team wants a measured speed-up number for the slide, generate it in the experiment notebook and report the real figure. Do not invent one.
- A primary citation for the Level 3 BLAS paper (Dongarra, Du Croz, Hammarling & Duff, ACM TOMS 16(1), 1990) was not fetched. Netlib and Demmel are cited instead.
- NVIDIA's page gives no A100 ops:byte ratio in the fetched text; only the V100 figure (138.9) is cited.
