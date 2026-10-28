# Linear-algebra connections and extensions (handoff §5 items 8–13)

Notation follows handoff §2. Vectors are columns. φ is the activation (never σ). ℓ is the loss. Calligraphic letters are reserved for subspaces. Products are written by juxtaposition. Every derivation in items 8, 9 and 12 was re-derived by hand. The key identities were also checked numerically in NumPy; the results are under "Numerical checks" in items 8, 9 and 12. **[EXACT]** marks a statement that holds exactly. **[SIMPLIFICATION]** marks one that holds only under stated assumptions.

---

## Item 8: a one-layer linear network with squared loss is least squares; gradient-descent convergence comes from the eigenvalues of A^T A

### Takeaway
With identity activation and squared loss, training the network *is* the Ch. 10 least-squares problem. Setting the gradient to **0** gives the normal equations. The gradient-descent error obeys the Ch. 11 difference equation **e**_{k+1} = [**I** − η**A**^T**A**]**e**_k. Diagonalising the symmetric matrix **A**^T**A** shows that gradient descent converges if and only if 0 < η < 2/λ_max. The speed depends on the condition number κ = λ_max/λ_min.

### Notation clash (say it out loud on the slide)
In this item **b** is the Ch. 10 right-hand side (the targets). It is **not** the bias **b**^{(l)}. Mapping to network notation: each row **a**_i^T of **A** ∈ ℝ^{m×n} is one training input. **x** ∈ ℝ^n is the weight vector being learned, so it plays the role of (**w**^{(1)})^T with n_1 = 1 and no bias. **b** ∈ ℝ^m stacks the m targets y_i. Alternative: rename to **X**^T**w** ≈ **y** on that slide. Note that this "samples as rows" design matrix is the Ch. 10 convention, and it is the transpose of our batch **X** ∈ ℝ^{n_0×m}. With **A** = **X**^T the clash disappears.

### Derivation (all [EXACT] unless flagged)
1. Loss: ℓ(**x**) = ½‖**Ax** − **b**‖² = ½[**Ax** − **b**]^T[**Ax** − **b**]. Shapes: **A** m×n, **x** n×1, **b** m×1.
2. Gradient: ∇ℓ(**x**) = **A**^T[**Ax** − **b**] = **A**^T**Ax** − **A**^T**b** (n×1).
3. Stationary point: ∇ℓ = **0** ⟺ **A**^T**A x̂** = **A**^T**b**, the Ch. 10 normal equations. If rank(**A**) = n, then **A**^T**A** is invertible and **x̂** = [**A**^T**A**]^{-1}**A**^T**b**. The loss is convex (Hessian **A**^T**A** is positive semidefinite), so this stationary point is the global minimum.
4. Gradient-descent step: **x**_{k+1} = **x**_k − η**A**^T[**A x**_k − **b**].
5. Error recursion: let **e**_k = **x**_k − **x̂**. Substitute **A**^T**b** = **A**^T**A x̂**. Then **x**_{k+1} − **x̂** = **x**_k − **x̂** − η**A**^T**A**[**x**_k − **x̂**], so **e**_{k+1} = [**I** − η**A**^T**A**]**e**_k and **e**_k = [**I** − η**A**^T**A**]^k **e**_0. This is the Ch. 11 difference equation **u**_{k+1} = **M u**_k with **M** = **I** − η**A**^T**A**, solved by the powers **M**^k.
6. Eigen-decomposition: **A**^T**A** is symmetric with eigenvalues λ_i ≥ 0, so it has orthonormal eigenvectors and **A**^T**A** = **SΛS**^{-1} with **S**^{-1} = **S**^T. Then **M** = **S**[**I** − η**Λ**]**S**^{-1} and **M**^k = **S**[**I** − η**Λ**]^k**S**^{-1}. In eigen-coordinates **c**_k = **S**^{-1}**e**_k, each component decouples: c_{i,k} = (1 − ηλ_i)^k c_{i,0}.
7. Convergence condition: **e**_k → **0** for every **e**_0 ⟺ |1 − ηλ_i| < 1 for all i ⟺ 0 < ηλ_i < 2. With every λ_i > 0 (full column rank) this is **0 < η < 2/λ_max**. If some λ_i = 0 (rank-deficient **A**), that component never changes. GD still converges, but to the solution whose component in 𝒩(**A**) equals that of **x**_0. (Inference: from **x**_0 = **0** this is the minimum-norm solution **A**^+**b**, because every update η**A**^T[⋯] lies in 𝒞(**A**^T).)
8. Rate: the slowest mode sets the speed, ρ(η) = max_i |1 − ηλ_i| = max{|1 − ηλ_min|, |1 − ηλ_max|}. The best fixed step is η* = 2/(λ_min + λ_max), which gives ρ* = (κ − 1)/(κ + 1) with κ = λ_max/λ_min. If κ = 1, GD converges in one step. A large κ means slow convergence.

### Cited findings
- Goodfellow et al. work this exact example. For f(**x**) = ½‖**Ax** − **b**‖²₂ the gradient is **A**^T**Ax** − **A**^T**b**, and the algorithm repeats **x** ← **x** − ε[**A**^T**Ax** − **A**^T**b**] until the gradient norm falls below a tolerance δ (§4.5, p. 94). The book also notes that Newton's method solves this quadratic in a single step. — [Goodfellow, Bengio & Courville, *Deep Learning*, Ch. 4](https://www.deeplearningbook.org/contents/numerical.html)
- In the same chapter, the condition number is the ratio of the largest to the smallest eigenvalue magnitude, and "When the Hessian has a poor condition number, gradient descent performs poorly" (p. 87). The optimal step along **g** from the second-order Taylor model is ε* = **g**^T**g** / **g**^T**Hg**. In the worst case (**g** along the top eigenvector) this becomes 1/λ_max (p. 86). — [Goodfellow et al., Ch. 4](https://www.deeplearningbook.org/contents/numerical.html)
- Goh (Distill, 2017) analyses f(**w**) = ½**w**^T**Aw** − **b**^T**w** with **A** symmetric and invertible. Changing to the eigenbasis "the iterations break apart": x_i^{k+1} = (1 − αλ_i)x_i^k, so x_i^k = (1 − αλ_i)^k x_i^0. "All workable step-sizes … fall in the interval" 0 < αλ_i < 2, i.e. 0 < α < 2/λ_n. The rate is max{|1 − αλ_1|, |1 − αλ_n|}. The optimal α = 2/(λ_1 + λ_n) gives rate (κ − 1)/(κ + 1), and the condition number is "a direct measure of pathological curvature". The loss gap is f(**w**^k) − f(**w**\*) = Σ_i (1 − αλ_i)^{2k} λ_i [x_i^0]². — [Goh, "Why Momentum Really Works", Distill 2017](https://distill.pub/2017/momentum/)
  - Mapping to our item: Goh's **A** is our **A**^T**A**, his **b** is our **A**^T**b**, and his α is our η. (½‖**Ax** − **b**‖² = ½**x**^T**A**^T**Ax** − [**A**^T**b**]^T**x** + const.) Goh writes **A** = **Q** diag(λ) **Q**^T, which is the symmetric case of **SΛS**^{-1}.

### Numerical checks (mine, NumPy, random **A** ∈ ℝ^{50×3})
- λ(**A**^T**A**) ≈ {33.9, 39.6, 64.9}, so 2/λ_max ≈ 0.0308 and κ ≈ 1.91.
- η = 0.9 · 2/λ_max: ‖**e**_200‖ ≈ 1e−16 (converged).
- η = 1.05 · 2/λ_max: ‖**e**_200‖ ≈ 3e7 (diverged). This confirms the bound is sharp.
- η* = 2/(λ_min + λ_max): converged.
- One step of the update matches [**I** − η**A**^T**A**]**e**_k to 3e−16.

### Why use gradient descent rather than the closed form (inferences, standard numerical-linear-algebra facts)
- Cost: forming **A**^T**A** costs about mn² flops, and solving the n×n system (Cholesky/LU) costs about n³/3 to 2n³/3 flops. The memory is n² entries. For a network with n ≈ 10⁶–10⁹ parameters, n² storage alone is infeasible. One GD step costs only two matrix–vector products (~4mn flops) and O(n) extra memory, and mini-batches make it cheaper still. *[I did not fetch a source for the flop counts. They are standard (e.g. Boyd & Vandenberghe, VMLS, Ch. 12), but cite them in the report only after checking.]*
- Conditioning: forming **A**^T**A** squares the condition number of **A**, which is why numerical libraries solve least squares via QR (Ch. 7) instead. *[Standard fact, not source-verified here.]*
- Non-linearity: once φ is non-linear, or there is more than one layer, ∇ℓ = **0** is a non-linear system with no closed form. GD/backprop still works. The linear case is the one setting where we can check GD against an exact answer. [Inference]
- Link to the deep case [SIMPLIFICATION]: near a minimum, any smooth loss is approximately quadratic with Hessian **H**. The same 2/λ_max(**H**) bound and κ(**H**) rate then hold locally. Goodfellow p. 86 makes this "to the extent that the function … can be approximated well by a quadratic" point. — [Goodfellow Ch. 4](https://www.deeplearningbook.org/contents/numerical.html)

### Gaps
- I did not fetch Strang, *Linear Algebra and Learning from Data* (2019) or MIT 18.065 to quote a page. They cover this exact material (Part VI, gradient descent), but no citation is verified here.
- I did not fetch Boyd & Vandenberghe for the flop counts.

---

## Item 9: vanishing and exploding gradients through the Ch. 11 lens

### Takeaway
Backprop multiplies **δ** by (**W**^{(l+1)})^T and by a diagonal matrix of φ′ values at every layer. In the toy case of the same **W** at every layer with linear φ, this is a matrix power (**W**^T)^k = [**S**^{-1}]^T **Λ**^k **S**^T. Eigenvalue components with |λ| > 1 blow up like |λ|^k, and those with |λ| < 1 decay to zero. This is **intuition** for feed-forward nets. For RNNs, which really do reuse one **W**, Pascanu, Mikolov & Bengio (2013) make it rigorous.

### Derivation
1. [EXACT] Rewrite the Hadamard form as a matrix product: **δ**^{(l)} = [(**W**^{(l+1)})^T**δ**^{(l+1)}] ⊙ φ′(**z**^{(l)}) = **D**^{(l)}(**W**^{(l+1)})^T**δ**^{(l+1)}, where **D**^{(l)} = diag(φ′(**z**^{(l)})). Unrolling gives
   **δ**^{(1)} = **D**^{(1)}(**W**^{(2)})^T**D**^{(2)}(**W**^{(3)})^T ⋯ **D**^{(L-1)}(**W**^{(L)})^T **δ**^{(L)}.
   Shapes: **D**^{(l)} is n_l×n_l and (**W**^{(l+1)})^T is n_l×n_{l+1}, so the product is n_1×n_L. Conformant.
2. [SIMPLIFICATION] Assume every **W**^{(l)} = **W** (square, n×n) and φ(z) = z, so every **D** = **I**. Then **δ**^{(1)} = (**W**^T)^{L-1}**δ**^{(L)}.
3. [EXACT, given diagonalisable **W**] If **W** = **SΛS**^{-1}, then **W**^T = (**S**^{-1})^T**Λ S**^T. This is still a diagonalisation, with eigenvector matrix (**S**^{-1})^T, because (**S**^T)^{-1} = (**S**^{-1})^T. Therefore (**W**^T)^k = (**S**^{-1})^T**Λ**^k**S**^T. (This replaces the unfinished "S^{-T}…" in the brief. **W** and **W**^T have the same eigenvalues but generally different eigenvectors.)
4. Write **δ**^{(L)} = Σ_i c_i **u**_i in the eigenvectors **u**_i of **W**^T (columns of (**S**^{-1})^T). Then **δ**^{(1)} = Σ_i c_i λ_i^{L-1} **u**_i. Components with |λ_i| > 1 grow like |λ_i|^{L-1} (explode). Those with |λ_i| < 1 shrink to 0 (vanish). Only |λ_i| ≈ 1 survives many layers. This is exactly the Fibonacci / **A**^k = **SΛ**^k**S**^{-1} reasoning from Ch. 11.
5. Activation factor [EXACT bound]: sigmoid′(z) = sigmoid(z)[1 − sigmoid(z)] ≤ 1/4, with equality at z = 0. So ‖**D**^{(l)}‖ ≤ 1/4, and each sigmoid layer shrinks the backward signal by a factor of at least 4, on top of whatever **W**^T does. Over 10 layers that is up to (1/4)^{10} ≈ 1e−6 from φ′ alone. tanh′ ≤ 1. ReLU′ ∈ {0, 1}, so it does not shrink active units, which is one reason ReLU helps.
6. Caveats to state on the slide: (i) real feed-forward nets have a different **W**^{(l)} at each layer, so there is no single Λ^k. (ii) φ′ ≠ 1 and changes with the input. (iii) **W** may be non-diagonalisable or have complex λ. (iv) The rigorous tool is singular values / matrix norms, not eigenvalues. A non-symmetric **W** with all |λ| < 1 can still transiently amplify, which is why Pascanu et al.'s eigenvalue condition is stated as sufficient/necessary, not as an equivalence.

### Cited findings
- Pascanu, Mikolov & Bengio use "the *power iteration method*" to analyse the product of Jacobians. With identity activation, the product reduces to (**W**_rec^T)^l with l = t − k. A largest eigenvalue (in absolute value) "smaller than 1" is **sufficient** for long-term components to vanish, and it is "necessary for it to be larger than 1 for gradients to explode". With a non-linear activation whose derivative is bounded by γ (‖diag(φ′)‖ ≤ γ), the sufficient condition for vanishing becomes λ_1 < 1/γ. They give γ = 1 for tanh and γ = 1/4 for sigmoid. Their fix for exploding gradients is gradient-norm clipping: if ‖**ĝ**‖ ≥ threshold, set **ĝ** ← (threshold/‖**ĝ**‖)**ĝ**. — [Pascanu, Mikolov & Bengio, "On the difficulty of training recurrent neural networks", ICML 2013 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1211.5063); venue confirmed as ICML 2013, PMLR 28(3): 1310–1318 — [PMLR](https://proceedings.mlr.press/v28/pascanu13.html); arXiv [1211.5063](https://arxiv.org/abs/1211.5063)
  - Note: the paper's own words call λ_1 "the absolute value of the largest eigenvalue". Its proof uses the norm (largest singular value), so the team should say "spectral radius / largest singular value" if pressed in Q&A. The paper writes σ for the activation; we write φ.
- Goodfellow et al. §8.2.5 ("Long-Term Dependencies"): repeated multiplication by **W** equals **W**^t. If **W** = **V** diag(λ) **V**^{-1}, then **W**^t = **V** diag(λ)^t **V**^{-1}, and eigenvalues "will either explode if they are greater than 1 in magnitude or vanish if they are less than 1 in magnitude". They note that feed-forward networks, which use a different matrix at each layer, largely avoid the pure-power version. This is good support for labelling our slide "intuition". — [Goodfellow et al., Ch. 8](https://www.deeplearningbook.org/contents/optimization.html)
- Glorot & Bengio (2010) studied deep feed-forward nets empirically and analysed how activation and gradient variances change across layers (see item 12). — [Glorot & Bengio, AISTATS 2010, PMLR 9](https://proceedings.mlr.press/v9/glorot10a)
- Bengio, Simard & Frasconi (1994), "Learning long-term dependencies with gradient descent is difficult", *IEEE Trans. Neural Networks* 5(2), is the classic RNN statement. It is cited as the origin by Pascanu et al. (the abstract says the problems were "described by Bengio et al. (1994)"). — [Pascanu et al. abstract](https://arxiv.org/abs/1211.5063)

### Numerical check
- For a random 3×3 **W**, (**W**^T)^5 matched (**S**^{-1})^T**Λ**^5**S**^T to 1e−13, using complex eigenvalues where they arose.

### Inferences
- Good slide pairing: item 8 shows that eigenvalues of [**I** − η**A**^T**A**] decide whether *training iterations* converge. Item 9 shows that eigenvalues of **W**^T decide whether *signals through depth* survive. Both are "powers of a matrix" (Ch. 11 difference equations), applied along two different axes: time (iterations) and depth (layers).

### Gaps
- I did not retrieve Hochreiter (1991), a German diploma thesis ("Untersuchungen zu dynamischen neuronalen Netzen", TU München), online. Cite it only as secondary attribution.
- I did not fetch the Bengio, Simard & Frasconi 1994 IEEE paper itself. Volume and issue are from memory, so verify them via IEEE Xplore before printing.

---

## Item 10: null space and rank, bottlenecks, and the linear autoencoder as PCA

### Takeaway
The first layer only "sees" the component of **x** in the row space 𝒞((**W**^{(1)})^T). Changes in 𝒩(**W**^{(1)}) leave **z**^{(1)} unchanged. A narrow layer caps the rank of everything downstream. A linear autoencoder with a narrow bottleneck learns exactly the PCA subspace.

### Statements (verified by hand, [EXACT])
- Every **x** ∈ ℝ^{n_0} splits uniquely as **x** = **x**_r + **x**_n, with **x**_r ∈ 𝒞((**W**^{(1)})^T) and **x**_n ∈ 𝒩(**W**^{(1)}). These are orthogonal complements (Ch. 9). Then **z**^{(1)} = **W**^{(1)}**x** + **b**^{(1)} = **W**^{(1)}**x**_r + **b**^{(1)}. Inputs that differ by a null-space vector produce identical activations in *every* later layer, so the whole network cannot tell them apart.
- dim 𝒩(**W**^{(1)}) = n_0 − rank(**W**^{(1)}) ≥ n_0 − n_1. So if n_1 < n_0, the null space is non-trivial and the layer *must* discard at least n_0 − n_1 input directions.
- Rank bottleneck (Ch. 5): for the linear part of a stack, rank(**W**^{(L)} ⋯ **W**^{(1)}) ≤ min_l n_l. With a non-linear φ, a narrow layer still forces **a**^{(l)} to live in ℝ^{n_l}, so all later layers are functions of only n_l numbers. [The rank bound is EXACT for the linear product. The "information bottleneck" reading for non-linear nets is interpretive.]
- Gradient view: ∂ℓ/∂**W**^{(1)} = **δ**^{(1)}**x**^T, so every row of the update is a multiple of **x**^T. Over a batch, the update rows lie in span{**x**_j}. Weight directions orthogonal to all training inputs are never changed by GD. [EXACT; this mirrors the minimum-norm remark in item 8.]
- Autoencoder: an encoder **W**^{(1)} ∈ ℝ^{p×n} (p < n) is followed by a decoder **W**^{(2)} ∈ ℝ^{n×p}, trained so that **W**^{(2)}**W**^{(1)}**x** ≈ **x**. The reconstruction map **W**^{(2)}**W**^{(1)} has rank ≤ p, so it can only reproduce a p-dimensional subspace of inputs.

### Cited findings
- Baldi & Hornik studied linear networks with n inputs, p ≤ n hidden units and squared error. In the auto-associative case, "E has, up to equivalence, a unique local and global minimum corresponding to an orthogonal projection" onto "the subspace generated by the first principal vectors of a covariance matrix", and "All other critical points of E are saddle points." The scope is linear units with no bias and a single hidden layer. — [Baldi & Hornik, "Neural networks and principal component analysis: learning from examples without local minima", *Neural Networks* 2(1): 53–58, 1989 (author PDF)](https://igb.uci.edu/~pfbaldi/publications/journals/1989/sdarticle.pdf)
  - SVD-free phrasing: "the first p principal vectors" = the eigenvectors **s**_1, …, **s**_p of the (symmetric) covariance matrix **C** = (1/m)**X̃ X̃**^T with the p largest eigenvalues, where **X̃** is the centred data with samples as columns. This is a pure Ch. 11 eigen-decomposition. The optimal **W**^{(2)}**W**^{(1)} is the projection matrix **P** = **S**_p**S**_p^T (Ch. 10 projection onto 𝒞(**S**_p)).
  - Caveat: the individual **W**^{(1)} rows need *not* equal the eigenvectors. Only the subspace 𝒞((**W**^{(1)})^T) (equivalently the product) is determined, "up to equivalence" (any invertible p×p mixing **G**: **W**^{(1)} → **GW**^{(1)}, **W**^{(2)} → **W**^{(2)}**G**^{-1}).
  - Journal, volume and pages are from my knowledge plus the PDF's page numbers 53+. The OCR text did not show the journal name, so verify the exact volume and pages.

### Gaps
- No source fetched for a general "information bottleneck" claim. Keep it interpretive.

---

## Item 11: universal approximation theorem

### Takeaway
One hidden layer with a non-polynomial activation can approximate any continuous function on a compact (closed, bounded) set as closely as we like, *if it is wide enough*. The theorem guarantees existence only. It does not say how wide, how to find the weights, or whether training will find them.

### Intuitive statement (our notation)
For any continuous f: K → ℝ on a compact K ⊂ ℝ^{n_0} and any ε > 0, there exist a width n_1 and **W**^{(1)} ∈ ℝ^{n_1×n_0}, **b**^{(1)} ∈ ℝ^{n_1}, **w**^{(2)} ∈ ℝ^{n_1} such that |[**w**^{(2)}]^T φ(**W**^{(1)}**x** + **b**^{(1)}) − f(**x**)| < ε for all **x** ∈ K. LA reading: f is approximated by a *linear combination* of the n_1 functions φ(**w**_i^T**x** + b_i), i.e. the function space spanned by these "ridge" functions is dense.

### Cited findings
- Cybenko, "Approximation by superpositions of a sigmoidal function", *Mathematics of Control, Signals, and Systems* 2(4): 303–314 (1989). Proves the result for sigmoidal φ using functional-analysis tools (Hahn–Banach, Riesz representation). — [Wikipedia: Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)
- Hornik, Stinchcombe & White, "Multilayer feedforward networks are universal approximators", *Neural Networks* 2(5): 359–366 (1989). Broadens the class of activations and gives a Stone–Weierstrass proof. Hornik (1991), *Neural Networks* 4(2): 251–257, argues that the architecture, not the specific activation, gives universality. — [Wikipedia](https://en.wikipedia.org/wiki/Universal_approximation_theorem)
- Leshno, Lin, Pinkus & Schocken, "Multilayer feedforward networks with a nonpolynomial activation function can approximate any function", *Neural Networks* 6(6): 861–867 (1993). Universality holds **if and only if** φ is not a polynomial (for locally bounded, piecewise-continuous φ). This covers ReLU. — [Wikipedia](https://en.wikipedia.org/wiki/Universal_approximation_theorem); report record at [NYU archive](https://archive.nyu.edu/handle/2451/14329)
- Limits: these are existence theorems that give no method for finding the parameters (that is left to optimisation such as backprop). They do not specify the required size; universality is reached by widening the hidden layer. They hold only on compact domains and say nothing outside K. — [Wikipedia](https://en.wikipedia.org/wiki/Universal_approximation_theorem)

### Inferences
- Links to the deck: (a) "non-polynomial" explains why identity φ fails. A polynomial of fixed degree (degree 1 = affine) spans a finite-dimensional function space, and the collapse theorem (item 3) is the degree-1 case. (b) The theorem says nothing about generalisation or trainability, which are good Q&A points.

### Gaps
- The Wikipedia article does not claim that the required width is exponential in general. Do **not** say "exponential" on the slide without a primary source. Say "may need to be very large".
- I did not fetch the primary papers. Bibliographic details come via Wikipedia, so cross-check before the references slide.

---

## Item 12: initialisation (Xavier/Glorot, He) and symmetry breaking

### Takeaway
Choose Var(w) so that each layer neither shrinks nor grows the variance of the signal (forward) or of **δ** (backward). In LA terms, this keeps the "typical gain" of each **W**^{(l)} near 1, which is the item 9 condition |λ| ≈ 1 achieved statistically. Weights must also be *random*: identical weights make every neuron in a layer identical forever.

### Variance-preservation argument [SIMPLIFICATION: weights i.i.d. with mean 0; inputs i.i.d., independent of weights; linear regime]
- One pre-activation: z_i = Σ_{j=1}^{n_{l-1}} w_{ij} a_j. With independent zero-mean weights, Var(z_i) = n_{l-1} Var(w) E[a_j²].
- Glorot (tanh/linear regime, φ′(0) = 1, zero-mean a): Var(z^{(l)}) = n_{l-1}Var(w)Var(z^{(l-1)}). Forward preservation needs n_{l-1}Var(w) = 1. The backward pass multiplies by (**W**^{(l)})^T, which sums over n_l terms, so it needs n_l Var(w) = 1. The compromise is **Var(w) = 2/(n_{l-1} + n_l) = 2/(n_in + n_out)**.
- He (ReLU): E[a²] = ½Var(z^{(l-1)}), since ReLU zeroes half of a symmetric input. So Var(z^{(l)}) = ½n_{l-1}Var(w)Var(z^{(l-1)}), and preservation needs **Var(w) = 2/n_in**.

### Cited findings
- Glorot & Bengio §4.2.1: the forward condition is n_i Var[W^i] = 1 (Eq. 10) and the backward condition is n_{i+1} Var[W^i] = 1 (Eq. 11). "As a compromise between these two constraints" they propose Var[W^i] = 2/(n_i + n_{i+1}) (Eq. 12). The "normalized initialization" is W ~ U[−√6/√(n_j + n_{j+1}), +√6/√(n_j + n_{j+1})] (Eq. 16), whose variance is 2/(n_j + n_{j+1}). Assumptions: a "linear regime at the initialization", a symmetric activation with f′(0) = 1, independent weights, and equal input-feature variances. — [Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks", AISTATS 2010, JMLR W&CP/PMLR 9: 249–256 (PDF)](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
  - Check: U[−a, a] has variance a²/3 = [6/(n_j + n_{j+1})]/3 = 2/(n_j + n_{j+1}). ✓
- He et al.: the forward sufficient condition is ½n_l Var[w_l] = 1 (Eq. 10), giving a zero-mean Gaussian with std √(2/n_l). The ½ appears because, for ReLU, E[x_l²] = ½Var[y_{l-1}]. The backward version is ½n̂_l Var[w_l] = 1 (Eq. 14), and "it is sufficient to use either" condition alone. (For convolutional layers n_l = k²c_l, i.e. kernel area times input channels.) — [He, Zhang, Ren & Sun, "Delving Deep into Rectifiers", arXiv 1502.01852 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1502.01852); published at ICCV 2015 — [CVF open access](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper)
- Symmetry breaking: initial parameters must "break symmetry" between units. Two hidden units with the same activation, connected to the same inputs and with identical initial parameters, are updated identically by a deterministic algorithm, so the parameters are initialised randomly. — [Goodfellow et al., Ch. 8.4](https://www.deeplearningbook.org/contents/optimization.html)

### Symmetry-breaking proof sketch in our notation [EXACT, induction]
Suppose at step t all rows of **W**^{(1)} are equal, all entries of **b**^{(1)} are equal, and all entries of **w**^{(2)} (the next layer's column weights from these units) are equal. Then every entry of **z**^{(1)} is equal, so every entry of **a**^{(1)} and of φ′(**z**^{(1)}) is equal. **δ**^{(1)} = [(**W**^{(2)})^T**δ**^{(2)}] ⊙ φ′(**z**^{(1)}) has equal entries because the columns of **W**^{(2)} are equal. Hence ∂ℓ/∂**W**^{(1)} = **δ**^{(1)}**x**^T has equal *rows*. The gradient for **W**^{(2)}, **δ**^{(2)}(**a**^{(1)})^T, has equal *columns*. After the update the equalities still hold, so by induction they hold forever. The layer then behaves like one neuron copied n_1 times: the rank of **W**^{(1)} stays ≤ 1, a nice Ch. 5 tie-in. (Biases set to 0 are fine. The randomness only needs to be in **W**.)
- Numerical check (mine): a 2→3→1 tanh network with all weights 0.5, trained for 100 GD steps on random data. The three rows of **W**^{(1)} remained exactly equal (max spread 0), although the weights themselves changed (0.5 → [0.435, 0.404]).

### Gaps
- Exact spectral link: for a random n×n Gaussian matrix with entry variance 1/n, the largest singular value is ≈ 2 and the typical gain is ≈ 1. That statement needs random-matrix theory (and SVD). Keep it as "roughly keeps the scale near 1" (as handoff §5.12 says) unless a source is added.

---

## Item 13: pointers to convolution and attention

### Takeaway
Both are still matrix products. Convolution is multiplication by a sparse, weight-shared (Toeplitz / doubly block circulant) matrix. Attention is two matrix products with a row-wise softmax in between.

### Cited findings
- Convolution: "Discrete convolution can be viewed as multiplication by a matrix", with entries constrained to equal other entries. In 1-D, "each row of the matrix is constrained to be equal to the row above shifted by one element", which is "known as a Toeplitz matrix". In 2-D, "a doubly block circulant matrix corresponds to convolution". The matrix is "very sparse" because the kernel is small (sparse interactions), and its repeated entries are parameter sharing. — [Goodfellow et al., *Deep Learning*, Ch. 9](https://www.deeplearningbook.org/contents/convnets.html)
  - Tiny example for a slide (mine, [EXACT]): a kernel [k_1 k_2 k_3] on an input **x** ∈ ℝ^5 with no padding gives **z** = **Kx** with **K** ∈ ℝ^{3×5}, rows [k_1 k_2 k_3 0 0], [0 k_1 k_2 k_3 0], [0 0 k_1 k_2 k_3]. This uses 3 parameters instead of 15. (Deep-learning "convolution" is really cross-correlation, i.e. the kernel is not flipped; Goodfellow Ch. 9 makes this point.)
- Attention: Attention(**Q**, **K**, **V**) = softmax(**QK**^T/√d_k)**V** (Eq. 1). Queries are "packed together into a matrix **Q**". The scaling exists because, for large d_k, "the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients". Their footnote: if components of **q**, **k** are independent with mean 0 and variance 1, then **q**^T**k** has variance d_k. — [Vaswani et al., "Attention Is All You Need", arXiv 1706.03762 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1706.03762); [arXiv abstract](https://arxiv.org/abs/1706.03762)
  - **Convention warning:** in transformer papers each **row** of **Q**, **K**, **V** is a token. That is the opposite of our columns-as-samples convention. Entry (i, j) of **QK**^T is **q**_i^T**k**_j, a Ch. 2 dot product (similarity) between token i's query and token j's key. Softmax is applied **row-wise**. In our column convention, with tokens as columns, the same computation is **V**_c softmax_col(**K**_c^T**Q**_c/√d_k), where **K**_c = **K**^T etc. State the row convention explicitly on the slide or convert it.
  - The √d_k footnote is the same variance argument as item 12 (a sum of d_k unit-variance products has variance d_k). This gives a nice "callback" for the closing slide. [Inference]

### Gaps
- I did not verify the venue (NeurIPS/NIPS 2017) from the fetched page. It is widely known, but confirm before printing.
