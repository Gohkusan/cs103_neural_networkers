# HANDOFF: CS103 Project, "Neural Networks Expressed in Linear Algebra"

> **How to use this file:** Read all of it before you start researching. It holds the course constraints, the professor's notation rules, what the class has already been taught (so we do not re-teach it), the grading rubric turned into concrete targets, and the slide-design rules. Where this file and your defaults disagree, this file wins. The project brief the team was given is quoted in §11.

---

## 0. Mission (in one paragraph)

Research how a feed-forward neural network is, at its core, a chain of matrix operations: **forward propagation** (matrix–vector and matrix–matrix products), **non-linear activations** (why they are needed), and **backpropagation** (transposes, outer products, Hadamard products). Then produce:

1. **A research report** for the team: accurate, cited, written in the professor's notation, with the Q&A preparation in §9.
2. **A 10-minute slide deck** that walks a CS103 classmate from "things we already know" to "how a neural network learns", with **as few mental steps between slides as possible** (see §7).

The audience is our classmates and the teaching team. They know CS103 weeks 1–11 well (§3) but **may know nothing about neural networks**.

---

## 1. Hard constraints (from the project brief)

| Constraint | Detail |
|---|---|
| Length | **10 min presentation + 5 min Q&A.** Aim for about 10–13 content slides (roughly 45–60 s each). |
| Originality | Must go **beyond** the CS103 syllabus. Do **not** re-teach class material. Keep background short and use it only as a bridge. |
| Required emphasis | Show *how linear algebra is applied* in neural networks: the interconnections between concepts, and the algorithms that use them. |
| Math formatting | LaTeX or Word equations only (no plain-text math in screenshots). **Must follow the LA4CS glossary notation** (§2). |
| Multiplication | **Never** use `*`, `·` (dot) or `×` (cross) for multiplication. Write products by juxtaposition: `Wx`, `x^T y`, `ηg`. (`×` only in dimensions such as `m × n`.) |
| Brackets | Matrices and vectors in **square brackets** `[ ]`, never parentheses. |
| Slides | Original content only: no web screenshots, no copied diagrams, no gimmicks. Draw our own figures. |
| AI use | Allowed, but **every team member must be able to defend every slide** in Q&A. Prefer depth the team can explain over breadth it cannot. |
| Submission | Nothing is submitted. Grading = instructor (5) + peer evaluation (5) + Q&A (5). |

---

## 2. Notation spec (non-negotiable; this follows the professor's LA4CS glossary)

### 2.1 Glossary rules

| Object | Write it as | Notes |
|---|---|---|
| Scalar | lowercase italic: `s`, `η`, `x_i` | |
| Vector | **bold lowercase**: **x**, **b**, **z**, **a** | Always a **column** vector, in square brackets. |
| Matrix | **bold uppercase**: **W**, **A**, **X** | Element `w_{ij}` (row *i*, column *j*). |
| Matrix by columns | **A** = [**a**_j] | Columns **a**_j. |
| Matrix by rows | **A** = [**a**_i^T] | Rows **a**_i^T. Useful for "each row of **W** is one neuron". |
| Identity / zero | **I**, **I**_n, bold **0** | |
| Transpose | **A**^T (upright T) | **A** ∈ ℝ^{m×n} ⟹ **A**^T ∈ ℝ^{n×m} |
| Inverse / pseudo-inverse | **A**^{-1}, **A**^+ | |
| Rank, trace | rank(**A**), trace(**A**) | |
| Determinant | \|**A**\| | |
| Fields | ℝ, ℝ^m, ℝ^{m×n} | |
| Spaces | calligraphic: 𝒞(**A**), 𝒞(**A**^T), 𝒩(**A**), 𝒩(**A**^T) | **Calligraphic letters are reserved for spaces.** Do not use 𝓛 for the loss. |
| Inner product | **x**^T**y** (preferred), rarely **x**·**y** | Orthogonal: **x** ⊥ **y** |
| Eigenvalues | λ | Eigenvectors **s** (the book uses **A** = **SΛS**^{-1}) |
| Singular values | σ, **Σ** | **σ is taken. Do not use σ for the sigmoid activation** (see 2.2). |
| Norm | ‖**x**‖ | |
| Logic | ∀, ∈, ⟹, ⊆ | |
| Hadamard (element-wise) product | **A** ⊙ **B** | Defined in LA4CS Ch. 2 ("Other Operations"), so it is in-notation. |

The book writes LaTeX bold as `\boldsymbol{}` / `\mathbf{}` and transpose as `^\mathsf{T}`. Use those.

### 2.2 Neural network notation we adopt (chosen to avoid clashes with the glossary)

Use this consistently in both report and slides, and define it on the slide where each symbol first appears.

| Meaning | Symbol | Shape | Why this choice |
|---|---|---|---|
| Input | **x** = **a**^{(0)} | ℝ^{n_0} | column vector |
| Layer index | superscript in parentheses ^{(l)}, l = 1, …, L | | parentheses, so it cannot be read as a power or as ^T |
| Neurons in layer l | n_l | | |
| Weights of layer l | **W**^{(l)} | ℝ^{n_l × n_{l-1}} | row *i* = weights of neuron *i* |
| Bias of layer l | **b**^{(l)} | ℝ^{n_l} | |
| Pre-activation | **z**^{(l)} = **W**^{(l)}**a**^{(l-1)} + **b**^{(l)} | ℝ^{n_l} | |
| Activation function | **φ** (applied element-wise) | | **Not σ** (σ = singular value in the glossary). Name it in words when concrete: ReLU(z), tanh(z), "sigmoid". |
| Activation | **a**^{(l)} = φ(**z**^{(l)}) | ℝ^{n_l} | |
| Output | **ŷ** = **a**^{(L)} | | |
| Target | **y** | | |
| Loss | ℓ (script lowercase l) or J | scalar | **Not** 𝓛 (calligraphic = spaces). Do not reuse L, which is the number of layers. |
| Error signal | **δ**^{(l)} = ∂ℓ/∂**z**^{(l)} | ℝ^{n_l} | |
| Gradient wrt a matrix | ∂ℓ/∂**W**^{(l)} | same shape as **W**^{(l)} | The glossary has no gradient symbol, so define this on the slide. |
| Learning rate | η | scalar | |
| Batch of m inputs | **X** = [**x**_j] = [**x**_1 **x**_2 ⋯ **x**_m] | ℝ^{n_0 × m} | samples as **columns**, matching the column-vector convention |
| Ones vector | **1** | ℝ^m | bold, like **0** |

Clash to watch: the book's **b** in **Ax** = **b** is the right-hand side, while ours is the bias. Fine, but when we bridge to Ch. 10 (regression), say so out loud or rename for that slide.

### 2.3 Core equations, written in the target notation (verify, do not reinvent)

Forward pass (one sample):
```latex
\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad
\mathbf{a}^{(l)} = \varphi\big(\mathbf{z}^{(l)}\big), \qquad l = 1,\dots,L
```
Forward pass (batch, samples as columns), where the bias broadcast is an **outer product** (LA4CS §2.7.5):
```latex
\mathbf{Z}^{(l)} = \mathbf{W}^{(l)}\mathbf{A}^{(l-1)} + \mathbf{b}^{(l)}\mathbf{1}^\mathsf{T}
```
Backpropagation (one sample):
```latex
\boldsymbol{\delta}^{(L)} = \frac{\partial \ell}{\partial \mathbf{a}^{(L)}} \odot \varphi'\big(\mathbf{z}^{(L)}\big), \qquad
\boldsymbol{\delta}^{(l)} = \Big[\big(\mathbf{W}^{(l+1)}\big)^\mathsf{T}\boldsymbol{\delta}^{(l+1)}\Big] \odot \varphi'\big(\mathbf{z}^{(l)}\big)
```
```latex
\frac{\partial \ell}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)}\big(\mathbf{a}^{(l-1)}\big)^\mathsf{T} \;\;(\text{outer product, rank 1}), \qquad
\frac{\partial \ell}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}
```
Batch version: ∂ℓ/∂**W**^{(l)} = **Δ**^{(l)}(**A**^{(l-1)})^T (a sum of m rank-1 outer products; state whether we average by 1/m).

Gradient descent update:
```latex
\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta\,\frac{\partial \ell}{\partial \mathbf{W}^{(l)}}
```

---

## 3. What the class already knows (weeks 1–11, LA4CS Ch. 1–11)

These are the **bridges** we may point at. **Do not re-teach them**: give one line ("recall from Week X …") and move on.

| Week / Ch. | Content covered | Hook it gives us |
|---|---|---|
| 1. Functions, Equations, Linearity | linearity, multivariable linear functions, linear transformations | Definition of linear ⟹ why **Wx** + **b** is *affine*, and why stacking linear layers is pointless |
| 2. Vectors & Matrices | scalar mult., linear combinations, dot product, matrix mult., conformance, **block multiplication**, **row/column pictures**, **inner vs outer products**, **Hadamard product**, tensors (briefly) | neuron = dot product; layer = matrix–vector product; batch = matrix–matrix; ∂ℓ/∂**W** = outer product; ⊙ in backprop |
| 3. Transposes & Determinants | transpose and its product rule (**AB**)^T = **B**^T**A**^T | **W**^T sends errors backwards; shape checking |
| 4. Gaussian Elimination | solvability, LU, PLU | contrast: we do *not* solve NN training exactly |
| 5. Rank, Gauss–Jordan, Inverses | rank properties, left/right inverses, Gram matrix **A**^T**A** | rank(**W**_2**W**_1) ≤ min(rank(**W**_1), rank(**W**_2)) ⟹ bottlenecks |
| 6. Vector Spaces | span, independence, subspaces, basis, dimension, orthogonal complements | a layer's outputs live in a subspace |
| 7. Orthogonality & Change of Basis | orthonormal bases, change of basis, Gram–Schmidt, QR, rotation matrices | a hidden layer as a learned change of coordinates |
| 8. (no chapter in our folder) | probably the midterm or a break | |
| 9. Four Fundamental Subspaces | 𝒞(**A**), 𝒩(**A**), 𝒞(**A**^T), 𝒩(**A**^T), complete solution | 𝒩(**W**) = input directions a layer is blind to |
| 10. Projection, Least Squares, Linear Regression | projection matrix, normal equations **A**^T**A** x̂ = **A**^T**b**, simple linear regression | a 1-layer network with identity activation and squared loss **is** linear regression; gradient descent reaches the same **x̂** |
| 11. Eigen-decomposition | **A**s = λ**s**, diagonalization **A** = **SΛS**^{-1}, powers **A**^k = **SΛ**^k**S**^{-1}, difference equations, Fibonacci | repeated multiplication by **W**^T in backprop ⟹ vanishing/exploding gradients (intuition) |

**Not covered in class, and so "original" for the rubric:** neurons and layers, activation functions, the composition of affine and non-linear maps, the universal approximation idea, loss functions, gradients with respect to vectors and matrices (matrix calculus, the chain rule as Jacobian products), backpropagation, gradient descent, batching, vanishing/exploding gradients, initialization. **SVD is not covered either** (Ch. 11 stops at eigen-decomposition). Mention it only if needed, and define it.

---

## 4. Rubric turned into concrete targets

| Rubric row | What "A/A+" means for this topic, concretely |
|---|---|
| Theoretical background | Background ≤ 1 slide. Spend the time on the *new* ideas: affine maps + non-linearity, matrix calculus, backprop as transposed matrix products. |
| Application of concepts | Each NN step is explicitly mapped to a named LA concept from §3, with **no conceptual errors** (check every shape). |
| Significance & benefits | Show *why* the matrix view matters: (a) **batching** turns m slow loops into one GEMM, which is why GPUs make deep learning practical; (b) backprop reuses forward results, so a gradient costs only a small constant multiple of a forward pass, versus perturbing each weight one at a time; (c) the collapse argument shows **why** non-linearity is essential; (d) the regression link shows NNs generalize something we already trust. Give numbers where possible (e.g. FLOP counts for a small network). |
| Delivery & language | Scholarly vocabulary used precisely: affine, composition, Jacobian, element-wise, outer product, rank-1, conformant. |
| Slide quality & originality | Our own diagrams and our own worked example. No 3B1B screenshots. |
| Visuals & equations | Every equation in LaTeX in LA4CS notation, **with shapes annotated under the matrices**. Charts: activation-function plots, decision boundaries, loss curve. |
| Q&A defense | §9 question bank with answers that every member can give; split ownership evenly. |

The brief's own examples of "interconnections" map directly onto us: *"Scalar multiplication ⟹ mapping of non-linear functions (e.g., in Neural Networks)"*. **Explicitly reproduce this chain of reasoning in the deck.**

---

## 5. Research questions (in priority order)

**P1: must have**
1. Neuron = dot product + bias + activation: **w**^T**x** + b as a hyperplane (geometric meaning of a single neuron; link to the Ch. 2 dot product).
2. Layer = matrix–vector product. Row picture (each neuron is a row **w**_i^T doing a dot product) versus column picture (the output is a linear combination of the columns of **W**). Both are from Ch. 2.
3. **The collapse theorem:** without φ, **W**^{(2)}(**W**^{(1)}**x** + **b**^{(1)}) + **b**^{(2)} = (**W**^{(2)}**W**^{(1)})**x** + (**W**^{(2)}**b**^{(1)} + **b**^{(2)}), which is one affine map, so depth adds nothing. Add the rank bound. Optional bridge: the homogeneous-coordinates trick [**W** **b**] [**x**; 1] makes the affine map linear.
4. Why activations break this; compare ReLU, sigmoid, tanh (formula, derivative, range, pros and cons). XOR as the minimal example that a single affine layer cannot separate but one hidden layer can.
5. Backprop derived as the chain rule on matrices: where **W**^T, ⊙ and the outer product come from. Include a **shape-check table** for every term.
6. Gradient descent update; learning rate.
7. Batching: **X** with samples as columns, **Z** = **WX** + **b1**^T, cost in FLOPs, why GPUs.

**P2: strong add-ons (choose 1–2, depending on time)**
8. A linear 1-layer network with squared loss ≡ least squares (Ch. 10): the gradient **A**^T(**Ax** − **b**) = **0** gives the normal equations. Gradient descent vs the closed form. Convergence depends on the eigenvalues of **A**^T**A** (η < 2/λ_max), which ties together Ch. 10 and Ch. 11.
9. Vanishing/exploding gradients through the Ch. 11 lens: δ is multiplied repeatedly by (**W**^T and the φ′ factors). In the simplified case of the same **W** and linear activation this is like **A**^k = **SΛ**^k**S**^{-1}, and |λ| > 1 explodes while |λ| < 1 vanishes. **Mark it clearly as intuition, not an exact theorem** (real networks have different **W** per layer, and φ′ ≠ 1).
10. Null space / rank interpretation: 𝒩(**W**^{(1)}) = input variations the first layer cannot see; a narrow layer is a rank bottleneck (autoencoder intuition).

**P3: only if time permits, otherwise keep for Q&A**
11. Universal approximation theorem: the statement only, intuitively.
12. Initialization (Xavier/He) as keeping the variance, or roughly the scale of singular values, near 1.
13. Pointer: convolution as a structured (Toeplitz) matrix; attention as **QK**^T. Ending on a "where this goes next" slide is enough.

**"Further exploration" from the brief:** implement a tiny network (NumPy, about 40 lines, matrix form only, no frameworks) on XOR or a 2D toy dataset; train it with identity / sigmoid / tanh / ReLU activations; plot **decision boundaries** and **loss curves** side by side. Report the actual observed results. **Do not fabricate numbers.** If code cannot be run, give the code and mark the results as "to be generated".

---

## 6. Sources

Starting points: the professor's link to the 3Blue1Brown neural-network playlist (`https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi`) and the "Neural Net in Matrix Form" hint. Suggested additional authoritative sources (verify, cite and prefer primary sources):
- Michael Nielsen, *Neural Networks and Deep Learning*, Ch. 1–2 (matrix-form backprop, the "four equations")
- Goodfellow, Bengio & Courville, *Deep Learning*, Ch. 6 (feed-forward networks, back-propagation)
- Parr & Howard, "The Matrix Calculus You Need for Deep Learning"
- Stanford CS231n notes (backprop, vectorized gradients, activation functions)
- Glorot & Bengio (2010) on initialization and vanishing gradients; Hornik/Cybenko for universal approximation

The research may follow other sources. Cite everything that is used in the report. Slides need only a short references slide.

---

## 7. Slide design rules: "minimize the mental steps"

The team's main request: **each slide should be reachable from the previous one in a single small step.**

1. **One idea per slide.** The title states the idea as a sentence ("A layer is just **Wx** + **b**"), not as a topic label ("Layers").
2. **Bridge first.** Each slide opens with the object the previous slide ended on, then changes exactly *one* thing (adds a symbol, a dimension, or a step).
3. **One running example through the whole deck.** Use a tiny fixed network, e.g. 2 inputs → 2 hidden (ReLU) → 1 output, on XOR, with concrete small integer weights. Every new equation is shown first on these numbers, then in general.
4. **Shapes under every matrix**, e.g. **W**^{(1)} (2×2), **x** (2×1). Shape-checking is the audience's anchor, and it is pure Week 2 conformance.
5. **Persistent colour coding**, the same on every slide and diagram: inputs/activations, weights, biases, pre-activations, errors **δ**.
6. **No symbol before its definition.** Introduce at most 2 new symbols per slide. Keep a small "symbol strip" in a corner that grows as symbols are introduced.
7. **"Recall from Week X" tags** (small badge) whenever an old concept is used. This shows interconnection without re-teaching.
8. **Forward-then-backward mirror:** the backprop slides reuse the exact forward-pass diagram with the arrows reversed and **W** → **W**^T.
9. **Visuals over text:** ≤ 25 words of prose per slide. Equations and diagrams carry the content; detail goes in the speaker notes.
10. **Every slide gets speaker notes:** what to say, about 45–60 s, and which team member says it.

### Draft storyline (≈ 10 min; research may refine it)

| # | Slide (idea as title) | LA bridge | ~time |
|---|---|---|---|
| 1 | Title + one-sentence thesis: "A neural network is a chain of matrix multiplications with a bend in between" | | 0:20 |
| 2 | A neuron is a dot product: φ(**w**^T**x** + b) | Wk 2 dot product | 0:50 |
| 3 | Stack neurons as rows ⟹ a layer is **Wx** + **b** (row vs column picture) | Wk 2 row/column pictures | 0:50 |
| 4 | Running example forward pass with numbers and shapes | Wk 2 conformance | 0:50 |
| 5 | Without φ, depth collapses: **W**^{(2)}**W**^{(1)} is one matrix, plus the rank bound | Wk 1 linearity, Wk 5 rank | 0:50 |
| 6 | Activations bend space: ReLU / sigmoid / tanh plots; XOR solved | NEW | 1:00 |
| 7 | Batching: **X** → **Z** = **WX** + **b1**^T (why GPUs) | Wk 2 matrix mult., outer product | 0:45 |
| 8 | Loss + gradient descent: "walk downhill" | NEW | 0:45 |
| 9 | Backprop I: the chain rule gives **W**^T and ⊙ (mirror diagram) | Wk 3 transpose, Wk 2 Hadamard | 1:00 |
| 10 | Backprop II: ∂ℓ/∂**W** = **δa**^T is a rank-1 outer product | Wk 2 outer product, Wk 5 rank | 0:45 |
| 11 | Connection: 1-layer linear net = linear regression (Ch. 10); η and eigenvalues (Ch. 11) | Wk 10, 11 | 0:50 |
| 12 | Experiment results: activations compared (decision boundaries + loss curves) | | 0:45 |
| 13 | Takeaways: the 3 LA operations that power deep learning + references | | 0:20 |

---

## 8. Accuracy checklist (run before finalizing)

- [ ] Every product is conformant. Write out the shapes for every equation in §2.3 in the report.
- [ ] No `*`, `·` or `×` as multiplication anywhere, including figures and code comments shown on slides (in code, use `@` for matmul and say so).
- [ ] No σ for an activation. No 𝓛 for the loss. Layer indices in parentheses.
- [ ] Vectors are columns. Batches are columns (**X** ∈ ℝ^{n_0×m}). If a cited source uses rows-as-samples (**XW**), **convert it**; do not mix conventions.
- [ ] "Linear" vs "affine" is used correctly (**Wx** + **b** is affine).
- [ ] The vanishing-gradient/eigenvalue argument is labelled as a simplification.
- [ ] Backprop cost claim stated correctly (the gradient of a scalar loss costs about a constant multiple of one forward pass).
- [ ] All numbers in the running example are recomputed and consistent across slides.
- [ ] Experiment results are real or clearly marked as placeholders.

---

## 9. Q&A preparation (deliver as part of the report)

Produce **at least 15 likely questions with concise, correct answers**, grouped so the team can split ownership evenly (the rubric rewards *all* members answering). It must include:
- Why do we need a non-linear activation? (the collapse argument)
- Why **W**^T in backprop? What is its shape and why?
- Why is ∂ℓ/∂**W** an outer product, and what is its rank?
- Is **Wx** + **b** linear? (No, it is affine. Then show the homogeneous trick.)
- How is this different from linear regression / least squares in Week 10? Why not just solve the normal equations?
- Why ReLU instead of sigmoid? (vanishing gradient: sigmoid′ ≤ 1/4)
- What happens if all weights are initialized to the same value? (symmetry: every neuron learns the same thing)
- What does the null space of **W**^{(1)} mean for the network?
- Why do GPUs matter? (batched GEMM)
- Where do eigenvalues show up in training?
- What are the limits of the matrix view? (CNNs, attention: these are still matrix products, just structured)
- One "gotcha" question per slide, derived from that slide's content.

---

## 10. Deliverables and format expected from the next chat

1. **Research report** (markdown): sections following §5, in LA4CS notation, with inline citations, plus the §9 Q&A bank and the §8 checklist ticked.
2. **Worked running example**: all numbers, forward and backward, one full gradient-descent step, with shapes.
3. **Experiment**: runnable NumPy code (matrix form) plus figures or placeholders.
4. **Slide deck** following §7, with speaker notes per slide and member assignments left as `[Member A/B/C/D]`.
5. **Before building the slides, ask the user:** team size and names; the slide tool (PowerPoint with native equations, Google Slides, LaTeX Beamer, or another); whether to include P2 topics 8, 9 or both; and the visual style preference.

---

## 11. Original brief from the professor (verbatim, for reference)

> Topic: **Neural network expressed in linear algebra**
> a. Hint: Neural networks can be expressed as matrix operations where weights and biases are adjusted through backpropagation. Focus on how the dot product and matrix multiplication are used in forward propagation.
> b. Further Exploration: Implement a simple neural network and experiment with different activation functions.
> c. Neural Net in Matrix Form
> d. https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi (3B1B)
>
> The "Hint" and "Further Exploration" were generated by ChatGPT and are listed here for your consideration. You are not required to follow them unless you find them helpful. The links provided serve as starting points for your exploration. When presenting, remember that your classmates (and possibly the teaching team) may not be familiar with your chosen topic: define key terms; focus on clarity and effective information transfer; avoid overwhelming the audience with complex mathematics or programming; ensure you fully understand the material you are presenting.
