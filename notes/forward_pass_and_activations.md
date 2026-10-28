# Forward Pass and Activations: Neurons, Layers, the Collapse Theorem, and Activation Functions (handoff §5 P1 items 1–4)

Notation follows handoff §2. Vectors are bold column vectors, matrices bold uppercase in square brackets, transpose ^T, layer index ^{(l)}, activation φ (never σ), loss ℓ. There is no `*`, `·` or `×` for multiplication. Where a source writes σ for the logistic sigmoid, these notes write **sigmoid(z)**. Where a source uses rows as samples (**XW**), it has been converted to columns as samples (**WX**).

Source conventions to be aware of when citing:
- Goodfellow et al. write a hidden layer as **h** = g(**W**^T**x** + **c**), so their **W** is our (**W**^{(1)})^T. Their batch is rows as samples, written **XW**.
- Nielsen writes a^l = σ(w^l a^{l-1} + b^l), with entry w^l_{jk} in row j (output neuron) and column k (input neuron). That is the same orientation as our **W**^{(l)} ∈ ℝ^{n_l × n_{l-1}}.
- CS231n code is `np.dot(W1, x)`, with each neuron's weights in a row of W1. That is the same orientation as ours.

---

## Q1. A neuron is **w**^T**x** + b followed by φ: what it means geometrically

### Takeaway
A neuron computes the scalar z = **w**^T**x** + b, which is one dot product (Week 2) plus a bias, and then outputs φ(z). The set {**x** : **w**^T**x** + b = 0} is a hyperplane in ℝ^{n_0}. **w** is normal (perpendicular) to it. The sign of z tells you which side of the hyperplane **x** lies on, and b shifts the hyperplane away from the origin. A neuron with a hard threshold (a perceptron) is therefore exactly a linear classifier, one hyperplane splitting space into two half-spaces.

### Cited Findings
- CS231n describes a neuron like this: "each neuron performs a dot product with the input and its weights, adds the bias and applies the non-linearity." — [CS231n, Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)
- Nielsen's perceptron rule: output 0 if **w**^T**x** + b ≤ 0 and output 1 if **w**^T**x** + b > 0. Nielsen writes the dot product as w·x; here it is rewritten as **w**^T**x**. He describes the bias as "a measure of how easy it is to get the perceptron to output a 1." — [Nielsen, NNDL Ch. 1](http://neuralnetworksanddeeplearning.com/chap1.html)
- Nielsen's sigmoid neuron outputs sigmoid(**w**^T**x** + b) with sigmoid(z) = 1 / [1 + e^{−z}], "a smoothed out version of a step function". If φ were a step function, "the sigmoid neuron would *be* a perceptron." He also suggests reading outputs ≥ 0.5 as one class and < 0.5 as the other. — [Nielsen, NNDL Ch. 1](http://neuralnetworksanddeeplearning.com/chap1.html)
- For the hyperplane **x**^T**w** + b = 0, "the weight vector **w** is normal to the plane", and b/‖**w**‖ is the (signed) distance from the origin. Points with f(**x**) > 0 lie on the positive side and points with f(**x**) < 0 on the negative side. — [Wang, HMC E161 lecture notes, "Linear separation of a feature space"](https://pages.hmc.edu/ruye/e161/lectures/NeuralNetworks/node18.html). The source states b/‖**w**‖ without an absolute value, so it is a *signed* quantity. The unsigned distance is |b|/‖**w**‖. Sign conventions differ between sources, and here the hyperplane is nearest the origin at **x** = −b**w**/‖**w**‖².
- CS231n (in the linear classifier notes) says changing a row of **W** makes the corresponding line in input space "rotate in different directions", while the bias lets the classifiers "translate the lines". Without a bias, every line is forced through the origin. — [CS231n, Linear Classification](https://cs231n.github.io/linear-classify/)

### Inferences
- A one-line proof that **w** is the normal, using only Week 2: if **x**_1 and **x**_2 both lie on the hyperplane, then **w**^T[**x**_1 − **x**_2] = [−b] − [−b] = 0. So **w** is orthogonal to every direction inside the hyperplane. The hyperplane is a translate of 𝒩(**w**^T), the null space of the 1 × n_0 matrix **w**^T (Week 9 bridge). It has dimension n_0 − 1 when **w** ≠ **0**.
- For a smooth φ (sigmoid, tanh), the level sets z = constant are parallel hyperplanes. The output changes only as **x** moves along **w** and is constant along the hyperplane directions. A single neuron therefore "sees" only one direction of the input.
- Signed distance of **x** from the hyperplane: [**w**^T**x** + b]/‖**w**‖. So z is the distance scaled by ‖**w**‖.

### Gaps
- No primary textbook statement of "normal vector + signed distance" was fetched from Goodfellow et al. or Nielsen. Those books do not dwell on it, and the geometric statement comes from a university lecture page (HMC). The fact is elementary analytic geometry, and the class can verify it with the proof above.

---

## Q2. A layer is **W x** + **b**: the row picture and the column picture, with shapes

### Takeaway
Stack the n_l neurons' weight vectors as rows: **W**^{(l)} = [**w**_i^T] ∈ ℝ^{n_l × n_{l-1}}. Then **z**^{(l)} = **W**^{(l)}**a**^{(l-1)} + **b**^{(l)}, with **a**^{(l-1)} ∈ ℝ^{n_{l-1}}, **b**^{(l)}, **z**^{(l)} ∈ ℝ^{n_l}, and **a**^{(l)} = φ(**z**^{(l)}) applied element-wise.
- **Row picture:** z_i = **w**_i^T**a**^{(l-1)} + b_i. This is n_l separate neurons, each doing one dot product, so n_l hyperplanes at once.
- **Column picture:** **W**^{(l)}**a**^{(l-1)} = Σ_j a_j^{(l-1)} **w**_{:,j}. The pre-activation is a linear combination of the columns of **W**^{(l)}, weighted by the entries of the incoming activation vector, so it lies in 𝒞(**W**^{(l)}) + **b**^{(l)}.

Both pictures are the Week 2 row and column views of a matrix–vector product.

### Cited Findings
- Nielsen defines the weight matrix **w**^l with "the entry in the j^th row and k^th column is w^l_{jk}". Here j indexes the receiving neuron in layer l and k the sending neuron in layer l−1. His vectorized forward pass is a^l = σ(w^l a^{l-1} + b^l) (his Eq. 25), which in our notation is **a**^{(l)} = φ(**W**^{(l)}**a**^{(l-1)} + **b**^{(l)}). — [Nielsen, NNDL Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- Nielsen explains the j-then-k index order: indexing the other way would mean "we'd need to replace the weight matrix in Equation (25) by the transpose of the weight matrix." He calls the vectorized view a way of "escaping index hell, while remaining precise". He defines z^l ≡ w^l a^{l-1} + b^l as "the weighted input to the neurons in layer l". — [Nielsen, NNDL Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- CS231n: "np.dot(W1,x) evaluates the activations of all neurons in that layer", because each neuron's weights are one row of W1. Its 3-layer example has shapes **x** 3×1, **W**^{(1)} 4×3, **W**^{(2)} 4×4, **W**^{(3)} 1×4, and typically no activation on the output layer. — [CS231n, Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)
- Goodfellow et al. describe the standard layer as "an affine transformation controlled by learned parameters, followed by a fixed nonlinear function called an activation function", with **h** = g(**W**^T**x** + **c**). — [Goodfellow, Bengio & Courville, *Deep Learning*, Ch. 6 §6.1](https://www.deeplearningbook.org/contents/mlp.html). In our orientation this is **a**^{(1)} = φ(**W**^{(1)}**x** + **b**^{(1)}), with **W**^{(1)} = [their **W**]^T and **b**^{(1)} = their **c**.
- Nielsen uses ⊙ for the Hadamard product, (**s** ⊙ **t**)_j = s_j t_j. This is the element-wise operation that appears with φ′ in backpropagation. — [Nielsen, NNDL Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html)

Shape table (one sample):

| Object | Shape |
|---|---|
| **a**^{(l-1)} | n_{l-1} × 1 |
| **W**^{(l)} = [**w**_i^T] | n_l × n_{l-1} |
| **w**_i (row i, transposed) | n_{l-1} × 1 |
| **w**_{:,j} (column j) | n_l × 1 |
| **b**^{(l)}, **z**^{(l)}, **a**^{(l)} | n_l × 1 |

### Inferences
- **Row picture = geometry.** Each row **w**_i^T defines its own hyperplane (Q1), so a layer of n_l neurons carves input space with n_l hyperplanes. The output vector records the signed and scaled position of **x** relative to each one.
- **Column picture = subspace.** Before the bias, every possible **W**^{(l)}**a** lies in 𝒞(**W**^{(l)}), whose dimension is rank(**W**^{(l)}) ≤ min(n_l, n_{l-1}). Any change of input inside 𝒩(**W**^{(l)}) is invisible to the layer (a Week 9 bridge, and handoff P2 item 10).
- Under the column-picture reading, n_{l-1} ≥ n_l is a "compression" layer and n_{l-1} < n_l embeds the input in a higher-dimensional space. Since rank ≤ n_{l-1}, the embedded image is still at most n_{l-1}-dimensional until φ bends it.

### Gaps
- None material. The row/column picture terminology is Week 2 class material, and the neural-network sources state the row-per-neuron fact directly (CS231n, Nielsen).

---

## Q3. The collapse theorem, the rank bound, the homogeneous-coordinates trick, and affine vs linear

### Takeaway
Without φ, two layers give **W**^{(2)}[**W**^{(1)}**x** + **b**^{(1)}] + **b**^{(2)} = [**W**^{(2)}**W**^{(1)}]**x** + [**W**^{(2)}**b**^{(1)} + **b**^{(2)}], which is a single affine map **W̃x** + **b̃**. By induction, any depth of affine layers is one affine layer, so depth adds no expressiveness. It can only remove some: rank(**W**^{(2)}**W**^{(1)}) ≤ min(rank **W**^{(1)}, rank **W**^{(2)}) ≤ n_1, so a narrow hidden layer is a bottleneck. In homogeneous coordinates the affine map becomes a genuinely linear one: [**W** **b**][**x**; 1]. Composition is then plain matrix multiplication of the augmented (n+1) × (n+1) block matrices, which is the Week 2 block-multiplication bridge.

### Cited Findings
- Goodfellow et al. (§6.1, XOR example): "if f^{(1)} were linear, then the feedforward network as a whole would remain a linear function of its input". Hence "we must use a nonlinear function to describe the features." — [Goodfellow et al., *Deep Learning*, Ch. 6 §6.1](https://www.deeplearningbook.org/contents/mlp.html). The quote is from the WebFetch extraction and has been lightly cleaned of PDF ligature artefacts.
- CS231n: without the non-linearity, two stacked weight matrices "could be collapsed to a single matrix", so the model would again be linear in its input. — [CS231n, Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)
- The CS231n "bias trick" extends **x** "with one additional dimension that always holds the constant 1". After that, "the extra column that **W** now corresponds to the bias **b**", and the score becomes one matrix multiply. — [CS231n, Linear Classification](https://cs231n.github.io/linear-classify/). In our notation: **W x** + **b** = [**W** **b**][**x**; 1], with [**W** **b**] ∈ ℝ^{n_l × (n_{l-1}+1)}.
- Goodfellow et al. explicitly call the layer map "an affine transformation" (not "linear"). — [Goodfellow et al., Ch. 6](https://www.deeplearningbook.org/contents/mlp.html)

### Inferences (derivations the team can defend)
- **Definitions (Week 1 bridge).** f is *linear* if f(α**x** + β**y**) = αf(**x**) + βf(**y**) for all scalars α, β, which forces f(**0**) = **0**. f is *affine* if f(**x**) = **A x** + **c**, that is, a linear map followed by a translation. **W x** + **b** with **b** ≠ **0** is affine but not linear, because it sends **0** to **b**.
- **Collapse, general depth.** Define **W̃** = **W**^{(L)}⋯**W**^{(1)} and **b̃** = Σ_{l=1}^{L} **W**^{(L)}⋯**W**^{(l+1)}**b**^{(l)} (empty product = **I**). Then the identity-activation network equals **W̃x** + **b̃**. The proof is by induction on L, using associativity and distributivity of matrix multiplication.
- **Homogeneous form of composition.** Let **M**^{(l)} = [ [**W**^{(l)} **b**^{(l)}] ; [**0**^T 1] ] ∈ ℝ^{(n_l+1) × (n_{l-1}+1)}. Then **M**^{(2)}**M**^{(1)} = [ [**W**^{(2)}**W**^{(1)}  **W**^{(2)}**b**^{(1)} + **b**^{(2)}] ; [**0**^T 1] ]. The collapse theorem is just block multiplication, and the collapsed bias appears automatically in the top-right block.
- **Rank bottleneck.** With **W**^{(1)} ∈ ℝ^{n_1 × n_0} and **W**^{(2)} ∈ ℝ^{n_2 × n_1}, rank(**W**^{(2)}**W**^{(1)}) ≤ min(rank **W**^{(1)}, rank **W**^{(2)}) ≤ min(n_0, n_1, n_2). This is the Week 5 rank property. So a deep linear network with a width-k layer anywhere can only represent affine maps whose matrix has rank ≤ k. Depth without φ adds no functions and a narrow layer *removes* some: for example 𝒞(**W̃**) ⊆ 𝒞(**W**^{(L)}), and 𝒩(**W**^{(1)}) ⊆ 𝒩(**W̃**).
- **Why this matters for the brief.** Since the collapsed network is affine, its decision boundary for any threshold is still a single hyperplane (Q1), and that is exactly what fails on XOR (Q4).
- **What depth without φ still changes.** Factorized parameterization changes the optimization dynamics but not the set of representable functions. This is a well-known result from the deep-linear-network literature, but no source was fetched for it, so see Gaps before using it.

### Gaps
- No single "authoritative" textbook sentence for the rank bound in a neural-network context was fetched. It is standard linear algebra (LA4CS Ch. 5, class material) and should be cited to the course text.
- The claim that deep linear networks have different training dynamics despite equal expressiveness (e.g. Saxe et al. 2014) was not verified in this session. Leave it out or verify it separately.

---

## Q4. Activation functions (ReLU, sigmoid, tanh) and the XOR example

### Takeaway
A non-linear φ applied element-wise between affine maps breaks the collapse. ReLU is the modern default for hidden layers: it is cheap, has gradient 1 when active, and gives sparse outputs, but units can "die". Sigmoid and tanh saturate. sigmoid′ ≤ 1/4 and sigmoid is not zero-centred. tanh′ ≤ 1 and tanh is zero-centred, being a rescaled sigmoid: tanh(z) = 2 sigmoid(2z) − 1. XOR is the minimal demonstration. No single affine map plus threshold separates it, while a 2-unit ReLU hidden layer maps the four points to a configuration a hyperplane can separate.

### Comparison table (formulas checked numerically in this session, see Verification)

| | ReLU | sigmoid (logistic) | tanh |
|---|---|---|---|
| φ(z) | max{0, z} | 1 / [1 + e^{−z}] | [e^{z} − e^{−z}] / [e^{z} + e^{−z}] |
| φ′(z) | 1 if z > 0, 0 if z < 0 (undefined at 0; use 0 or 1 by convention) | sigmoid(z)[1 − sigmoid(z)] | 1 − tanh²(z) |
| Range | [0, ∞) | (0, 1) | (−1, 1) |
| max φ′ | 1 (whenever active) | 1/4, at z = 0 | 1, at z = 0 |
| φ(0) | 0 | 1/2 | 0 |
| Saturates? | Only on the negative side (hard 0, "dying") | Both sides | Both sides |
| Zero-centred? | No (outputs ≥ 0) | No | Yes |
| Exact zeros (sparsity) | Yes | No | No |
| Cost | A comparison, with no exponential | An exponential | Exponentials |

Relationship: tanh(z) = 2 sigmoid(2z) − 1, so tanh′(z) = 4 sigmoid′(2z), and that explains max tanh′ = 4 [1/4] = 1.

### Cited Findings: activations
- Goodfellow et al.: "the default recommendation is to use the rectified linear unit", g(z) = max{0, z}. ReLUs are "easy to optimize because they are so similar to linear units". The derivative is 1 wherever the unit is active and the second derivative is 0 almost everywhere. Drawback: they "cannot learn via gradient-based methods on examples for which their activation is zero." — [Goodfellow et al., Ch. 6 (§6.1 p. 170; §6.3.1)](https://www.deeplearningbook.org/contents/mlp.html)
- Goodfellow et al. on sigmoid saturation: sigmoidal units saturate for large |z|, which "can make gradient-based learning very difficult", so their use as hidden units is discouraged. tanh(z) = 2 sigmoid(2z) − 1, and tanh "typically performs better" because it resembles the identity near 0 (tanh(0) = 0, while sigmoid(0) = 1/2). — [Goodfellow et al., Ch. 6 §6.3.2, p. 191](https://www.deeplearningbook.org/contents/mlp.html). The p. 191 page number comes from the WebFetch summary of the HTML, so check it against the print edition.
- CS231n on sigmoid: its gradients "saturate and kill" the learning signal near 0 or 1, and its outputs are not zero-centred, which causes zig-zagging weight updates. On tanh: tanh(x) = 2 sigmoid(2x) − 1, and "tanh non-linearity is always preferred to the sigmoid nonlinearity." — [CS231n, Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)
- CS231n on ReLU: "It was found to greatly accelerate" SGD convergence, by "a factor of 6" in Krizhevsky et al. (2012), which CS231n attributes to its linear, non-saturating form. It is cheap: thresholding at zero versus exponentials. Drawback: units can die, and "as much as 40% of your network can be 'dead'" if the learning rate is too high. Recommendation: use ReLU, monitor dead units, try Leaky ReLU or Maxout. Summary line: "Never use sigmoid" (in hidden layers). — [CS231n, Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/). The factor of 6 is CS231n reporting Krizhevsky et al. It was not checked against the AlexNet paper itself.
- Nair & Hinton (2010) introduced rectified linear units to RBMs. Binary units "can be generalized by replacing each binary unit by an infinite number of copies", which "can be approximated efficiently by noisy, rectified linear units". Σ_{i=1}^{N} sigmoid(x − i + 0.5) ≈ log(1 + e^{x}) (softplus). For discrimination they use y = max(0, x), and "rectified linear units preserve information about relative intensities". — [Nair & Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines", ICML 2010](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf). The paper's own symbol σ there denotes the logistic. The extraction gave no page numbers. ICML 2010 proceedings pages are commonly cited as 807–814, but that was not verified here.
- Glorot, Bordes & Bengio (2011): rectifiers create "sparse representations with true zeros". "After uniform initialization of the weights, around 50% of hidden units continuous output values are real zeros." The reported average exact sparsity of hidden layers is 83.4% on MNIST, 72.0% on CIFAR10, 68.0% on NISTP and 73.8% on NORB. Advantages: "Computation is linear on this subset" of active neurons, "gradients flow well on the active paths of neurons", and "there is no need for computing the exponential function". Concerns: "the hard saturation at 0 may hurt optimization by blocking gradient back-propagation" (their experiments suggested it did not hurt supervised training) and "unbounded behavior of the activations", for which they used an L1 penalty on activations. — [Glorot, Bordes & Bengio, "Deep Sparse Rectifier Neural Networks", AISTATS 2011, JMLR W&CP 15:315–323](https://proceedings.mlr.press/v15/glorot11a.html) ([PDF mirror used for reading](https://papers.baulab.info/papers/also/Glorot-2011.pdf))

### Cited Findings: XOR
- **Linear model fails.** Fitting the linear model f(**x**) = **w**^T**x** + b by mean squared error on the four XOR points gives **w** = **0**, b = 1/2, so it outputs 0.5 everywhere. — [Goodfellow et al., §6.1, p. 168](https://www.deeplearningbook.org/contents/mlp.html). *Verified in this session* by least squares: **w** ≈ [0, 0]^T (residual entries ~10^{−16}), b = 0.5.
- **Why it fails, in their words.** "A linear model must apply a fixed coefficient w_2 to x_2". It cannot let x_1 change how x_2 affects the output: when x_1 = 0 the output must rise with x_2, and when x_1 = 1 it must fall. — [Goodfellow et al., §6.1](https://www.deeplearningbook.org/contents/mlp.html)
- **Network solution (Eq. 6.3).** f(**x**) = **w**^T max{0, **W**^T**x** + **c**} + b with **W** = [[1, 1], [1, 1]], **c** = [0, −1]^T, **w** = [1, −2]^T, b = 0. On the batch it produces [0, 1, 1, 0], which is XOR. — [Goodfellow et al., §6.1, p. 170–171](https://www.deeplearningbook.org/contents/mlp.html)
- **Fig. 6.1 interpretation.** The learned features map [1, 0]^T and [0, 1]^T "to a single point in feature space", where a linear model can separate the classes. Goodfellow et al. also note the clean integer solution is for illustration, and gradient descent typically finds a different, non-integer solution. — [Goodfellow et al., §6.1](https://www.deeplearningbook.org/contents/mlp.html)
- **History.** Minsky & Papert's *Perceptrons* (1969) proved limits of single-layer perceptrons. Parity (XOR is the 2-input case) and connectedness are the central examples, and "the single-layer perceptron could not compute parity under the condition of conjunctive localness". The book is "often thought to have caused a decline in neural net research in the 1970s and early 1980s". That causal story is contested: critics such as Pollack say the community over-read a narrow result, the authors attributed the decline to inherent problems, and Rumelhart et al. later called the authors' "pessimism about learning in multilayer machines … misplaced". — [Wikipedia, "Perceptrons (book)"](https://en.wikipedia.org/wiki/Perceptrons_(book)). This is a secondary source and Wikipedia itself flags parts as needing citations. Present it as "commonly told history", not as settled fact.

### XOR restated in our notation (columns as samples), verified numerically in this session

Network 2 → 2 (ReLU) → 1 (identity):
- **W**^{(1)} = [[1, 1], [1, 1]] (2 × 2). This equals Goodfellow's **W**^T, and since that matrix is symmetric the entries are unchanged.
- **b**^{(1)} = [0, −1]^T (2 × 1).
- **W**^{(2)} = [1 −2] (1 × 2), Goodfellow's **w**^T.
- b^{(2)} = 0.

Batch **X** = [**x**_1 **x**_2 **x**_3 **x**_4] = [[0, 0, 1, 1], [0, 1, 0, 1]] (2 × 4), for inputs [0,0]^T, [0,1]^T, [1,0]^T, [1,1]^T.

- **Z**^{(1)} = **W**^{(1)}**X** + **b**^{(1)}**1**^T = [[0, 1, 1, 2], [−1, 0, 0, 1]] (2 × 4)
- **A**^{(1)} = ReLU(**Z**^{(1)}) = [[0, 1, 1, 2], [0, 0, 0, 1]] (2 × 4)
- **Ŷ** = **W**^{(2)}**A**^{(1)} + b^{(2)}**1**^T = [0 1 1 0] (1 × 4), which is XOR.

These match Goodfellow's transposed intermediates exactly. Their **XW** = [[0,0],[1,1],[1,1],[2,2]], **XW** + **c** = [[0,−1],[1,0],[1,0],[2,1]], and ReLU gives [[0,0],[1,0],[1,0],[2,1]]. Each is the transpose of our matrix.

### Inferences (proofs and geometry for slides and Q&A)
- **Proof that no affine map plus threshold separates XOR.** Suppose ŷ = 1 iff **w**^T**x** + b > 0. The four points require:
  1. [0,0]^T → 0, so b ≤ 0.
  2. [1,1]^T → 0, so w_1 + w_2 + b ≤ 0.
  3. [1,0]^T → 1, so w_1 + b > 0.
  4. [0,1]^T → 1, so w_2 + b > 0.

  Adding (3) and (4) gives w_1 + w_2 + 2b > 0. Adding (1) and (2) gives w_1 + w_2 + 2b ≤ 0. That is a contradiction, so no such **w**, b exists. Geometrically, the segments joining the two classes ([0,0]–[1,1] and [1,0]–[0,1]) cross at [1/2, 1/2]. The midpoint argument is the same inequality sum divided by 2. By the collapse theorem this also rules out *any* depth of affine layers plus a final threshold.
- **A linear-algebra reading of the hidden layer.** **W**^{(1)} = [[1,1],[1,1]] has **rank 1**: 𝒞(**W**^{(1)}) = span{[1,1]^T} and 𝒩(**W**^{(1)}) = span{[1,−1]^T}. So **W**^{(1)}**x** = s[1,1]^T with s = x_1 + x_2 ∈ {0, 1, 2}. The layer is blind to the direction [1, −1]^T, which is exactly the direction separating [1,0]^T from [0,1]^T. That is fine here, because XOR depends only on s. Without ReLU, the output would be an affine function of s, so ŷ(s) would take values in arithmetic progression at s = 0, 1, 2. But the target values 0, 1, 0 are not in arithmetic progression, which is another one-line proof that φ is needed.
- **What ReLU does geometrically.** The bias puts the two hidden units at s and s − 1. ReLU clips the second at 0, so h_2 = max{0, s − 1} "switches on" only at s = 2. The four inputs land at hidden points [0,0]^T (class 0), [1,0]^T (twice, class 1) and [2,1]^T (class 0). In hidden space the output is ŷ = h_1 − 2h_2: it is 0 on the line h_1 = 2h_2 (which contains both class-0 points) and 1 at [1,0]^T. The hyperplane h_1 − 2h_2 = 1/2 now separates the classes. So the hidden layer is an affine map, then a fold (ReLU bends the line s ↦ [s, s−1] into a broken line), and the fold makes the classes linearly separable.
- **ReLU as piecewise linear.** On each region where the pattern of active units is fixed, a ReLU network *is* an affine map. That follows from Glorot et al.'s "computation is linear on this subset". The non-linearity comes from switching between different affine pieces across regions bounded by the hidden neurons' hyperplanes (Q1). For XOR the pieces are s ≤ 1 (ŷ = s) and s ≥ 1 (ŷ = s − 2(s − 1) = 2 − s), giving a "tent" in s.

### Phrasing the brief's interconnection: "scalar multiplication ⟹ mapping of non-linear functions"
- A precise version of the chain (an inference built from the cited facts above):
  1. **Scalar multiplication and addition** build every linear combination (Week 2).
  2. A neuron is a dot product **w**^T**x** = Σ_j w_j x_j, a linear combination of inputs. A layer is **W x**, a linear combination of the columns of **W** with scalars x_j (column picture).
  3. Compositions of such maps (plus biases) remain affine: the collapse theorem, by associativity of matrix multiplication.
  4. Hence such maps can only model affine functions of the input, with hyperplane decision boundaries.
  5. Inserting an element-wise non-linear φ between the linear-combination steps breaks additivity, φ(**u** + **v**) ≠ φ(**u**) + φ(**v**) in general.
  6. The composite **x** ↦ **W**^{(2)}φ(**W**^{(1)}**x** + **b**^{(1)}) + **b**^{(2)} can therefore map inputs non-linearly, as in XOR. With one hidden layer of sufficient width it can approximate a very broad class of functions, which CS231n states as "Neural Networks with at least one hidden layer are *universal approximators*" ([CS231n](https://cs231n.github.io/neural-networks-1/)).
- One-sentence slide version: "A network alternates the only two things we need — *weighted sums* (scalar multiplication + addition, i.e. matrix products) and an element-wise *bend* φ — and it is the bend that lets weighted sums express non-linear mappings."
- A subtle point useful for Q&A: ReLU *keeps* homogeneity for non-negative scalars, ReLU(αz) = α ReLU(z) for α ≥ 0, but it breaks additivity, e.g. ReLU(1) + ReLU(−1) = 1 ≠ ReLU(0) = 0. So a ReLU network without biases still satisfies f(α**x**) = αf(**x**) for α ≥ 0, yet it is not linear. Sigmoid and tanh break both properties. These follow from the definitions and need no external source.
- CS231n also notes that universal approximation "says little" about practical success, and that depth helps empirically even though it does not add representational power ([CS231n](https://cs231n.github.io/neural-networks-1/)). Say this if asked "why go deep if one hidden layer suffices?"

### Verification log (run in this session with NumPy)
- XOR forward pass in column form: **Z**^{(1)}, **A**^{(1)} and **Ŷ** = [0 1 1 0] exactly as listed above.
- Least-squares linear fit to XOR: **w** = [−3.2e−16, −7.5e−17]^T ≈ **0**, b = 0.5.
- On z ∈ [−10, 10] with a grid step of 1e−4: max sigmoid′ = 0.25, max tanh′ = 1.0, and max |tanh(z) − [2 sigmoid(2z) − 1]| = 3.3e−16 (floating-point zero).

### Gaps
- Krizhevsky et al.'s "factor of 6" was taken from CS231n only. The AlexNet paper was not opened.
- Nair & Hinton (2010) page numbers were not verified (commonly cited as 807–814).
- Goodfellow et al. page numbers (pp. 167–172 for §6.1, p. 191 for sigmoid and tanh) come from the WebFetch summary of the HTML chapter and should be checked against the print/PDF edition before going on a slide.
- Minsky & Papert history comes from Wikipedia (secondary) and the original book was not consulted. Phrase it cautiously ("commonly credited with…; the extent of its impact is debated").
- Leaky ReLU, Maxout, GELU and softplus were mentioned only in passing (CS231n) and were not researched in depth.
