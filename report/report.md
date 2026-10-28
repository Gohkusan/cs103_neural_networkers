# Matrix products plus a bend teach networks

A feed-forward neural network is a chain of alternating steps: an affine map $\mathbf{z}^{(l)}=\mathbf{W}^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}$, which is one matrix–vector product (or, for a batch, one matrix–matrix product), followed by an element-wise non-linear "bend" $\mathbf{a}^{(l)}=\varphi(\mathbf{z}^{(l)})$. Training runs the same chain backwards. **Backpropagation multiplies the error by $(\mathbf{W}^{(l+1)})^\mathsf{T}$, applies a Hadamard product $\odot\,\varphi'(\mathbf{z}^{(l)})$, and gets every weight gradient as an outer product $\boldsymbol{\delta}^{(l)}(\mathbf{a}^{(l-1)})^\mathsf{T}$ of rank 1** [[2]](http://neuralnetworksanddeeplearning.com/chap2.html). The bend is essential, not decorative. Without it, any depth of layers collapses by associativity into a single affine map $(\mathbf{W}^{(2)}\mathbf{W}^{(1)})\mathbf{x}+(\mathbf{W}^{(2)}\mathbf{b}^{(1)}+\mathbf{b}^{(2)})$, which cannot fit XOR. Our own run confirms this: the best affine fit predicts 0.5 at every XOR point, while a 2 → 2 (ReLU) → 1 network fits it exactly [[3]](https://www.deeplearningbook.org/contents/mlp.html). The matrix view also explains why training is practical. Backprop costs a small constant multiple of one forward pass (Baydin et al. give $c<6$, typically 2–3) [[16]](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf). Batching turns $m$ memory-bound matrix–vector products into one compute-friendly GEMM (general matrix–matrix multiply); on a 2-core CPU this measured about 6 to 7.5 times faster for the same 208 million FLOPs. Every step maps onto CS103 material. Week 10 least squares is exactly a one-layer linear network, and gradient descent on it converges if and only if $0<\eta<2/\lambda_{\max}(\mathbf{A}^\mathsf{T}\mathbf{A})$, which is a Week 11 eigenvalue argument [[23]](https://distill.pub/2017/momentum/). The same "powers of a matrix" idea gives the *intuition*, not a theorem, for vanishing and exploding gradients. Null spaces and the rank bound explain what a layer cannot see.

> **Notation used throughout** (handoff §2). Vectors are bold columns in square brackets and matrices are bold uppercase. $\mathbf{W}^{(l)}\in\mathbb{R}^{n_l\times n_{l-1}}$ (row $i$ holds neuron $i$'s weights). $\varphi$ is the activation, written in words when it is concrete: ReLU, tanh, logistic (sigmoid). **σ is never used for an activation**, because it denotes singular values. The loss is $\ell$, the error signal is $\boldsymbol{\delta}^{(l)}=\partial\ell/\partial\mathbf{z}^{(l)}$, the learning rate is $\eta$, and a batch is $\mathbf{X}=[\mathbf{x}_1\ \cdots\ \mathbf{x}_m]\in\mathbb{R}^{n_0\times m}$ (**samples as columns**). Calligraphic letters denote subspaces only. Products are written by juxtaposition. Sources that use other symbols (Nielsen's σ and $C$, Goodfellow's $\mathbf{h}=g(\mathbf{W}^\mathsf{T}\mathbf{x}+\mathbf{c})$, rows-as-samples $\mathbf{XW}$) have been converted.

---

## Part I: the core chain (handoff §5, P1 items 1–7)

## 1. A neuron is one dot product that draws a hyperplane

A **neuron** (one computing unit of the network) computes the scalar $z=\mathbf{w}^\mathsf{T}\mathbf{x}+b$, which is a Week 2 dot product plus a **bias** (a constant offset), and outputs $\varphi(z)$. CS231n puts it in one line: "each neuron performs a dot product with the input and its weights, adds the bias and applies the non-linearity" [[7]](https://cs231n.github.io/neural-networks-1/). With a hard threshold instead of a smooth $\varphi$, the unit is Nielsen's **perceptron** (a neuron that outputs 1 if $\mathbf{w}^\mathsf{T}\mathbf{x}+b>0$ and 0 otherwise). He describes the bias as "a measure of how easy it is to get the perceptron to output a 1". His smooth "sigmoid neuron" uses $\mathrm{logistic}(z)=1/[1+e^{-z}]$, "a smoothed out version of a step function" [[1]](http://neuralnetworksanddeeplearning.com/chap1.html).

Geometrically, the set $\{\mathbf{x}:\mathbf{w}^\mathsf{T}\mathbf{x}+b=0\}$ is a **hyperplane, and $\mathbf{w}$ is normal (perpendicular) to it**. The sign of $z$ tells you which side of the hyperplane $\mathbf{x}$ lies on [[12]](https://pages.hmc.edu/ruye/e161/lectures/NeuralNetworks/node18.html). The proof needs only Week 2. If $\mathbf{x}_1$ and $\mathbf{x}_2$ both lie on the hyperplane, then $\mathbf{w}^\mathsf{T}[\mathbf{x}_1-\mathbf{x}_2]=0$, so $\mathbf{w}$ is orthogonal to every direction inside it. The hyperplane is therefore a translate of $\mathcal{N}(\mathbf{w}^\mathsf{T})$ (a Week 9 bridge) and has dimension $n_0-1$. The bias moves the hyperplane off the origin. Its closest point to the origin is $-b\,\mathbf{w}/\lVert\mathbf{w}\rVert^2$, at unsigned distance $|b|/\lVert\mathbf{w}\rVert$. Sources state the signed distance with differing sign conventions. Without $b$, every boundary is forced through the origin. CS231n says that changing a row of $\mathbf{W}$ makes the line "rotate", while the bias lets it "translate" [[8]](https://cs231n.github.io/linear-classify/). A useful consequence: $z$ is the signed distance to the hyperplane scaled by $\lVert\mathbf{w}\rVert$. So a single neuron "sees" only one direction of the input, the direction of $\mathbf{w}$, and its output is constant along the hyperplane.

## 2. Stacking neurons as rows turns a layer into $\mathbf{Wx}+\mathbf{b}$

Put the $n_l$ neurons' weight vectors in the rows of $\mathbf{W}^{(l)}=[\mathbf{w}_i^\mathsf{T}]\in\mathbb{R}^{n_l\times n_{l-1}}$. The whole **layer** (a group of neurons that all read the same input) is then $\mathbf{z}^{(l)}=\mathbf{W}^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}$, followed by $\mathbf{a}^{(l)}=\varphi(\mathbf{z}^{(l)})$ applied element by element. This is exactly Nielsen's vectorised forward pass, which he calls a way of "escaping index hell, while remaining precise". His weight $w^l_{jk}$ sits in row $j$ (the receiving neuron) and column $k$ (the sending neuron), the same orientation as ours. He notes that the opposite indexing would force a transpose into the equation [[2]](http://neuralnetworksanddeeplearning.com/chap2.html). CS231n's code `np.dot(W1, x)` "evaluates the activations of all neurons in that layer" for the same reason [[7]](https://cs231n.github.io/neural-networks-1/). Goodfellow et al. name the layer precisely: "an affine transformation controlled by learned parameters, followed by a fixed nonlinear function called an activation function". Their $\mathbf{h}=g(\mathbf{W}^\mathsf{T}\mathbf{x}+\mathbf{c})$ is our $\varphi(\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)})$ with $\mathbf{W}^{(1)}$ equal to their $\mathbf{W}^\mathsf{T}$ [[3]](https://www.deeplearningbook.org/contents/mlp.html).

The two Week 2 pictures of a matrix–vector product give two readings of the same layer. In the **row picture**, $z_i=\mathbf{w}_i^\mathsf{T}\mathbf{a}^{(l-1)}+b_i$: the layer is $n_l$ neurons, each one a dot product, so $n_l$ hyperplanes carve up the input space at once. In the **column picture**, $\mathbf{W}^{(l)}\mathbf{a}^{(l-1)}=\sum_j a^{(l-1)}_j\,\mathbf{w}_{:,j}$: the output is a linear combination of the columns of $\mathbf{W}^{(l)}$, so before the bias it lies in $\mathcal{C}(\mathbf{W}^{(l)})$, whose dimension is at most $\min(n_l,n_{l-1})$. The row picture is the geometry and the column picture is the subspace, and each sets up a later section: the row picture leads to XOR in §4, and the column picture leads to the null-space reading in §10.

| Object | Shape |
|---|---|
| $\mathbf{a}^{(l-1)}$ | $n_{l-1}\times 1$ |
| $\mathbf{W}^{(l)}=[\mathbf{w}_i^\mathsf{T}]$ | $n_l\times n_{l-1}$ |
| $\mathbf{w}_i$ (row $i$, transposed) | $n_{l-1}\times 1$ |
| $\mathbf{w}_{:,j}$ (column $j$) | $n_l\times 1$ |
| $\mathbf{b}^{(l)},\ \mathbf{z}^{(l)},\ \mathbf{a}^{(l)}$ | $n_l\times 1$ |

## 3. Without $\varphi$, any depth collapses into one affine map

Remove the activation and two layers give

$$
\mathbf{W}^{(2)}\big[\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)}\big]+\mathbf{b}^{(2)}=\big[\mathbf{W}^{(2)}\mathbf{W}^{(1)}\big]\mathbf{x}+\big[\mathbf{W}^{(2)}\mathbf{b}^{(1)}+\mathbf{b}^{(2)}\big]=\tilde{\mathbf{W}}\mathbf{x}+\tilde{\mathbf{b}},
$$

which is one affine map. By induction, using associativity and distributivity, $L$ layers give $\tilde{\mathbf{W}}=\mathbf{W}^{(L)}\cdots\mathbf{W}^{(1)}$ and $\tilde{\mathbf{b}}=\sum_{l}\mathbf{W}^{(L)}\cdots\mathbf{W}^{(l+1)}\mathbf{b}^{(l)}$. **Depth without a non-linearity adds no expressive power.** Goodfellow et al. state that "if $f^{(1)}$ were linear, then the feedforward network as a whole would remain a linear function of its input" [[3]](https://www.deeplearningbook.org/contents/mlp.html). CS231n says the stacked matrices "could be collapsed to a single matrix" [[7]](https://cs231n.github.io/neural-networks-1/). The vocabulary matters here (Week 1). A map $f$ is *linear* if $f(\alpha\mathbf{x}+\beta\mathbf{y})=\alpha f(\mathbf{x})+\beta f(\mathbf{y})$, which forces $f(\mathbf{0})=\mathbf{0}$. Since $\mathbf{Wx}+\mathbf{b}$ sends $\mathbf{0}$ to $\mathbf{b}$, **a layer is affine, not linear**, and Goodfellow et al. use the word "affine" deliberately.

Depth without a bend can also *remove* expressive power. By the Week 5 rank property, $\mathrm{rank}(\mathbf{W}^{(2)}\mathbf{W}^{(1)})\le\min(\mathrm{rank}\,\mathbf{W}^{(1)},\mathrm{rank}\,\mathbf{W}^{(2)})\le\min(n_0,n_1,n_2)$. A width-$k$ layer anywhere in a linear stack therefore limits the whole network to affine maps whose matrix has rank at most $k$, with $\mathcal{C}(\tilde{\mathbf{W}})\subseteq\mathcal{C}(\mathbf{W}^{(L)})$ and $\mathcal{N}(\mathbf{W}^{(1)})\subseteq\mathcal{N}(\tilde{\mathbf{W}})$. The **homogeneous-coordinates trick** turns "affine" back into "linear". CS231n's "bias trick" appends a constant 1 to the input, so that $\mathbf{Wx}+\mathbf{b}=[\mathbf{W}\ \ \mathbf{b}]\begin{bmatrix}\mathbf{x}\\1\end{bmatrix}$ [[8]](https://cs231n.github.io/linear-classify/). Write each layer as the block matrix $\mathbf{M}^{(l)}=\begin{bmatrix}\mathbf{W}^{(l)}&\mathbf{b}^{(l)}\\\mathbf{0}^\mathsf{T}&1\end{bmatrix}\in\mathbb{R}^{(n_l+1)\times(n_{l-1}+1)}$. Week 2 block multiplication then gives $\mathbf{M}^{(2)}\mathbf{M}^{(1)}=\begin{bmatrix}\mathbf{W}^{(2)}\mathbf{W}^{(1)}&\mathbf{W}^{(2)}\mathbf{b}^{(1)}+\mathbf{b}^{(2)}\\\mathbf{0}^\mathsf{T}&1\end{bmatrix}$, and the collapsed bias appears automatically in the top-right block. In our running example (Part IV) the collapse is concrete: with identity activation, $\mathbf{W}^{(2)}\mathbf{W}^{(1)}=[-1\ \ -1]$ and $\mathbf{W}^{(2)}\mathbf{b}^{(1)}+b^{(2)}=2$, and both the two-layer and the one-layer computation output $[2\ 1\ 1\ 0]$.

## 4. An element-wise bend breaks the collapse and solves XOR

Insert a non-linear $\varphi$ between the affine maps and additivity fails: $\varphi(\mathbf{u}+\mathbf{v})\neq\varphi(\mathbf{u})+\varphi(\mathbf{v})$ in general. ReLU shows the subtlety well. It *keeps* positive homogeneity, $\mathrm{ReLU}(\alpha z)=\alpha\,\mathrm{ReLU}(z)$ for $\alpha\ge0$, but breaks additivity, since $\mathrm{ReLU}(1)+\mathrm{ReLU}(-1)=1\neq\mathrm{ReLU}(0)=0$. The three standard activations differ mainly in their derivatives, and the derivative is what backprop multiplies by:

| | ReLU | logistic (sigmoid) | tanh |
|---|---|---|---|
| $\varphi(z)$ | $\max\{0,z\}$ | $1/[1+e^{-z}]$ | $[e^{z}-e^{-z}]/[e^{z}+e^{-z}]$ |
| $\varphi'(z)$ | 1 if $z>0$, 0 if $z<0$ (we set $\mathrm{ReLU}'(0):=0$) | $\mathrm{logistic}(z)[1-\mathrm{logistic}(z)]$ | $1-\tanh^2(z)$ |
| Range | $[0,\infty)$ | $(0,1)$ | $(-1,1)$ |
| $\max\varphi'$ | 1 (when active) | **1/4** at $z=0$ | 1 at $z=0$ |
| Saturates? | only for $z<0$ ("dying") | both sides | both sides |
| Zero-centred? | no | no | yes |
| Cost | one comparison | one exponential | exponentials |

These facts were checked numerically: on a fine grid, max logistic′ = 0.25, max tanh′ = 1.0, and $\tanh(z)=2\,\mathrm{logistic}(2z)-1$ holds to $3.3\mathrm{e}{-16}$. A unit **saturates** (its output flattens so its derivative is near 0) for large $|z|$, which "can make gradient-based learning very difficult". Goodfellow et al. therefore make ReLU "the default recommendation" and note that tanh "typically performs better" than the logistic because it resembles the identity near 0 [[3]](https://www.deeplearningbook.org/contents/mlp.html). CS231n reports that ReLU "was found to greatly accelerate" stochastic gradient descent, by "a factor of 6" in Krizhevsky et al. It also warns that "as much as 40% of your network can be 'dead'" (stuck at output 0 for every input, so no gradient flows) when the learning rate is too high [[7]](https://cs231n.github.io/neural-networks-1/). Glorot, Bordes & Bengio found that rectifiers give "sparse representations with true zeros", with an average exact sparsity of 83.4% on MNIST, and that "computation is linear on this subset" of active units [[14]](https://proceedings.mlr.press/v15/glorot11a.html). A ReLU network is therefore **piecewise affine** (affine on each region where the pattern of active units is fixed). Its non-linearity comes from switching between pieces. Nair & Hinton introduced rectified units, observing that they "preserve information about relative intensities" [[13]](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf).

**XOR is the smallest problem that needs the bend.** Suppose a single threshold unit outputs 1 iff $\mathbf{w}^\mathsf{T}\mathbf{x}+b>0$. The four points require $b\le0$, $w_1+w_2+b\le0$, $w_1+b>0$ and $w_2+b>0$. Adding the last two gives $w_1+w_2+2b>0$, while adding the first two gives $w_1+w_2+2b\le0$, a contradiction. By §3 the same impossibility holds for any depth of affine layers. Goodfellow et al. show that the least-squares linear model on XOR gives $\mathbf{w}=\mathbf{0}$, $b=1/2$, which "outputs 0.5 everywhere", because "a linear model must apply a fixed coefficient $w_2$ to $x_2$" [[3]](https://www.deeplearningbook.org/contents/mlp.html). Their network solution, converted to our orientation, is $\mathbf{W}^{(1)}=\begin{bmatrix}1&1\\1&1\end{bmatrix}$, $\mathbf{b}^{(1)}=[0\ \ -1]^\mathsf{T}$, $\mathbf{W}^{(2)}=[1\ \ -2]$, $b^{(2)}=0$, and it outputs exactly $[0\ 1\ 1\ 0]$.

The linear-algebra reading of this hidden layer is the most defensible slide content in the deck. $\mathbf{W}^{(1)}$ has **rank 1**, with $\mathcal{C}(\mathbf{W}^{(1)})=\mathrm{span}\{[1\ 1]^\mathsf{T}\}$ and $\mathcal{N}(\mathbf{W}^{(1)})=\mathrm{span}\{[1\ \ -1]^\mathsf{T}\}$. So $\mathbf{W}^{(1)}\mathbf{x}=s\,[1\ 1]^\mathsf{T}$ with $s=x_1+x_2\in\{0,1,2\}$. Without ReLU, the output would be an affine function of $s$, so its values at $s=0,1,2$ would form an arithmetic progression. The targets 0, 1, 0 do not, which is a second one-line proof that $\varphi$ is needed. With ReLU, the different biases make the two identical rows do different jobs: $h_1=s$ and $h_2=\max\{0,s-1\}$, so the second unit "switches on" only at $s=2$. The four inputs land at hidden points $[0\ 0]^\mathsf{T}$, $[1\ 0]^\mathsf{T}$ (twice) and $[2\ 1]^\mathsf{T}$. There the output $h_1-2h_2$ is a "tent" in $s$ that separates the classes. Goodfellow et al. describe this as the features mapping $[1\ 0]^\mathsf{T}$ and $[0\ 1]^\mathsf{T}$ "to a single point in feature space" [[3]](https://www.deeplearningbook.org/contents/mlp.html). The story that Minsky & Papert's 1969 limits on single-layer perceptrons caused a decline in neural-network research is commonly told but contested, and the source is secondary [[15]](https://en.wikipedia.org/wiki/Perceptrons_(book)). If it appears at all, phrase it with that caution.

**The brief's interconnection, "scalar multiplication ⟹ mapping of non-linear functions", can be stated as a chain.** Scalar multiplication and addition build linear combinations (Week 2). A neuron is a linear combination of inputs, and a layer is a linear combination of the columns of $\mathbf{W}$. Compositions of such maps stay affine (§3), so their decision boundaries stay hyperplanes. An element-wise bend $\varphi$ between the steps breaks additivity, and the alternation of weighted sums and bends can then express non-linear mappings such as XOR. With enough width it can approximate a very broad class of functions [[7]](https://cs231n.github.io/neural-networks-1/).

## 5. Backpropagation is the chain rule run through $\mathbf{W}^\mathsf{T}$, $\odot$ and outer products

**Backpropagation** (the algorithm that computes the gradient of the loss with respect to every weight) uses one rule. A **Jacobian** $\mathbf{J}=\partial\mathbf{u}/\partial\mathbf{v}$ is the matrix of all partial derivatives $\partial u_i/\partial v_j$, with one row per output. For a scalar loss $\ell$ that depends on $\mathbf{v}$ only through $\mathbf{u}=g(\mathbf{v})$, the chain rule $\partial\ell/\partial v_j=\sum_i J_{ij}\,\partial\ell/\partial u_i$ becomes

$$
\frac{\partial\ell}{\partial\mathbf{v}}=\mathbf{J}^\mathsf{T}\frac{\partial\ell}{\partial\mathbf{u}}.
$$

The transpose appears because the sum runs down a *column* of $\mathbf{J}$. This is Goodfellow et al.'s Eq. 6.46, and they summarise: "The back-propagation algorithm consists of performing such a Jacobian-gradient product for each operation in the graph" [[3]](https://www.deeplearningbook.org/contents/mlp.html). Only two Jacobians are needed. For the affine map, $\partial\mathbf{z}/\partial\mathbf{a}=\mathbf{W}$ (a linear map's Jacobian is its own matrix). For the element-wise $\varphi$, $\partial\mathbf{a}/\partial\mathbf{z}=\mathrm{diag}(\varphi'(\mathbf{z}))$, a diagonal matrix, and $\mathrm{diag}(\varphi'(\mathbf{z}))\,\mathbf{v}=\varphi'(\mathbf{z})\odot\mathbf{v}$. Parr & Howard note that element-wise operations give a Jacobian that is "very often a diagonal matrix" [[11]](https://explained.ai/matrix-calculus/). Applying the rule layer by layer gives Nielsen's four equations in our notation [[2]](http://neuralnetworksanddeeplearning.com/chap2.html):

$$
\textbf{BP1: }\ \boldsymbol{\delta}^{(L)}=\frac{\partial\ell}{\partial\mathbf{a}^{(L)}}\odot\varphi'\big(\mathbf{z}^{(L)}\big),\qquad
\textbf{BP2: }\ \boldsymbol{\delta}^{(l)}=\Big[\big(\mathbf{W}^{(l+1)}\big)^\mathsf{T}\boldsymbol{\delta}^{(l+1)}\Big]\odot\varphi'\big(\mathbf{z}^{(l)}\big),
$$

$$
\textbf{BP3: }\ \frac{\partial\ell}{\partial\mathbf{b}^{(l)}}=\boldsymbol{\delta}^{(l)},\qquad
\textbf{BP4: }\ \frac{\partial\ell}{\partial\mathbf{W}^{(l)}}=\boldsymbol{\delta}^{(l)}\big(\mathbf{a}^{(l-1)}\big)^\mathsf{T}.
$$

BP4 comes from a single entry. $w^{(l)}_{ij}$ appears only in $z^{(l)}_i$, with $\partial z^{(l)}_i/\partial w^{(l)}_{ij}=a^{(l-1)}_j$, so $\partial\ell/\partial w^{(l)}_{ij}=\delta^{(l)}_i a^{(l-1)}_j$. The matrix of these entries is the **outer product** $\boldsymbol{\delta}\mathbf{a}^\mathsf{T}$ (Week 2). Nielsen's mnemonic is "$\partial C/\partial w=a_{\mathrm{in}}\delta_{\mathrm{out}}$". Column $j$ of $\boldsymbol{\delta}\mathbf{a}^\mathsf{T}$ is $a_j\boldsymbol{\delta}$, so $\mathcal{C}(\boldsymbol{\delta}\mathbf{a}^\mathsf{T})\subseteq\mathrm{span}\{\boldsymbol{\delta}\}$ and **the per-sample weight gradient has rank at most 1**, with equality iff $\boldsymbol{\delta}\neq\mathbf{0}$ and $\mathbf{a}\neq\mathbf{0}$. A ReLU input that is 0 produces a whole zero column, so weights fed by an inactive unit get no update from that sample.

The backward pass mirrors the forward pass exactly. Forward, the chain multiplies by $\mathbf{W}^{(l)}$ and then applies $\varphi$. Backward, it applies $\odot\varphi'$ and then multiplies by $(\mathbf{W}^{(l)})^\mathsf{T}$, mapping errors from $\mathbb{R}^{n_{l}}$ back to $\mathbb{R}^{n_{l-1}}$. This is the transpose of a product of layer Jacobians taken in reverse order, which is the Week 3 rule $(\mathbf{AB})^\mathsf{T}=\mathbf{B}^\mathsf{T}\mathbf{A}^\mathsf{T}$. Implementations never form the Jacobians. Johnson's CS231n handout gives an example in which a linear layer's Jacobian would need about 256 GB [[10]](https://cs231n.stanford.edu/handouts/derivatives.pdf). Every quantity the backward pass needs ($\mathbf{z}^{(l)}$, $\mathbf{a}^{(l-1)}$) was already stored during the forward pass.

**Layout convention.** We use the "same shape as the variable" (denominator) layout for gradients of the scalar $\ell$, so $\partial\ell/\partial\mathbf{W}$ has the shape of $\mathbf{W}$, as CS231n insists: "dW must be of the same size as W" [[9]](https://cs231n.github.io/optimization-2/). Parr & Howard use numerator layout, in which a scalar's gradient is a row: "many papers and software will use the denominator layout", which is "just transpose of the numerator layout" [[11]](https://explained.ai/matrix-calculus/). Do not copy their formulas without transposing. Rows-as-samples sources (Johnson's $\mathbf{y}=\mathbf{xW}$) must likewise be transposed, with the product order reversed, before use. **All matrix-form gradients here were checked against central finite differences**: a 3 → 4 → 2 network with a batch of 5 agreed to about $3\mathrm{e}{-10}$ (cross-entropy loss) and $6\mathrm{e}{-11}$ (squared loss).

### Shape-check table: every equation in handoff §2.3

| Equation (§2.3) | Term | Shape | Conformance |
|---|---|---|---|
| Forward, one sample | $\mathbf{W}^{(l)}\mathbf{a}^{(l-1)}$ | $n_l\times1$ | $(n_l\times n_{l-1})(n_{l-1}\times1)$ |
| | $\mathbf{z}^{(l)}=\mathbf{W}^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}$ | $n_l\times1$ | sum of two $n_l\times1$ |
| | $\mathbf{a}^{(l)}=\varphi(\mathbf{z}^{(l)})$ | $n_l\times1$ | element-wise, shape kept |
| Forward, batch | $\mathbf{W}^{(l)}\mathbf{A}^{(l-1)}$ | $n_l\times m$ | $(n_l\times n_{l-1})(n_{l-1}\times m)$ |
| | $\mathbf{b}^{(l)}\mathbf{1}^\mathsf{T}$ | $n_l\times m$ | outer product $(n_l\times1)(1\times m)$ |
| | $\mathbf{Z}^{(l)}=\mathbf{W}^{(l)}\mathbf{A}^{(l-1)}+\mathbf{b}^{(l)}\mathbf{1}^\mathsf{T}$ | $n_l\times m$ | sum of two $n_l\times m$ |
| BP1 | $\partial\ell/\partial\mathbf{a}^{(L)}$, $\varphi'(\mathbf{z}^{(L)})$ | $n_L\times1$ each | $\odot$ needs equal shapes |
| | $\boldsymbol{\delta}^{(L)}$ | $n_L\times1$ | |
| BP2 | $(\mathbf{W}^{(l+1)})^\mathsf{T}$ | $n_l\times n_{l+1}$ | transpose of $n_{l+1}\times n_l$ |
| | $(\mathbf{W}^{(l+1)})^\mathsf{T}\boldsymbol{\delta}^{(l+1)}$ | $n_l\times1$ | $(n_l\times n_{l+1})(n_{l+1}\times1)$ |
| | $\boldsymbol{\delta}^{(l)}=[\cdots]\odot\varphi'(\mathbf{z}^{(l)})$ | $n_l\times1$ | $(n_l\times1)\odot(n_l\times1)$ |
| | Jacobian $\mathrm{diag}(\varphi'(\mathbf{z}^{(l)}))$ (never formed) | $n_l\times n_l$ | |
| BP4 | $\partial\ell/\partial\mathbf{W}^{(l)}=\boldsymbol{\delta}^{(l)}(\mathbf{a}^{(l-1)})^\mathsf{T}$ | $n_l\times n_{l-1}$ | $(n_l\times1)(1\times n_{l-1})$, same as $\mathbf{W}^{(l)}$ ✔ |
| BP3 | $\partial\ell/\partial\mathbf{b}^{(l)}=\boldsymbol{\delta}^{(l)}$ | $n_l\times1$ | same as $\mathbf{b}^{(l)}$ ✔ |
| Batch backprop | $\boldsymbol{\Delta}^{(l)}=[(\mathbf{W}^{(l+1)})^\mathsf{T}\boldsymbol{\Delta}^{(l+1)}]\odot\varphi'(\mathbf{Z}^{(l)})$ | $n_l\times m$ | $(n_l\times n_{l+1})(n_{l+1}\times m)$, then $\odot$ |
| | $\partial\ell/\partial\mathbf{W}^{(l)}=\boldsymbol{\Delta}^{(l)}(\mathbf{A}^{(l-1)})^\mathsf{T}$ | $n_l\times n_{l-1}$ | $(n_l\times m)(m\times n_{l-1})$ |
| | $\partial\ell/\partial\mathbf{b}^{(l)}=\boldsymbol{\Delta}^{(l)}\mathbf{1}$ | $n_l\times1$ | $(n_l\times m)(m\times1)$ |
| GD update | $\mathbf{W}^{(l)}-\eta\,\partial\ell/\partial\mathbf{W}^{(l)}$ | $n_l\times n_{l-1}$ | scalar times matrix, then subtract equal shapes |

## 6. Gradient descent steps against the gradient, scaled by $\eta$

**Gradient descent** (repeatedly nudging every parameter downhill on the loss) updates $\mathbf{W}^{(l)}\leftarrow\mathbf{W}^{(l)}-\eta\,\partial\ell/\partial\mathbf{W}^{(l)}$. Here the **learning rate** $\eta$ (a positive step size) is chosen by hand. The negative gradient is the steepest-descent direction, and the proof uses only Week 2 tools. To first order, $\ell(\boldsymbol{\theta}+t\mathbf{u})\approx\ell(\boldsymbol{\theta})+t\,\mathbf{u}^\mathsf{T}\mathbf{g}$, and by Cauchy–Schwarz $\mathbf{u}^\mathsf{T}\mathbf{g}\ge-\lVert\mathbf{g}\rVert$ for unit $\mathbf{u}$, with equality iff $\mathbf{u}=-\mathbf{g}/\lVert\mathbf{g}\rVert$. For a matrix parameter, the same argument works with the Frobenius inner product $\mathrm{trace}(\mathbf{U}^\mathsf{T}\mathbf{G})$. Goodfellow et al.: "the negative gradient points directly downhill". They also warn that a step that is too large "can actually move uphill" when the curvature term $\mathbf{g}^\mathsf{T}\mathbf{Hg}$ is large. The optimal step along $\mathbf{g}$ is $\mathbf{g}^\mathsf{T}\mathbf{g}/\mathbf{g}^\mathsf{T}\mathbf{Hg}$, which in the worst case is $1/\lambda_{\max}$ [[4]](https://www.deeplearningbook.org/contents/numerical.html). This is the hook for §8. **Mini-batch stochastic gradient descent** (each step uses the average gradient over a small random batch) is Nielsen's $\mathbf{W}\to\mathbf{W}-(\eta/m)\sum_x\boldsymbol{\delta}^x(\mathbf{a}^x)^\mathsf{T}$ [[2]](http://neuralnetworksanddeeplearning.com/chap2.html). Goodfellow et al. stress that "back-propagation refers only to the method for computing the gradient", while gradient descent does the learning [[3]](https://www.deeplearningbook.org/contents/mlp.html).

**The cost claim, worded for the checklist:** for a scalar loss, backprop (reverse-mode automatic differentiation) computes the gradient with respect to *all* parameters for a small constant multiple of one forward pass. Baydin et al., citing Griewank & Walther, give "$c<6$ and typically $c\sim[2,3]$", while forward mode or finite differences need on the order of one evaluation per parameter. They call backpropagation "only a special case" of automatic differentiation [[16]](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf). Griewank adds the caveat that the bound ignores memory, because reverse mode stores the forward intermediates [[17]](https://papers.baulab.info/papers/also/Griewank-2012.pdf). A later paper notes that the practical factor can be "between 5 and 10" and is "highly program dependent" [[18]](https://arxiv.org/pdf/2202.08587). Nielsen's version for a network with "a million weights": finite differences need "a million forward passes", while the backward pass costs "about the same as the forward pass" [[2]](http://neuralnetworksanddeeplearning.com/chap2.html). Our FLOP count agrees (§7): for 784 → 128 → 10 with 101,770 parameters, forward plus backward is about 2.0 times the forward pass, against roughly 101,771 forward passes for one-sided finite differences, a ratio of about 50,000.

## 7. Batching turns $m$ matrix–vector products into one matrix–matrix product

Put $m$ samples side by side as columns and every matrix–vector product becomes one matrix–matrix product: $\mathbf{Z}^{(l)}=\mathbf{W}^{(l)}\mathbf{A}^{(l-1)}+\mathbf{b}^{(l)}\mathbf{1}^\mathsf{T}$. The bias **broadcast** (copying $\mathbf{b}$ into every column) is the outer product $\mathbf{b1}^\mathsf{T}=[\mathbf{b}\ \mathbf{b}\ \cdots\ \mathbf{b}]$, and column $j$ of $\mathbf{Z}$ is the single-sample pass $\mathbf{Wa}_j+\mathbf{b}$. Backward, the weight gradient $\boldsymbol{\Delta}^{(l)}(\mathbf{A}^{(l-1)})^\mathsf{T}=\sum_{j=1}^m\boldsymbol{\Delta}^{(l)}_{:,j}(\mathbf{a}^{(l-1)}_j)^\mathsf{T}$ is exactly the **sum of $m$ rank-1 outer products**, by the Week 2 column-row expansion of a matrix product. Its rank is at most $\min(m,n_l,n_{l-1})$. The bias gradient $\boldsymbol{\Delta}^{(l)}\mathbf{1}$ is the row sums, because the transpose of "copy into $m$ columns" is "add up the $m$ columns". CS231n gives the same rule in our column layout: for $\mathbf{D}=\mathbf{WX}$, the gradients are $d\mathbf{W}=d\mathbf{D}\,\mathbf{X}^\mathsf{T}$ and $d\mathbf{X}=\mathbf{W}^\mathsf{T}d\mathbf{D}$ [[9]](https://cs231n.github.io/optimization-2/). **Averaging decision:** with the mean loss $\ell=(1/m)\sum_j\ell_j$, the $1/m$ goes once into $\boldsymbol{\Delta}^{(L)}$, which matches Nielsen's $(\eta/m)\sum$ form and makes the step size independent of batch size [[2]](http://neuralnetworksanddeeplearning.com/chap2.html). The experiment (Part V) averages. The 4-sample running example (Part IV) uses a *sum*, which the slides must say.

Batching does **not** change the FLOP count. A product $(M\times K)(K\times N)$ costs $2MKN$ FLOPs, "a multiply and an add" per FMA (fused multiply–add) [[19]](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html). For 784 → 128 → 10 with $m=64$:

| Pass | Product | Shapes | FLOPs |
|---|---|---|---|
| Forward, layer 1 | $\mathbf{W}^{(1)}\mathbf{X}$ | $(128\times784)(784\times64)$ | $2(128)(784)(64)=12{,}845{,}056$ |
| Forward, layer 2 | $\mathbf{W}^{(2)}\mathbf{A}^{(1)}$ | $(10\times128)(128\times64)$ | $163{,}840$ |
| **Forward total** | | | **≈ 13.01 MFLOP** (million FLOPs) |
| Backward $\partial\ell/\partial\mathbf{W}^{(2)}$ | $\boldsymbol{\Delta}^{(2)}(\mathbf{A}^{(1)})^\mathsf{T}$ | $(10\times64)(64\times128)$ | $163{,}840$ |
| Backward error to layer 1 | $(\mathbf{W}^{(2)})^\mathsf{T}\boldsymbol{\Delta}^{(2)}$ | $(128\times10)(10\times64)$ | $163{,}840$ |
| Backward $\partial\ell/\partial\mathbf{W}^{(1)}$ | $\boldsymbol{\Delta}^{(1)}\mathbf{X}^\mathsf{T}$ | $(128\times64)(64\times784)$ | $12{,}845{,}056$ |
| **Backward total** | (no $\partial\ell/\partial\mathbf{X}$) | | **≈ 13.17 MFLOP** |

What batching changes is **arithmetic intensity** (FLOPs done per byte moved from memory). NVIDIA states that matrix–vector products (GEMV, general matrix–vector multiply) "are always memory limited; their arithmetic intensity is less than 1", while a V100 needs about 138.9 FLOPs per byte to be compute-bound [[19]](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html). Demmel's memory-hierarchy analysis gives the same ladder: BLAS (Basic Linear Algebra Subprograms) level 1 has a flops-to-memory ratio $q=2/3$, level 2 has $q\approx2$, and level 3 (matrix–matrix) has $q=n/2$ [[22]](https://people.eecs.berkeley.edu/~demmel/cs267-1995/lecture02.html), [[21]](https://www.netlib.org/blas/). For layer 1 at fp32, our minimum-traffic estimate is **0.50 FLOP/B at $m=1$, 20.2 at $m=64$ and 38.5 at $m=256$**. With $m=64$ the weights are read once and reused 64 times. This is the "why GPUs" argument: only level-3 work keeps their arithmetic units busy. State it honestly, though. NVIDIA estimates that a 4096-wide layer on an A100 is still memory-bound "for batch sizes 128 and below" [[20]](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html), so batching moves the work *toward* the compute-bound regime and does not guarantee saturation. NVIDIA's GEMM mapping also shows that the batch size is the summed $K$ dimension of the weight-gradient GEMM, which is the sum of $m$ outer products seen from the hardware side.

---

## Part II: connections to the CS103 syllabus (P2 items 8 and 9, then item 10)

## 8. A one-layer linear network is Week 10 least squares, and $\eta<2/\lambda_{\max}$ is Week 11

**Notation clash, to say out loud:** in this section $\mathbf{b}$ is the Ch. 10 right-hand side (the targets), not a bias. $\mathbf{A}\in\mathbb{R}^{m\times n}$ holds one training input per *row*, which is the transpose of our batch $\mathbf{X}$. On the slide, rename the problem to $\mathbf{X}^\mathsf{T}\mathbf{w}\approx\mathbf{y}$ so the clash disappears. With identity activation, no bias and squared loss, training minimises $\ell(\mathbf{x})=\tfrac12\lVert\mathbf{Ax}-\mathbf{b}\rVert^2$. The gradient is $\mathbf{A}^\mathsf{T}[\mathbf{Ax}-\mathbf{b}]$ ($n\times1$), and setting it to $\mathbf{0}$ gives **the normal equations $\mathbf{A}^\mathsf{T}\mathbf{A}\hat{\mathbf{x}}=\mathbf{A}^\mathsf{T}\mathbf{b}$**. The Hessian $\mathbf{A}^\mathsf{T}\mathbf{A}$ is positive semidefinite, so this stationary point is the global minimum. Goodfellow et al. work exactly this example with the update $\mathbf{x}\leftarrow\mathbf{x}-\epsilon[\mathbf{A}^\mathsf{T}\mathbf{Ax}-\mathbf{A}^\mathsf{T}\mathbf{b}]$ [[4]](https://www.deeplearningbook.org/contents/numerical.html). Our running example (Part IV, A4) shows the bridge in action: the normal equations for the best affine fit to XOR give $\boldsymbol{\theta}=[0\ 0\ 0.5]^\mathsf{T}$, the same 0.5-everywhere answer.

Now treat gradient descent as a Week 11 difference equation. Let $\mathbf{e}_k=\mathbf{x}_k-\hat{\mathbf{x}}$ and substitute $\mathbf{A}^\mathsf{T}\mathbf{b}=\mathbf{A}^\mathsf{T}\mathbf{A}\hat{\mathbf{x}}$. Then $\mathbf{e}_{k+1}=[\mathbf{I}-\eta\mathbf{A}^\mathsf{T}\mathbf{A}]\mathbf{e}_k$, so $\mathbf{e}_k=[\mathbf{I}-\eta\mathbf{A}^\mathsf{T}\mathbf{A}]^k\mathbf{e}_0$. Because $\mathbf{A}^\mathsf{T}\mathbf{A}=\mathbf{S\Lambda S}^{-1}$ is symmetric with $\lambda_i\ge0$, $\mathbf{S}^{-1}=\mathbf{S}^\mathsf{T}$ and each eigen-component decouples: $c_{i,k}=(1-\eta\lambda_i)^k c_{i,0}$. **Gradient descent converges for every start iff $0<\eta<2/\lambda_{\max}$** (assuming full column rank). The rate is $\max_i|1-\eta\lambda_i|$, and the best fixed step $\eta^*=2/(\lambda_{\min}+\lambda_{\max})$ gives rate $(\kappa-1)/(\kappa+1)$ with condition number $\kappa=\lambda_{\max}/\lambda_{\min}$. Goh (Distill) derives exactly this ("the iterations break apart") and calls $\kappa$ "a direct measure of pathological curvature" [[23]](https://distill.pub/2017/momentum/). Goodfellow et al. add that "when the Hessian has a poor condition number, gradient descent performs poorly" [[4]](https://www.deeplearningbook.org/contents/numerical.html). A NumPy check on a random $50\times3$ matrix ($\lambda\approx\{33.9,39.6,64.9\}$) confirmed that **the bound is sharp**: $\eta=0.9(2/\lambda_{\max})$ converged to $\lVert\mathbf{e}_{200}\rVert\approx10^{-16}$, while $1.05(2/\lambda_{\max})$ diverged to about $3\mathrm{e}7$. If $\mathbf{A}$ is rank-deficient, the components in $\mathcal{N}(\mathbf{A})$ never move. From $\mathbf{x}_0=\mathbf{0}$, gradient descent then reaches the minimum-norm solution $\mathbf{A}^+\mathbf{b}$, because every update lies in $\mathcal{C}(\mathbf{A}^\mathsf{T})$ (this is our inference).

*Why not just solve the normal equations?* There are three answers. Once $\varphi$ is non-linear or there are two or more layers, $\nabla\ell=\mathbf{0}$ is a non-linear system with no closed form, while gradient descent still applies. The linear case is the one place where gradient descent can be checked against an exact answer. Second, forming and factoring $\mathbf{A}^\mathsf{T}\mathbf{A}$ needs $n^2$ storage, which is infeasible for millions of parameters, whereas a gradient step needs two matrix–vector products. Third, $\mathbf{A}^\mathsf{T}\mathbf{A}$ squares the condition number. The last two are standard numerical facts that the notes flag as not source-verified (see "Verify before presenting"). [SIMPLIFICATION] Near a minimum, any smooth loss is approximately quadratic, so the same $2/\lambda_{\max}(\mathbf{H})$ bound and $\kappa(\mathbf{H})$ rate hold *locally* only [[4]](https://www.deeplearningbook.org/contents/numerical.html).

## 9. Vanishing and exploding gradients are matrix powers, as intuition only

> **[SIMPLIFICATION, not an exact theorem for feed-forward networks.]** The eigenvalue picture below assumes the *same* square $\mathbf{W}$ in every layer and identity activation. Real feed-forward networks have a different $\mathbf{W}^{(l)}$ per layer, $\varphi'\neq1$ varies with the input, $\mathbf{W}$ may be non-diagonalisable or have complex eigenvalues, and the rigorous tool is singular values or matrix norms. Label the slide "intuition".

The exact part comes first. BP2 can be rewritten as $\boldsymbol{\delta}^{(l)}=\mathbf{D}^{(l)}(\mathbf{W}^{(l+1)})^\mathsf{T}\boldsymbol{\delta}^{(l+1)}$ with $\mathbf{D}^{(l)}=\mathrm{diag}(\varphi'(\mathbf{z}^{(l)}))$, so

$$
\underset{n_1\times1}{\boldsymbol{\delta}^{(1)}}=\mathbf{D}^{(1)}(\mathbf{W}^{(2)})^\mathsf{T}\mathbf{D}^{(2)}(\mathbf{W}^{(3)})^\mathsf{T}\cdots\mathbf{D}^{(L-1)}(\mathbf{W}^{(L)})^\mathsf{T}\underset{n_L\times1}{\boldsymbol{\delta}^{(L)}},
$$

a product of $n_l\times n_l$ diagonals and $n_l\times n_{l+1}$ transposes. Under the simplification ($\mathbf{W}^{(l)}=\mathbf{W}$, $\mathbf{D}=\mathbf{I}$) this becomes $(\mathbf{W}^\mathsf{T})^{L-1}\boldsymbol{\delta}^{(L)}$. If $\mathbf{W}=\mathbf{S\Lambda S}^{-1}$, then $(\mathbf{W}^\mathsf{T})^k=(\mathbf{S}^{-1})^\mathsf{T}\mathbf{\Lambda}^k\mathbf{S}^\mathsf{T}$, which is still a diagonalisation, with the same eigenvalues as $\mathbf{W}$ but generally different eigenvectors. Expanding $\boldsymbol{\delta}^{(L)}=\sum_i c_i\mathbf{u}_i$ in the eigenvectors of $\mathbf{W}^\mathsf{T}$ gives $\boldsymbol{\delta}^{(1)}=\sum_i c_i\lambda_i^{L-1}\mathbf{u}_i$. **Components with $|\lambda_i|>1$ explode and those with $|\lambda_i|<1$ vanish.** This is the Fibonacci / $\mathbf{A}^k=\mathbf{S\Lambda}^k\mathbf{S}^{-1}$ reasoning from Ch. 11, and a NumPy check matched $(\mathbf{W}^\mathsf{T})^5$ to $10^{-13}$. The activation adds an *exact* bound: logistic′ ≤ 1/4, so each logistic layer shrinks the backward signal by at least a factor of 4. Over 10 layers that is $(1/4)^{10}\approx10^{-6}$ from $\varphi'$ alone. tanh′ ≤ 1, and ReLU′ ∈ {0, 1} does not shrink active units.

The picture becomes rigorous for **recurrent networks**, which genuinely reuse one $\mathbf{W}$. Pascanu, Mikolov & Bengio show that a largest eigenvalue magnitude below 1 is *sufficient* for long-term gradient components to vanish, and above 1 is *necessary* for them to explode. With $\lVert\mathrm{diag}(\varphi')\rVert\le\gamma$ the vanishing condition becomes $\lambda_1<1/\gamma$, with $\gamma=1/4$ for the logistic. Their proof uses norms, so in Q&A say "spectral radius / largest singular value" [[24]](https://ar5iv.labs.arxiv.org/html/1211.5063). Goodfellow et al. give the same $\mathbf{W}^t=\mathbf{V}\,\mathrm{diag}(\boldsymbol{\lambda})^t\,\mathbf{V}^{-1}$ argument and note that feed-forward networks, with a different matrix per layer, largely avoid the pure-power version [[5]](https://www.deeplearningbook.org/contents/optimization.html). This supports the "intuition" label. **The pairing with §8 is the strongest syllabus bridge in the deck.** Eigenvalues of $[\mathbf{I}-\eta\mathbf{A}^\mathsf{T}\mathbf{A}]$ decide whether *training iterations* converge, and eigenvalues of $\mathbf{W}^\mathsf{T}$ decide whether *signals through depth* survive. Both are Week 11 matrix powers, along two different axes: time and depth.

## 10. The null space of $\mathbf{W}^{(1)}$ is what the network cannot see

Every input splits uniquely as $\mathbf{x}=\mathbf{x}_r+\mathbf{x}_n$ with $\mathbf{x}_r\in\mathcal{C}((\mathbf{W}^{(1)})^\mathsf{T})$ and $\mathbf{x}_n\in\mathcal{N}(\mathbf{W}^{(1)})$, which are orthogonal complements (Ch. 9). Since $\mathbf{z}^{(1)}=\mathbf{W}^{(1)}\mathbf{x}_r+\mathbf{b}^{(1)}$, **two inputs that differ by a null-space vector produce identical activations in every later layer**, so the whole network cannot tell them apart. Because $\dim\mathcal{N}(\mathbf{W}^{(1)})=n_0-\mathrm{rank}(\mathbf{W}^{(1)})\ge n_0-n_1$, a first layer narrower than its input must discard at least $n_0-n_1$ directions. XOR shows that this can be harmless. The Goodfellow $\mathbf{W}^{(1)}$ is blind to $[1\ \ -1]^\mathsf{T}$, yet XOR depends only on $x_1+x_2$. Training respects the same geometry: $\partial\ell/\partial\mathbf{W}^{(1)}=\boldsymbol{\delta}^{(1)}\mathbf{x}^\mathsf{T}$, so every row of every update lies in $\mathrm{span}\{\mathbf{x}_j\}$, and weight directions orthogonal to all training inputs never change.

A narrow layer is a **rank bottleneck**: in the linear part, $\mathrm{rank}(\mathbf{W}^{(L)}\cdots\mathbf{W}^{(1)})\le\min_l n_l$. With a non-linear $\varphi$, all later layers are functions of only $n_l$ numbers. That reading is interpretive, not a theorem. An **autoencoder** (a network trained to reproduce its own input through a narrow middle layer) makes this concrete. Baldi & Hornik proved that a linear one with $p$ hidden units has "up to equivalence, a unique local and global minimum corresponding to an orthogonal projection" onto "the subspace generated by the first principal vectors", and that "all other critical points … are saddle points" [[26]](https://igb.uci.edu/~pfbaldi/publications/journals/1989/sdarticle.pdf). This can be stated without SVD: the optimum $\mathbf{W}^{(2)}\mathbf{W}^{(1)}$ is the Ch. 10 projection $\mathbf{P}=\mathbf{S}_p\mathbf{S}_p^\mathsf{T}$ onto the top-$p$ eigenvectors of the covariance matrix (Ch. 11). Only the subspace is determined. The individual rows of $\mathbf{W}^{(1)}$ can be mixed by any invertible $p\times p$ matrix.

---

## Part III: if time permits (P3 items 11–13, briefly)

**Universal approximation (item 11).** One hidden layer with a non-polynomial $\varphi$ can approximate any continuous function on a compact set to any accuracy, *if it is wide enough*. In LA terms, linear combinations of the ridge functions $\varphi(\mathbf{w}_i^\mathsf{T}\mathbf{x}+b_i)$ are dense. Cybenko (1989) proved it for sigmoidal $\varphi$, Hornik, Stinchcombe & White (1989) generalised it, and Leshno et al. (1993) showed it holds **iff $\varphi$ is not a polynomial**, which covers ReLU. These are existence results: they give no width, no algorithm and no generalisation guarantee [[27]](https://en.wikipedia.org/wiki/Universal_approximation_theorem). "Non-polynomial" explains the collapse theorem from another angle, since the identity is the degree-1 polynomial. CS231n notes that the theorem "says little" about practical success and that depth helps empirically [[7]](https://cs231n.github.io/neural-networks-1/). Do not claim the required width is "exponential".

**Initialisation (item 12).** [SIMPLIFICATION: i.i.d. zero-mean weights, inputs independent of the weights, linear regime.] Since $\mathrm{Var}(z_i)=n_{l-1}\mathrm{Var}(w)\,E[a_j^2]$, keeping the signal's variance constant needs $n_{l-1}\mathrm{Var}(w)=1$ forward and $n_l\mathrm{Var}(w)=1$ backward. Glorot & Bengio's compromise is $\mathrm{Var}(w)=2/(n_{\mathrm{in}}+n_{\mathrm{out}})$, realised as $U[-\sqrt6/\sqrt{n_j+n_{j+1}},\ +\sqrt6/\sqrt{n_j+n_{j+1}}]$ [[25]](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf). For ReLU, $E[a^2]=\tfrac12\mathrm{Var}(z)$, so He et al. use $\mathrm{Var}(w)=2/n_{\mathrm{in}}$ [[28]](https://ar5iv.labs.arxiv.org/html/1502.01852). This is §9's "$|\lambda|\approx1$" achieved statistically. The weights must also be *random*. Goodfellow et al. explain that identical units with identical initial parameters are updated identically [[5]](https://www.deeplearningbook.org/contents/optimization.html). An induction proof in our notation runs as follows. Equal rows of $\mathbf{W}^{(1)}$ and equal entries of $\mathbf{W}^{(2)}$ give a $\boldsymbol{\delta}^{(1)}$ with equal entries, so $\boldsymbol{\delta}^{(1)}\mathbf{x}^\mathsf{T}$ has equal rows and the symmetry persists forever, with $\mathrm{rank}(\mathbf{W}^{(1)})\le1$. A run of 100 steps of a 2 → 3 → 1 tanh network from all-0.5 weights kept the three rows exactly equal.

**Beyond dense layers (item 13).** Convolution "can be viewed as multiplication by a matrix" whose rows are shifted copies of the kernel, a sparse **Toeplitz** matrix (constant along each diagonal) [[6]](https://www.deeplearningbook.org/contents/convnets.html). A 3-tap kernel on $\mathbf{x}\in\mathbb{R}^5$ is a $3\times5$ matrix with 3 parameters instead of 15. Attention is $\mathrm{softmax}(\mathbf{QK}^\mathsf{T}/\sqrt{d_k})\mathbf{V}$, where entry $(i,j)$ of $\mathbf{QK}^\mathsf{T}$ is a dot product $\mathbf{q}_i^\mathsf{T}\mathbf{k}_j$. The $\sqrt{d_k}$ comes from the same variance argument as initialisation [[29]](https://ar5iv.labs.arxiv.org/html/1706.03762). **Convention warning:** transformer papers put tokens in *rows*. Convert or state this on the slide.

---

## Part IV: worked running example (2 → 2 (ReLU) → 1 on XOR)

All numbers were produced by `code/worked_example.py` (output in `code/worked_example_output.txt`) and are reproduced unchanged. **Conventions:** the output is linear ($\hat y=z^{(2)}$, so $\varphi'=1$ there). The batch loss is a **sum with no $1/m$**, $\ell=\tfrac12\sum_{j=1}^4(\hat y_j-y_j)^2$. $\mathrm{ReLU}'(0):=0$.

$$
\underset{2\times4}{\mathbf{X}}=\begin{bmatrix}0&0&1&1\\0&1&0&1\end{bmatrix},\qquad
\underset{1\times4}{\mathbf{y}^\mathsf{T}}=\begin{bmatrix}0&1&1&0\end{bmatrix},\qquad
\underset{4\times1}{\mathbf{1}}=\begin{bmatrix}1&1&1&1\end{bmatrix}^\mathsf{T}.
$$

**A1. Exact solution, forward (Goodfellow et al. §6.1).**

$$
\underset{2\times2}{\mathbf{W}^{(1)}}=\begin{bmatrix}1&1\\1&1\end{bmatrix},\quad
\underset{2\times1}{\mathbf{b}^{(1)}}=\begin{bmatrix}0\\-1\end{bmatrix},\quad
\underset{1\times2}{\mathbf{W}^{(2)}}=\begin{bmatrix}1&-2\end{bmatrix},\quad
\underset{1\times1}{b^{(2)}}=0 .
$$

| $\mathbf{x}$ | $\mathbf{z}^{(1)}$ | $\mathbf{a}^{(1)}$ | $\hat y$ | $y$ |
|---|---|---|---|---|
| $[0\ 0]^\mathsf{T}$ | $[0\ \ {-1}]^\mathsf{T}$ | $[0\ 0]^\mathsf{T}$ | 0 | 0 |
| $[0\ 1]^\mathsf{T}$ | $[1\ 0]^\mathsf{T}$ | $[1\ 0]^\mathsf{T}$ | 1 | 1 |
| $[1\ 0]^\mathsf{T}$ | $[1\ 0]^\mathsf{T}$ | $[1\ 0]^\mathsf{T}$ | 1 | 1 |
| $[1\ 1]^\mathsf{T}$ | $[2\ 1]^\mathsf{T}$ | $[2\ 1]^\mathsf{T}$ | 0 | 0 |

$$
\underset{2\times4}{\mathbf{b}^{(1)}\mathbf{1}^\mathsf{T}}=\begin{bmatrix}0&0&0&0\\-1&-1&-1&-1\end{bmatrix},\quad
\underset{2\times4}{\mathbf{Z}^{(1)}}=\underset{2\times2}{\mathbf{W}^{(1)}}\underset{2\times4}{\mathbf{X}}+\mathbf{b}^{(1)}\mathbf{1}^\mathsf{T}=\begin{bmatrix}0&1&1&2\\-1&0&0&1\end{bmatrix},
$$

$$
\underset{2\times4}{\mathbf{A}^{(1)}}=\mathrm{ReLU}(\mathbf{Z}^{(1)})=\begin{bmatrix}0&1&1&2\\0&0&0&1\end{bmatrix},\quad
\underset{1\times4}{\hat{\mathbf{y}}^\mathsf{T}}=\underset{1\times2}{\mathbf{W}^{(2)}}\underset{2\times4}{\mathbf{A}^{(1)}}+b^{(2)}\mathbf{1}^\mathsf{T}=\begin{bmatrix}0&1&1&0\end{bmatrix},\quad \ell=0 .
$$

**A2. One gradient-descent step from a perturbed start.** Only $\mathbf{W}^{(2)}$ changes, to $[1\ \ -1]$, and $\eta=0.1$. $\mathbf{Z}^{(1)}$ and $\mathbf{A}^{(1)}$ are as in A1.

$$
\hat{\mathbf{y}}^\mathsf{T}=\begin{bmatrix}1&-1\end{bmatrix}\begin{bmatrix}0&1&1&2\\0&0&0&1\end{bmatrix}=\begin{bmatrix}0&1&1&1\end{bmatrix},\quad
\hat{\mathbf{y}}^\mathsf{T}-\mathbf{y}^\mathsf{T}=\begin{bmatrix}0&0&0&1\end{bmatrix},\quad \ell=0.5 .
$$

Backward. Linear output plus squared loss gives $\boldsymbol{\delta}^{(2)}=\hat y-y$ per sample:

$$
\underset{1\times4}{\boldsymbol{\Delta}^{(2)}}=\begin{bmatrix}0&0&0&1\end{bmatrix},\quad
\underset{1\times2}{\frac{\partial\ell}{\partial\mathbf{W}^{(2)}}}=\underset{1\times4}{\boldsymbol{\Delta}^{(2)}}\underset{4\times2}{(\mathbf{A}^{(1)})^\mathsf{T}}=\begin{bmatrix}2&1\end{bmatrix},\quad
\frac{\partial\ell}{\partial b^{(2)}}=\boldsymbol{\Delta}^{(2)}\mathbf{1}=1 ,
$$

$$
\underset{2\times4}{(\mathbf{W}^{(2)})^\mathsf{T}\boldsymbol{\Delta}^{(2)}}=\begin{bmatrix}1\\-1\end{bmatrix}\begin{bmatrix}0&0&0&1\end{bmatrix}=\begin{bmatrix}0&0&0&1\\0&0&0&-1\end{bmatrix},\quad
\underset{2\times4}{\mathrm{ReLU}'(\mathbf{Z}^{(1)})}=\begin{bmatrix}0&1&1&1\\0&0&0&1\end{bmatrix},\quad
\underset{2\times4}{\boldsymbol{\Delta}^{(1)}}=\begin{bmatrix}0&0&0&1\\0&0&0&-1\end{bmatrix},
$$

$$
\underset{2\times2}{\frac{\partial\ell}{\partial\mathbf{W}^{(1)}}}=\underset{2\times4}{\boldsymbol{\Delta}^{(1)}}\underset{4\times2}{\mathbf{X}^\mathsf{T}}=\begin{bmatrix}1&1\\-1&-1\end{bmatrix},\qquad
\underset{2\times1}{\frac{\partial\ell}{\partial\mathbf{b}^{(1)}}}=\boldsymbol{\Delta}^{(1)}\mathbf{1}=\begin{bmatrix}1\\-1\end{bmatrix}.
$$

As a sum of outer products, $\boldsymbol{\delta}^{(1)}_j\mathbf{x}_j^\mathsf{T}=\mathbf{0}$ for $j=1,2,3$, and $\boldsymbol{\delta}^{(1)}_4\mathbf{x}_4^\mathsf{T}=\begin{bmatrix}1\\-1\end{bmatrix}\begin{bmatrix}1&1\end{bmatrix}=\begin{bmatrix}1&1\\-1&-1\end{bmatrix}$. The sum equals $\boldsymbol{\Delta}^{(1)}\mathbf{X}^\mathsf{T}$ (script: `True`) and has rank 1, because only one sample contributes. The $\mathrm{ReLU}'(0)$ convention changes nothing here, since $\boldsymbol{\Delta}^{(2)}$ is 0 on samples 1–3. Finite differences ($h=10^{-6}$) agreed to within 2.50e−7 ($\mathbf{W}^{(1)}$), 5.00e−7 ($\mathbf{b}^{(1)}$), 8.1e−11 ($\mathbf{W}^{(2)}$) and 1.0e−12 ($b^{(2)}$). At a random point away from the ReLU kink the agreement was 1.24e−10.

Update with $\eta=0.1$:

$$
\mathbf{W}^{(1)}\leftarrow\begin{bmatrix}0.9&0.9\\1.1&1.1\end{bmatrix},\quad
\mathbf{b}^{(1)}\leftarrow\begin{bmatrix}-0.1\\-0.9\end{bmatrix},\quad
\mathbf{W}^{(2)}\leftarrow\begin{bmatrix}0.8&-1.1\end{bmatrix},\quad
b^{(2)}\leftarrow-0.1 .
$$

Forward again:

$$
\mathbf{Z}^{(1)}=\begin{bmatrix}-0.1&0.8&0.8&1.7\\-0.9&0.2&0.2&1.3\end{bmatrix},\quad
\mathbf{A}^{(1)}=\begin{bmatrix}0&0.8&0.8&1.7\\0&0.2&0.2&1.3\end{bmatrix},\quad
\hat{\mathbf{y}}^\mathsf{T}=\begin{bmatrix}-0.1&0.32&0.32&-0.17\end{bmatrix},
$$

and **$\ell$: 0.500000 → 0.481850**. By hand, the residuals are $-0.1, -0.68, -0.68, -0.17$, and $\tfrac12(0.01+0.4624+0.4624+0.0289)=0.48185$. Note for the slides: sample 4 improved a lot ($1\to-0.17$), but samples 2 and 3, which were perfect, got worse ($1\to0.32$), because they share the changed weights. **One gradient step lowers the total loss, not every per-sample loss.**

**A3. Single-sample slide version: use $\mathbf{x}=[1\ 1]^\mathsf{T}$, $y=0$, not $[1\ 0]^\mathsf{T}$.** The sample $[1\ 0]^\mathsf{T}$ gives $\hat y=1=y$ at this start, so its loss and all its gradients are zero. Same start, $\eta=0.1$:

| quantity | value | shape |
|---|---|---|
| $\mathbf{z}^{(1)}=\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)}$ | $[2\ 1]^\mathsf{T}$ | 2×1 |
| $\mathbf{a}^{(1)}=\mathrm{ReLU}(\mathbf{z}^{(1)})$ | $[2\ 1]^\mathsf{T}$ | 2×1 |
| $\hat y=\mathbf{W}^{(2)}\mathbf{a}^{(1)}+b^{(2)}$ | 1 | 1×1 |
| $\ell$ | 0.5 | scalar |
| $\delta^{(2)}=\hat y-y$ | 1 | 1×1 |
| $\partial\ell/\partial\mathbf{W}^{(2)}=\delta^{(2)}(\mathbf{a}^{(1)})^\mathsf{T}$ | $[2\ 1]$ | 1×2 |
| $(\mathbf{W}^{(2)})^\mathsf{T}\delta^{(2)}$ | $[1\ \ {-1}]^\mathsf{T}$ | 2×1 |
| $\mathrm{ReLU}'(\mathbf{z}^{(1)})$ | $[1\ 1]^\mathsf{T}$ | 2×1 |
| $\boldsymbol{\delta}^{(1)}$ | $[1\ \ {-1}]^\mathsf{T}$ | 2×1 |
| $\partial\ell/\partial\mathbf{W}^{(1)}=\boldsymbol{\delta}^{(1)}\mathbf{x}^\mathsf{T}$ | $\begin{bmatrix}1&1\\-1&-1\end{bmatrix}$ | 2×2 (rank 1) |
| $\partial\ell/\partial\mathbf{b}^{(1)}=\boldsymbol{\delta}^{(1)}$, $\partial\ell/\partial b^{(2)}=\delta^{(2)}$ | $[1\ \ {-1}]^\mathsf{T}$, 1 | 2×1, 1×1 |

These gradients are **identical** to the A2 batch gradients, because only sample 4 had a non-zero error, so the updated parameters are the same as in A2. After the step, this sample has $\hat y=-0.17$ and $\ell=0.01445$ (from 0.5).

**A4. Collapse demo and best linear fit.** With $\varphi=$ identity and the A1 weights, $\underset{1\times2}{\mathbf{W}^{(2)}\mathbf{W}^{(1)}}=[-1\ \ -1]$ and $\mathbf{W}^{(2)}\mathbf{b}^{(1)}+b^{(2)}=2$. The two-layer output and the single affine map both give $[2\ 1\ 1\ 0]$ (script: `True`), and $\mathrm{rank}(\mathbf{W}^{(1)})=\mathrm{rank}(\mathbf{W}^{(2)}\mathbf{W}^{(1)})=1$. For the least-squares affine fit, set $\tilde{\mathbf{X}}=\begin{bmatrix}\mathbf{X}\\\mathbf{1}^\mathsf{T}\end{bmatrix}$ (3×4). The normal equations $\underset{3\times3}{\begin{bmatrix}2&1&2\\1&2&2\\2&2&4\end{bmatrix}}\boldsymbol{\theta}=\underset{3\times1}{\begin{bmatrix}1\\1\\2\end{bmatrix}}$ give $\boldsymbol{\theta}=[0\ 0\ 0.5]^\mathsf{T}$, predictions $[0.5\ 0.5\ 0.5\ 0.5]$ and $\ell=0.5$. Any identity-activation network collapses to such a map and can do no better.

---

## Part V: activation experiment (real, observed results)

**Code:** `code/experiment.py` (78 lines, NumPy only, matrix form, samples as columns). In it, `@` is matrix multiplication and `*` is the Hadamard product, but `*` also appears as scalar multiplication (e.g. `ETA*g`). The figures are produced by `code/make_figures.py` and the raw numbers are in `code/experiment_results.json`. **Setup:** a $2\to8\to1$ network with $\mathbf{Z}^{(1)}=\mathbf{W}^{(1)}\mathbf{X}+\mathbf{b}^{(1)}\mathbf{1}^\mathsf{T}$, hidden $\varphi\in$ {identity, logistic, tanh, ReLU}, logistic output, and **binary cross-entropy** (the standard loss for 0/1 targets, $-[y\log\hat y+(1-y)\log(1-\hat y)]$) averaged with $1/m$. With this loss the logistic derivative cancels, so $\boldsymbol{\Delta}^{(2)}=\tfrac1m(\hat{\mathbf{y}}^\mathsf{T}-\mathbf{y}^\mathsf{T})$. The weights start as $w_{ij}\sim N(0,1/n_{\mathrm{in}})$ with zero biases, and every activation gets the same initial weights for a given seed. Training is full-batch gradient descent with $\eta=0.5$ for 5000 **epochs** (one epoch = one step on the whole dataset), over seeds 0–4. The datasets are XOR ($m=4$) and two moons ($m=200$, noise 0.15). Accuracy is measured on the **training** data only. Backprop matched finite differences to 1.1e−10 or better for all four activations.

| dataset | hidden $\varphi$ | final loss $\ell$ (mean ± std [min, max]) | final train accuracy | epochs until $\ell<0.1$, seeds 0–4 |
|---|---|---|---|---|
| XOR | identity | 0.6931 ± 0.0000 [0.6931, 0.6931] | 0.500 [0.25, 0.75] **(meaningless, see below)** | never |
| XOR | logistic | 0.0026 ± 0.0004 [0.0021, 0.0033] | 1.000 | 828, 1416, 804, 655, 651 |
| XOR | tanh | 0.0005 ± 0.0001 [0.0004, 0.0006] | 1.000 | 167, 160, 129, 106, 105 |
| XOR | ReLU | 0.0004 ± 0.0002 [0.0002, 0.0006] | 1.000 | 133, 95, 123, 66, 101 |
| moons | identity | 0.2714 ± 0.0000 [0.2714, 0.2714] | 0.880 | never |
| moons | logistic | 0.2234 ± 0.0949 [0.0337, 0.2729] | 0.903 [0.880, 0.995] | only seed 2: 2808 |
| moons | tanh | 0.0052 ± 0.0006 [0.0048, 0.0065] | 1.000 | 407, 1308, 599, 751, 549 |
| moons | ReLU | 0.0972 ± 0.1091 [0.0061, 0.2422] | 0.958 [0.890, 1.000] | 510, never, 1545, 376, never |

![Activation functions and their derivatives](<../figures/fig1_activations.png>)

*Figure 1. $\varphi$ (solid) and $\varphi'$ (dashed). logistic′ peaks at 0.25, tanh′ at 1, and ReLU′ is a 0/1 step.*

![Decision boundaries, seed 0](<../figures/fig2_decision_boundaries.png>)

*Figure 2. Decision boundaries ($\hat y=0.5$), seed 0. XOR/identity has no boundary because $\hat y=0.5$ everywhere.*

![Loss curves, mean over 5 seeds with min–max band](<../figures/fig3_loss_curves.png>)

*Figure 3. Training loss (log scale), mean over 5 seeds with a min–max band.*

**What the data show, and only that.** First, **identity cannot solve XOR**. On every seed the network output 0.5 at all four points, the loss sat at $\ln2\approx0.6931$, and the collapsed weight $\mathbf{W}^{(2)}\mathbf{W}^{(1)}$ was numerically zero (entries around $10^{-16}$). This is exactly the A4 least-squares answer. **The 0.25–0.75 "accuracy" is an artefact.** The outputs are 0.5 up to rounding noise, so the $>0.5$ threshold breaks ties at random. The correct reading is "no decision boundary", as Figure 2 shows. Second, on two moons identity reproduced logistic regression on all seeds (a straight line, accuracy 0.88). The moons are roughly linearly separable, and the non-linear models are needed only for the last 12%. Third, all three non-linear activations solved XOR on every seed. ReLU (66–133 epochs) and tanh (105–167) were similarly fast, and logistic was clearly slowest (651–1416), with an initial plateau visible in Figure 3. This is consistent with logistic′ ≤ 1/4, but the experiment does not isolate that cause. Fourth, on two moons with this budget, tanh was the only activation that reached 100% training accuracy on all 5 seeds. **ReLU stalled on 2 of 5 seeds** (seeds 1 and 4, $\ell\approx0.22$–0.24, accuracy 0.89–0.90). Dead units do **not** explain this: seed 1 had 1 of 8 units dead, seed 4 had 0, and the successful seed 0 also had 1. **The cause is unknown.** Logistic stayed near the linear solution on 4 of 5 seeds, and only seed 2 escaped (accuracy 0.995). **Caveats:** one $\eta$, one width, one budget, one initialisation scheme and 5 seeds. Logistic might benefit from a larger $\eta$, which was not tested. Present these as what *this* experiment showed, not as general rankings of the activations.

## Part VI: batching benchmark (CPU sandbox, not GPU)

`code/batching_benchmark.py` runs 784 → 128 (ReLU) → 10 with $m=1024$ columns, comparing (a) a Python loop of $m$ matrix–vector products per layer with (b) one matrix–matrix product per layer, $\mathbf{W}^{(1)}\mathbf{X}+\mathbf{b}^{(1)}\mathbf{1}^\mathsf{T}$. The outputs were identical (`allclose`). Both versions do exactly $2n_l n_{l-1}m$ FLOPs per layer, $205{,}520{,}896+2{,}621{,}440=208{,}142{,}336$ in total. The hardware was 2 cores (Intel Xeon at 2.10 GHz) with NumPy 2.5.3 and OpenBLAS 0.3.34. Each time is the best of 7 runs, with the median in brackets.

| run | loop of mat–vec | one mat–mat | speed-up best/best (median/median) |
|---|---|---|---|
| 1 (default threads) | 21.18 ms (27.15), 9.8 GFLOP/s | 3.52 ms (3.58), 59.1 GFLOP/s | 6.0-fold (7.6-fold) |
| 2 (default threads) | 18.03 ms (23.17), 11.5 GFLOP/s | 3.11 ms (3.16), 66.8 GFLOP/s | 5.8-fold (7.3-fold) |
| 3 (1 thread) | 19.85 ms (23.87), 10.5 GFLOP/s | 3.19 ms (3.25), 65.2 GFLOP/s | 6.2-fold (7.3-fold) |

**Batching was about 6 (best time) to 7.5 (median) times faster for identical arithmetic.** The single-thread run shows that the gain is not parallelism. It comes from one GEMM call reusing each weight from cache across many columns, plus avoiding Python loop overhead. GPU speed-ups were **not measured**, so label any slide that shows this number "CPU sandbox".

---

## Recommended slide storyline (refines handoff §7)

The research suggests five changes to the draft table. (i) Use the Goodfellow weights for the forward slide ($\ell=0$, all integers) and the perturbed $\mathbf{W}^{(2)}=[1\ \ -1]$ for backprop. **The single-sample backprop slide must use $\mathbf{x}=[1\ 1]^\mathsf{T}$, $y=0$**, because $[1\ 0]^\mathsf{T}$ has zero loss and zero gradients. (ii) The team chose both P2 items, so split draft slide 11 into a least-squares/$\eta$ slide and a vanishing-gradient slide labelled "intuition", and merge the references into the takeaways slide. That gives 14 slides, of which 12 carry content. (iii) The collapse slide should show the concrete $[-1\ \ -1]$, $2$ and $[2\ 1\ 1\ 0]$ numbers, and the activation slide should put the rank-1 $\mathbf{W}^{(1)}$ and "tent in $s$" picture next to Figure 1. (iv) The batching slide should cite the measured CPU number, not a GPU claim, and state that FLOPs are unchanged. (v) The GD slide should show $\ell$ falling from 0.5 to 0.48185 *and* samples 2 and 3 getting worse. It must also say "sum loss, no $1/m$". Times total about 9:55.

| # | Slide (idea as a sentence) | Running-example content / LA bridge | Time | Presenter |
|---|---|---|---|---|
| 1 | "A neural network is a chain of matrix multiplications with a bend in between" | thesis only | 0:20 | Zyik |
| 2 | "A neuron is a dot product that draws a hyperplane" | $\mathbf{w}$ normal, $b$ shifts; Wk 2 dot product, Wk 9 $\mathcal{N}(\mathbf{w}^\mathsf{T})$ | 0:45 | Zyik |
| 3 | "Stack neurons as rows: a layer is $\mathbf{Wx}+\mathbf{b}$" | row vs column picture, shapes; Wk 2 | 0:45 | Zyik |
| 4 | "Four XOR inputs become one matrix product" | A1 batch: $\mathbf{Z}^{(1)}$, $\mathbf{A}^{(1)}$, $\hat{\mathbf{y}}^\mathsf{T}=[0\ 1\ 1\ 0]$; $\mathbf{b1}^\mathsf{T}$ as an outer product | 0:45 | Voon |
| 5 | "Without $\varphi$, depth collapses to one affine map" | A4: $[-1\ \ -1]$, 2, $[2\ 1\ 1\ 0]$; rank bound; homogeneous block matrix; Wk 1, Wk 5 | 0:50 | Voon |
| 6 | "A bend makes XOR separable" | Fig. 1, inequality proof, rank-1 $\mathbf{W}^{(1)}$ and the tent in $s$ | 0:55 | Voon |
| 7 | "Batching turns $m$ GEMVs into one GEMM" | FLOPs unchanged; 6–7.5-fold CPU speed-up; arithmetic intensity | 0:40 | Gong Zu |
| 8 | "Learning is walking downhill: $\mathbf{W}\leftarrow\mathbf{W}-\eta\,\partial\ell/\partial\mathbf{W}$" | loss defined; steepest descent via Cauchy–Schwarz; cost claim ($c<6$) | 0:40 | Gong Zu |
| 9 | "Backprop sends errors back through $\mathbf{W}^\mathsf{T}$ and $\odot\varphi'$" | A3 with $\mathbf{x}=[1\ 1]^\mathsf{T}$: $\delta^{(2)}=1$, $\boldsymbol{\delta}^{(1)}=[1\ \ -1]^\mathsf{T}$; mirror diagram; Wk 3 | 0:55 | Zhe An |
| 10 | "Each weight gradient is a rank-1 outer product" | $\boldsymbol{\delta}^{(1)}\mathbf{x}^\mathsf{T}$; the A2 update, with $\ell$ going 0.5 → 0.48185; Wk 2, Wk 5 | 0:45 | Zhe An |
| 11 | "A one-layer linear net is least squares; $\eta<2/\lambda_{\max}$" | normal equations; $\mathbf{e}_{k+1}=[\mathbf{I}-\eta\mathbf{A}^\mathsf{T}\mathbf{A}]\mathbf{e}_k$; rename to $\mathbf{X}^\mathsf{T}\mathbf{w}\approx\mathbf{y}$; Wk 10, 11 | 0:50 | Haide |
| 12 | "Matrix powers explain vanishing gradients (intuition)" | $(\mathbf{W}^\mathsf{T})^k$, logistic′ ≤ 1/4; "SIMPLIFICATION" badge; Wk 11 | 0:40 | Haide |
| 13 | "Our experiment: identity fails, bends succeed" | Figs. 2 and 3, the table, caveats | 0:45 | Gong Zu |
| 14 | "Three LA operations power deep learning" + references | matrix product, outer product, Hadamard product | 0:20 | Zyik |

Speaking time comes to Zyik 2:10, Voon 2:30, Gong Zu 2:05, Zhe An 1:40 and Haide 1:30. The two hardest slide pairs (backprop, eigenvalues) are deliberately shorter. Each presenter's Q&A ownership below matches their slides.

---

## Q&A bank (handoff §9): 27 questions, owned by theme

**Zyik: neurons, layers, framing (slides 1–3, 14)**

| # | Question | Answer |
|---|---|---|
| 1 | *(Slide 1 gotcha)* You say "chain of matrix multiplications", but layers also *add* $\mathbf{b}$ and apply $\varphi$. Is the thesis literally true? | The bias becomes a multiplication in homogeneous coordinates: $[\mathbf{W}\ \ \mathbf{b}]$ times $[\mathbf{x};1]$. $\varphi$ is not a matrix product in general, and that is the "bend". For ReLU with a fixed active pattern, it acts like multiplying by a diagonal 0/1 matrix, which is why ReLU nets are piecewise affine. |
| 2 | Is $\mathbf{Wx}+\mathbf{b}$ linear? | No, it is **affine**: it sends $\mathbf{0}$ to $\mathbf{b}$, so $f(\mathbf{0})\neq\mathbf{0}$. Append a 1 and it becomes linear: $\begin{bmatrix}\mathbf{W}&\mathbf{b}\\\mathbf{0}^\mathsf{T}&1\end{bmatrix}\begin{bmatrix}\mathbf{x}\\1\end{bmatrix}$. Composition is then block multiplication. |
| 3 | *(Slide 2 gotcha)* $\mathbf{w}$ is normal to the hyperplane. What exactly does $b$ do? | It translates the hyperplane without rotating it. The closest point to the origin is $-b\,\mathbf{w}/\lVert\mathbf{w}\rVert^2$, at distance $|b|/\lVert\mathbf{w}\rVert$. Without $b$, every boundary passes through the origin. |
| 4 | *(Slide 3 gotcha)* In the column picture, where can $\mathbf{W}\mathbf{a}$ land? | In $\mathcal{C}(\mathbf{W})$, of dimension $\mathrm{rank}(\mathbf{W})\le\min(n_l,n_{l-1})$. Adding $\mathbf{b}$ gives a translate of that subspace. A wider layer does not raise the dimension past $n_{l-1}$ until $\varphi$ bends it. |
| 5 | *(Slide 14 gotcha)* Name the three LA operations and one place each appears. | Matrix product: forward $\mathbf{WA}$, and $\mathbf{W}^\mathsf{T}\boldsymbol{\delta}$ backward. Outer product: $\partial\ell/\partial\mathbf{W}=\boldsymbol{\delta}\mathbf{a}^\mathsf{T}$ and the bias broadcast $\mathbf{b1}^\mathsf{T}$. Hadamard product: $\odot\varphi'(\mathbf{z})$ in backprop. |
| 6 | If one hidden layer is a universal approximator, why go deep? | The theorem only guarantees existence, with possibly huge width, no training method and no generalisation guarantee. Depth helps empirically [[7]](https://cs231n.github.io/neural-networks-1/). Do not say "exponential width" without a primary source. |

**Voon: collapse and activations (slides 4–6)**

| # | Question | Answer |
|---|---|---|
| 7 | Why do we need a non-linear activation? | The collapse: $\mathbf{W}^{(2)}[\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)}]+\mathbf{b}^{(2)}=[\mathbf{W}^{(2)}\mathbf{W}^{(1)}]\mathbf{x}+[\mathbf{W}^{(2)}\mathbf{b}^{(1)}+\mathbf{b}^{(2)}]$, so any depth equals one affine layer with a hyperplane boundary, which cannot fit XOR. Our identity runs predicted 0.5 everywhere. |
| 8 | *(Slide 4 gotcha)* $\mathbf{W}^{(1)}$ has two identical rows. Isn't one neuron wasted? | The rows are equal (rank 1), but the biases differ (0 and −1). After ReLU, $h_1=s$ and $h_2=\max\{0,s-1\}$ do different jobs. The second fires only at $s=2$, which produces the "tent". |
| 9 | *(Slide 5 gotcha)* In a linear 2 → 1 → 2 network, what can it represent? | Only affine maps whose matrix has rank ≤ 1, since $\mathrm{rank}(\mathbf{W}^{(2)}\mathbf{W}^{(1)})\le1$. $\mathcal{N}(\tilde{\mathbf{W}})$ is at least 1-dimensional, so it is blind to one input direction. Depth added nothing and the narrow layer removed something. |
| 10 | Why ReLU instead of sigmoid (logistic)? | logistic′ ≤ 1/4, so each layer shrinks the backward signal by at least 4 (up to about $10^{-6}$ over 10 layers), and it saturates on both sides. ReLU′ = 1 when active, it is cheap, and it gives sparse activations. Downside: dead units [[3]](https://www.deeplearningbook.org/contents/mlp.html), [[7]](https://cs231n.github.io/neural-networks-1/). In our XOR run, logistic took 651–1416 epochs against 66–133 for ReLU. |
| 11 | *(Slide 6 gotcha)* ReLU is piecewise linear, so isn't a ReLU network still linear? | It is affine on each region with a fixed active pattern, but it switches pieces across regions. Additivity fails: $\mathrm{ReLU}(1)+\mathrm{ReLU}(-1)=1\neq\mathrm{ReLU}(0)=0$. That switching is what makes XOR separable. |

**Zhe An: backpropagation (slides 9–10)**

| # | Question | Answer |
|---|---|---|
| 12 | Why $\mathbf{W}^\mathsf{T}$ in backprop? What shape is it, and why? | The Jacobian of $\mathbf{z}=\mathbf{Wa}+\mathbf{b}$ with respect to $\mathbf{a}$ is $\mathbf{W}$, and the chain rule for a scalar loss uses $\mathbf{J}^\mathsf{T}$. $(\mathbf{W}^{(l+1)})^\mathsf{T}$ is $n_l\times n_{l+1}$, which maps an $n_{l+1}$-error back to $n_l$, the mirror of the forward map. |
| 13 | Why is $\partial\ell/\partial\mathbf{W}$ an outer product, and what is its rank? | $\partial\ell/\partial w_{ij}=\delta_i a_j$, and the matrix of $\delta_i a_j$ is $\boldsymbol{\delta}\mathbf{a}^\mathsf{T}$. Every column is a multiple of $\boldsymbol{\delta}$, so the rank is ≤ 1 (exactly 1 if both vectors are non-zero). Example: $[1\ \ -1]^\mathsf{T}[1\ 1]$. |
| 14 | *(Slide 9 gotcha)* Why a Hadamard product $\odot$ and not a matrix product with $\varphi'$? | The Jacobian of an element-wise $\varphi$ is $\mathrm{diag}(\varphi'(\mathbf{z}))$, and $\mathrm{diag}(\mathbf{v})\mathbf{u}=\mathbf{v}\odot\mathbf{u}$. So it *is* a matrix product, computed cheaply without forming the $n\times n$ diagonal. |
| 15 | *(Slide 10 gotcha)* Is the batch gradient $\boldsymbol{\Delta}\mathbf{A}^\mathsf{T}$ still rank 1? | Not in general. It is a sum of $m$ rank-1 matrices, with rank ≤ $\min(m,n_l,n_{l-1})$. In our example it is rank 1 only because samples 1–3 had zero error. |
| 16 | What if all weights start equal? | Symmetry never breaks. Equal rows give equal $\boldsymbol{\delta}^{(1)}$ entries, so $\boldsymbol{\delta}^{(1)}\mathbf{x}^\mathsf{T}$ has equal rows and every update preserves equality (by induction). The layer acts like one neuron, with $\mathrm{rank}\,\mathbf{W}^{(1)}\le1$. Our 100-step test kept the rows exactly equal. The fix is random initialisation [[5]](https://www.deeplearningbook.org/contents/optimization.html). |

**Gong Zu: gradient descent, cost, batching, experiment (slides 7, 8, 13)**

| # | Question | Answer |
|---|---|---|
| 17 | Why do GPUs matter? | Batching makes each layer one GEMM. GEMV has arithmetic intensity below 1 FLOP/byte and is memory-bound, while GEMM reuses each loaded weight $m$ times. GPUs need high intensity (about 139 FLOP/B on a V100) to keep their units busy [[19]](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html). |
| 18 | *(Slide 7 gotcha)* Does batching reduce the number of FLOPs? | No. Both versions do the same $2n_l n_{l-1}m$ FLOPs (208,142,336 in our benchmark). The 6–7.5-fold CPU speed-up comes from memory reuse and fewer calls. GPU speed-up was not measured. |
| 19 | *(Slide 8 gotcha)* After your GD step, samples 2 and 3 got worse. Did gradient descent fail? | No. It lowers the *total* loss to first order (0.5 → 0.48185). The samples share weights, so improving sample 4 ($1\to-0.17$) moved samples 2 and 3 ($1\to0.32$). |
| 20 | How expensive is backprop compared with perturbing each weight? | For a scalar loss, backprop costs a small constant multiple of one forward pass ($c<6$, typically 2–3) [[16]](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf). Finite differences need about one forward pass per parameter: 101,771 against about 2 for our 784 → 128 → 10 example. The catch is memory, since forward values must be stored. |
| 21 | *(Slide 13 gotcha)* Identity on XOR shows 0.50 accuracy, ranging 0.25–0.75. Does it partly work? Why did ReLU fail on some moons seeds? | The accuracy is meaningless: the outputs are 0.5 everywhere, so the threshold breaks ties by rounding noise, and there is no boundary. ReLU stalled on 2 of 5 moons seeds. Dead units do not explain it (the stalled seeds had 1 and 0 dead units, and a successful seed also had 1), so **the cause is unknown**. This was one setting only. |

**Haide: syllabus connections (slides 11–12)**

| # | Question | Answer |
|---|---|---|
| 22 | How is this different from Week 10 least squares? Why not solve the normal equations? | A one-layer linear net with squared loss *is* least squares: $\nabla\ell=\mathbf{A}^\mathsf{T}(\mathbf{Ax}-\mathbf{b})=\mathbf{0}$ gives the normal equations. With $\varphi$ or depth there is no closed form. Also, $\mathbf{A}^\mathsf{T}\mathbf{A}$ is $n\times n$, which is infeasible for millions of parameters, and it squares the conditioning (flagged as unverified). |
| 23 | Where do eigenvalues show up in training? | (a) Gradient-descent stability on quadratics, $0<\eta<2/\lambda_{\max}(\mathbf{A}^\mathsf{T}\mathbf{A})$, with speed set by $\kappa$. (b) Powers of $\mathbf{W}^\mathsf{T}$ in backprop (intuition for vanishing/exploding gradients). (c) A linear autoencoder learns the top eigenvectors of the covariance (PCA) [[26]](https://igb.uci.edu/~pfbaldi/publications/journals/1989/sdarticle.pdf). |
| 24 | *(Slide 11 gotcha)* Why $2/\lambda_{\max}$ and not $1/\lambda_{\max}$? | Each eigen-component is multiplied by $(1-\eta\lambda_i)$ per step. Convergence needs $|1-\eta\lambda_i|<1$, i.e. $0<\eta\lambda_i<2$. $1/\lambda_{\max}$ is Goodfellow's worst-case *optimal* step along the top eigenvector, not the stability limit. We confirmed sharpness: $0.9(2/\lambda_{\max})$ converged and $1.05(2/\lambda_{\max})$ diverged. |
| 25 | *(Slide 12 gotcha)* Your eigenvalue argument assumes the same $\mathbf{W}$ in every layer. Does it hold for real networks? | No, it is intuition. Real nets have a different $\mathbf{W}^{(l)}$ per layer, $\varphi'$ varies, and $\mathbf{W}$ may not be diagonalisable. The rigorous version uses singular values or norms and applies to RNNs, which reuse $\mathbf{W}$ [[24]](https://ar5iv.labs.arxiv.org/html/1211.5063). The exact part is logistic′ ≤ 1/4. |
| 26 | What does the null space of $\mathbf{W}^{(1)}$ mean for the network? | Inputs differing by a vector in $\mathcal{N}(\mathbf{W}^{(1)})$ give identical outputs, so the network cannot distinguish them. $\dim\mathcal{N}\ge n_0-n_1$. In XOR, $\mathcal{N}=\mathrm{span}\{[1\ \ -1]^\mathsf{T}\}$, which is harmless because XOR depends only on $x_1+x_2$. |
| 27 | What are the limits of the matrix view? | CNNs and attention are still matrix products, just structured. Convolution is a sparse Toeplitz matrix with shared weights [[6]](https://www.deeplearningbook.org/contents/convnets.html), and attention is $\mathrm{softmax}(\mathbf{QK}^\mathsf{T}/\sqrt{d_k})\mathbf{V}$ [[29]](https://ar5iv.labs.arxiv.org/html/1706.03762). The view covers *computation*. It says nothing about why trained networks generalise, and the non-linear $\varphi$ and softmax are not matrix products. |

That makes 27 questions: Zyik 6, Voon 5, Zhe An 5, Gong Zu 5, Haide 6. They include all 11 handoff §9 bullets (Q2, 7, 10, 12, 13, 16, 17, 22, 23, 26, 27) and one gotcha per slide of the refined 14-slide storyline, which covers every draft §7 slide (draft slide 11 is now slides 11–12, and the experiment and takeaways slides are now 13–14).

---

## Verify before presenting

The research notes flag the following items as unverified or simplified. Check each one before it goes on a slide or the references slide.

**Page numbers and quotations.** Goodfellow et al. page numbers came from a web-text summary, not the print edition: §6.1 pp. 167–172 (p. 168 linear XOR fit, pp. 170–171 Eq. 6.3), p. 191 (sigmoid/tanh), Ch. 4 pp. 86, 87 and 94, and p. 206 for Algorithm 6.4, which was not retrieved at all. The transpose in Goodfellow Eq. 6.46 was lost in the web extraction, so confirm it in print. Johnson's CS231n handout Eq. 12 appears to contain a typo ($\partial L/\partial\mathbf{y}\,\mathbf{x}^\mathsf{T}$ where the derivation gives $\partial L/\partial\mathbf{y}\,\mathbf{W}^\mathsf{T}$), so do not quote it verbatim. Nair & Hinton's ICML 2010 pages (commonly 807–814) were not verified.

**Second-hand or missing citations.** Krizhevsky et al.'s "factor of 6" comes only via CS231n. The Minsky & Papert decline story comes from Wikipedia, is contested, and should be phrased cautiously. The universal-approximation bibliographic details (Cybenko, Hornik et al., Leshno et al.) come via Wikipedia, so cross-check them, and do not claim exponential width. Bengio, Simard & Frasconi (1994), *IEEE Trans. Neural Networks* 5(2): the volume and issue are from memory. Hochreiter (1991) was not retrieved. Baldi & Hornik's journal volume and pages (*Neural Networks* 2(1): 53–58) need checking. The Vaswani et al. venue (NeurIPS 2017) is unconfirmed. The Level-3 BLAS paper (Dongarra et al. 1990) was not fetched. NVIDIA's fetched text gives no A100 ops:byte ratio, only the V100's 138.9. Nielsen Ch. 3 (the cross-entropy cancellation) was not fetched; the algebra is verified but has no source quote. The softmax plus cross-entropy $\boldsymbol{\delta}=\hat{\mathbf{y}}-\mathbf{y}$ result is uncited. Griewank & Walther's own "4–5" bound could not be verified, so use Baydin's "$c<6$, typically 2–3". The flop counts for forming and solving the normal equations ($mn^2$, $n^3/3$) and the claim that $\mathbf{A}^\mathsf{T}\mathbf{A}$ squares the condition number are standard facts but unsourced (Boyd & Vandenberghe or Strang not fetched). Saxe et al. (deep linear network dynamics) is unverified and was excluded from this report. The "random Gaussian matrix has largest singular value ≈ 2" link for initialisation needs random-matrix theory and a source, so keep it out. The hyperplane normal and distance facts come from a university lecture page rather than a textbook, and the one-line proof in §1 is the backup.

**Labelled simplifications.** §9 (eigenvalue intuition for vanishing and exploding gradients in feed-forward nets), §8's extension to deep nets (local quadratic approximation only), §12 (variance-preserving initialisation assumptions) and §10's "information bottleneck" reading for non-linear nets (interpretive) are all simplifications. The experiment's rankings hold for one configuration only, the ReLU stall cause is unknown, and the GPU speed-up was not measured.

---

## Accuracy checklist (handoff §8)

- [x] **Every product is conformant; shapes are written for every §2.3 equation.** See the shape-check table in §5 and the shapes under every matrix in Part IV.
- [ ] **(Partly satisfied.) No `*`, `·` or `×` as multiplication.** Satisfied in the report text itself. `×` appears only in dimensions, and speed-ups are written as "6-fold". **Not satisfied for code shown on slides:** `code/experiment.py` uses `*` for both the Hadamard product and scalar multiplication (`ETA*g`). If code appears on a slide, rewrite those lines or annotate that `@` is matmul and `*` is element-wise.
- [x] **No σ for an activation, no calligraphic L for the loss, layer indices in parentheses.** σ appears only as a reference to singular values, and the figures say "logistic (sigmoid)".
- [x] **Vectors and batches are columns** ($\mathbf{X}\in\mathbb{R}^{n_0\times m}$). Goodfellow's $\mathbf{XW}$ and $\mathbf{W}^\mathsf{T}\mathbf{x}$ were converted, and the rows-as-samples conventions (Johnson, Ch. 10 $\mathbf{A}$, transformer $\mathbf{Q}$) are flagged.
- [x] **"Linear" vs "affine" is used correctly.** $\mathbf{Wx}+\mathbf{b}$ is called affine throughout.
- [x] **The vanishing-gradient eigenvalue argument is labelled as a simplification** (§9 banner, slide 12 badge, Q25).
- [x] **The backprop cost claim is stated correctly**: a small constant multiple of one forward pass for a scalar loss, with $c<6$ (typically 2–3) and the memory caveat (§6).
- [x] **All running-example numbers are recomputed and consistent.** They are reproduced unchanged from the script output, and the A2 and A3 gradients and updates agree. One addition for the slides: state that the running example uses a summed loss (no $1/m$), while the experiment averages.
- [x] **Experiment results are real**, observed from `experiment.py`, with no placeholders. Caveats included: XOR/identity accuracy is meaningless, ReLU stalled on 2 of 5 moons seeds with cause unknown, and training-set accuracy only.

---

## Conclusion

The matrix view turns three ideas that look separate into one object seen from different sides. The forward pass is products by $\mathbf{W}$. The backward pass is products by $\mathbf{W}^\mathsf{T}$ with diagonal Jacobians folded in as $\odot$. The weight gradient is the outer product $\boldsymbol{\delta}\mathbf{a}^\mathsf{T}$ that connects the two passes. The non-linearity is the only part that is not a matrix product, and the collapse theorem and the XOR experiment both show that it is the part that matters. For a CS103 audience the strongest claim is that nothing on the slides needs a new kind of mathematics. Conformance, transposes, outer products, rank, null spaces, normal equations and matrix powers already cover the mechanics of learning. The one genuinely new tool is the chain rule written as Jacobian-transpose products.

Two findings should shape the deck's tone. First, the eigenvalue argument does double duty: it is exact for gradient-descent convergence on least squares and only heuristic for gradients through depth. Saying this distinction out loud will hold up better in Q&A than overclaiming. Second, the team's own data includes honest anomalies, ReLU stalling on two moons seeds and logistic stuck near the linear solution. Presenting these as observed, with "cause unknown", shows the kind of understanding the rubric rewards. The 6-to-7.5-fold CPU batching speed-up gives the "why GPUs" argument a measured number instead of a borrowed one.

---

## References

1. Nielsen, M. *Neural Networks and Deep Learning*, Ch. 1. http://neuralnetworksanddeeplearning.com/chap1.html
2. Nielsen, M. *Neural Networks and Deep Learning*, Ch. 2 (backpropagation). http://neuralnetworksanddeeplearning.com/chap2.html
3. Goodfellow, I., Bengio, Y. & Courville, A. *Deep Learning*, Ch. 6 (Deep Feedforward Networks). MIT Press, 2016. https://www.deeplearningbook.org/contents/mlp.html
4. Goodfellow et al. *Deep Learning*, Ch. 4 (Numerical Computation). https://www.deeplearningbook.org/contents/numerical.html
5. Goodfellow et al. *Deep Learning*, Ch. 8 (Optimization for Training Deep Models). https://www.deeplearningbook.org/contents/optimization.html
6. Goodfellow et al. *Deep Learning*, Ch. 9 (Convolutional Networks). https://www.deeplearningbook.org/contents/convnets.html
7. Stanford CS231n. Neural Networks Part 1: Setting up the Architecture. https://cs231n.github.io/neural-networks-1/
8. Stanford CS231n. Linear Classification. https://cs231n.github.io/linear-classify/
9. Stanford CS231n. Backpropagation, Intuitions (optimization-2). https://cs231n.github.io/optimization-2/
10. Johnson, J. Derivatives, Backpropagation, and Vectorization (CS231n handout). https://cs231n.stanford.edu/handouts/derivatives.pdf
11. Parr, T. & Howard, J. The Matrix Calculus You Need for Deep Learning. https://explained.ai/matrix-calculus/
12. Wang, R. Linear separation of a feature space (HMC E161 lecture notes). https://pages.hmc.edu/ruye/e161/lectures/NeuralNetworks/node18.html
13. Nair, V. & Hinton, G. Rectified Linear Units Improve Restricted Boltzmann Machines. ICML 2010. https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf
14. Glorot, X., Bordes, A. & Bengio, Y. Deep Sparse Rectifier Neural Networks. AISTATS 2011, PMLR 15: 315–323. https://proceedings.mlr.press/v15/glorot11a.html
15. Wikipedia. Perceptrons (book). https://en.wikipedia.org/wiki/Perceptrons_(book)
16. Baydin, A. G., Pearlmutter, B. A., Radul, A. A. & Siskind, J. M. Automatic Differentiation in Machine Learning: a Survey. JMLR 18(153), 2018. https://www.jmlr.org/papers/volume18/17-468/17-468.pdf
17. Griewank, A. Who Invented the Reverse Mode of Differentiation? Documenta Mathematica, Extra Vol. ISMP (2012) 389–400. https://papers.baulab.info/papers/also/Griewank-2012.pdf
18. Baydin, A. G. et al. Gradients without Backpropagation. 2022. https://arxiv.org/pdf/2202.08587
19. NVIDIA. Matrix Multiplication Background User's Guide. https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
20. NVIDIA. Linear/Fully-Connected Layers User's Guide. https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html
21. Netlib. BLAS (Basic Linear Algebra Subprograms). https://www.netlib.org/blas/
22. Demmel, J. CS267 Lecture 2: Designing fast linear algebra kernels in the presence of memory hierarchies. https://people.eecs.berkeley.edu/~demmel/cs267-1995/lecture02.html
23. Goh, G. Why Momentum Really Works. Distill, 2017. https://distill.pub/2017/momentum/
24. Pascanu, R., Mikolov, T. & Bengio, Y. On the difficulty of training recurrent neural networks. ICML 2013, PMLR 28(3): 1310–1318. https://ar5iv.labs.arxiv.org/html/1211.5063 (also https://proceedings.mlr.press/v28/pascanu13.html)
25. Glorot, X. & Bengio, Y. Understanding the difficulty of training deep feedforward neural networks. AISTATS 2010, PMLR 9: 249–256. https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
26. Baldi, P. & Hornik, K. Neural networks and principal component analysis: learning from examples without local minima. *Neural Networks* 2(1): 53–58, 1989 (volume and pages to verify). https://igb.uci.edu/~pfbaldi/publications/journals/1989/sdarticle.pdf
27. Wikipedia. Universal approximation theorem. https://en.wikipedia.org/wiki/Universal_approximation_theorem
28. He, K., Zhang, X., Ren, S. & Sun, J. Delving Deep into Rectifiers. ICCV 2015. https://ar5iv.labs.arxiv.org/html/1502.01852 (also https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper)
29. Vaswani, A. et al. Attention Is All You Need. 2017. https://ar5iv.labs.arxiv.org/html/1706.03762
