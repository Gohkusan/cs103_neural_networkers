# Neural Networks Expressed in Linear Algebra

SMU CS103 (Linear Algebra for Computing) group project by Zyik, Voon, Zhe An, Gong Zu and Haide.

**Thesis:** a feed-forward neural network is a chain of matrix multiplications with a bend in between. The forward pass is matrix–vector and matrix–matrix products. Backpropagation uses only three more operations from Week 2: the transpose, the Hadamard (element-wise) product and the outer product.

## What's here

| Folder | Contents |
|---|---|
| `report/report.md` | Full research report: derivations in LA4CS notation, worked XOR example, experiment results, 27-question Q&A bank, accuracy checklist, references |
| `notes/` | Cited research notes behind the report (forward pass and activations; backprop, gradient descent and batching; links to least squares, eigenvalues and null space; worked example and experiment) |
| `code/` | NumPy-only scripts in matrix form, with their recorded outputs |
| `figures/` | Activation functions, decision boundaries and loss curves produced by the code |
| `slides/` | Source of the 14-slide presentation deck (one HTML file per slide, with speaker notes) |
| `docs/handoff.md` | Project brief, notation rules and slide-design rules the work follows |

## Notation

Vectors are bold column vectors and matrices are bold capitals in square brackets. **W**⁽ˡ⁾ ∈ ℝ^(nₗ × nₗ₋₁), and the superscript (l) is a layer index, not a power. φ is the activation; σ is never used for it, because σ means singular value in the course notation. The loss is ℓ, the error signal is **δ**⁽ˡ⁾ = ∂ℓ/∂**z**⁽ˡ⁾, and a batch stores samples as columns: **X** ∈ ℝ^(n₀ × m). In the code, `@` is matrix multiplication and `*` is the element-wise product.

## Running the code

Requires Python 3 with NumPy and Matplotlib.

```bash
cd code
python3 worked_example.py      # XOR running example: forward pass, backprop, one gradient step, finite-difference check
python3 backprop_gradcheck.py  # matrix-form backprop checked against finite differences
python3 experiment.py          # identity / logistic / tanh / ReLU on XOR and two moons, 5 seeds
python3 make_figures.py        # redraws ../figures/ from experiment.py's saved state
python3 batching_benchmark.py  # loop of matrix–vector products vs one matrix–matrix product
```

The `*_output.txt` files hold the outputs we recorded. Timings depend on hardware; ours came from a 2-core CPU, and no GPU was measured.

## Key results

- With the identity activation the network cannot learn XOR: it outputs 0.5 everywhere, exactly as the collapse argument predicts.
- ReLU, tanh and logistic all solve XOR on every seed. Logistic is several times slower to converge.
- One matrix–matrix product ran about 6× faster than a loop of matrix–vector products doing the same 208 million FLOPs.

## Main references

Goodfellow, Bengio & Courville, *Deep Learning* (2016); Nielsen, *Neural Networks and Deep Learning*; Parr & Howard, "The Matrix Calculus You Need for Deep Learning"; Stanford CS231n notes; Baydin et al., JMLR (2018); Pascanu, Mikolov & Bengio, ICML (2013); Goh, Distill (2017). The full list with links is at the end of the report.
