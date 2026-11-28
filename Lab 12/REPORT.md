# Submission Report — CNN Feature Visualization, Parameter Audit & Critical Analysis

**Deep Learning & Computer Vision — Practical Laboratory Assignment**

| Field | Value |
|---|---|
| Student Name | _(fill in)_ |
| Registration / Roll No. | _(fill in)_ |
| Course / Section | _(fill in)_ |
| Submission Date | _(fill in)_ |

**Files produced by this submission:** `CNN_Visualization_Assignment.ipynb` (full code + answers), `results/` (grid, summaries, CSVs), `README.md`, `HANDWRITTEN_CALCULATIONS.md`.

---

## Part II — Parameter Audit (summary)

Formula given in the assignment: **Trainable parameters per Conv layer = (Kh × Kw × Cin + 1) × Cout**

| Model | Conv Architecture | Kernel Size | Pooling Type | Manual Count | Code Output |
|---|---|---|---|---|---|
| Model A (Standard) | Conv1: 5 filters; Conv2: 10 filters | 3 × 3 | Max Pooling (2×2, stride 2) | **710** | **710** ✓ |
| Model B (Large Kernel) | Conv1: 5 filters; Conv2: 10 filters | 5 × 5 | Average Pooling (2×2, stride 2) | **1750** | **1750** ✓ |

* Step-by-step hand calculations to copy onto your handwritten sheet → **`HANDWRITTEN_CALCULATIONS.md`**
* Framework summaries (attach alongside the handwritten sheet) → `results/model_A_summary.txt/.png`, `results/model_B_summary.txt/.png`
* Machine-readable audit → `results/parameter_audit.csv`, `results/parameter_audit_detailed.csv`
* Full increase / receptive-field arithmetic is verified in the notebook (Part II, last code cell).

---

## Part III — Critical Thinking & Viva Defense Answers

> Every answer references **specific rows and columns of our own 5 × 4 grid**
> (`results/fig1_5x4_feature_grid.png`) and the numbers in `results/stage_statistics.csv`.

### Question 1 — The ReLU "Blackout" Effect & Activation Alternatives

**Which regions turned to zero, and why were they negative.**
Look at **Row 1 (Horizontal Edge, Sobel-X), columns A → B**. In Stage A every vertical seam of the
checkerboard façade appears as a *pair* of streaks: a bright one (black→white transition) and a dark
one (white→black transition). In Stage B the **dark streaks vanish completely** — they are pinned to
exactly 0. Concretely, **62.72 % of all pixels in Row-1 Stage-B are exactly zero**
(`stage_statistics.csv`), and **57.99 % in Row 2 (Sobel-Y)**. The regions that went to zero are:

1. **The negative half of every edge** — for Sobel-X, the *right-hand side of each white panel where it
   meets a black panel* and the bright-frame→dark-glass transitions inside the windows; for Sobel-Y
   (Row 2), horizontal seams where the pixel *below* is darker than the pixel above (bottom edge of a
   bright frame into black glass).
2. **Every flat interior** (inside a black panel, inside a white panel) — the weighted sum there is
   already 0, so it stays 0.

**Why those pixels held negative values:** the kernels are *linear difference* operators. Sobel-X is
`[-1 0 1]` across columns, i.e. a smoothed `∂I/∂x = right − left`: wherever the neighbourhood gets
**darker toward +x**, the weighted sum is negative. Sobel-Y computes `∂I/∂y` and goes negative wherever
the neighbourhood darkens downward. Negative values are therefore *anti-correlated* local structure —
the "shadow side" of each edge — not noise. (Row 4, Gaussian Blur, has an **all-positive kernel summing
to 1**, so Stage A ≥ 0.12 and **0.00 %** of its pixels are zeroed: direct evidence that ReLU only kills
negatives. Row 5 Emboss zeroes 6.22 % — exactly its shadow sides; Row 3 Sharpen zeroes only 3.24 % —
the thin dark ringing lines around the window rims plus the zero-padded border.)

**If ReLU were replaced by Leaky ReLU (α = 0.1):**
mathematically `y = x if x ≥ 0 else 0.1x`. The blackout regions would **not be zero** — Row-1 Stage-B
would show the dark streaks again at 10 % amplitude (min ≈ −0.1 × 674 ≈ **−67** instead of 0), i.e.
faint grey "shadow sides" of every edge on the shared row scale, while bright sides stay unchanged
(max 606). Gradients become `1` for x ≥ 0 and `0.1` for x < 0, so **no neuron can die** and the map
keeps a dense, asymmetric dynamic range instead of a sparse binary one.

**If ReLU were replaced by Sigmoid** `σ(x) = 1/(1+e^(−x))`:
the output range becomes **(0, 1)** — **no pixel is ever 0**, so there is *no blackout at all*: flat
regions (x = 0) map to 0.5 (mid-grey), negatives map into (0, 0.5), positives into (0.5, 1). Because
our raw responses reach ±600, the sigmoid **saturates** (|x| > 5 ⇒ ≈ 0 or 1), so the graded edge ramps
collapse into nearly-binary black/white with flat gradients (vanishing-gradient problem), contrast is
compressed, and the output is no longer zero-centred — which slows subsequent-layer learning.

---

### Question 2 — Max Pooling vs. Average Pooling Information Retention

**Columns C vs D, Rows 1–3 (Sharpen and the two Edge filters).**
Max Pooling keeps the **strongest pixel of each 2 × 2 window**, so thin high-frequency structures keep
their **full amplitude**: the peak of each pooled row equals the Stage-A maximum exactly
(`C_max = 606 / 499 / 450` for Rows 1/2/3). Average Pooling replaces the window by its mean, mixing
the peak with its (often zero-valued) neighbours, so the same peaks drop to `D_max = 435.25 / 377.5 /
303.5` — the **Sharpen rim highlights lose 32 % of their intensity** (450 → 303.5) and the edge streaks
of Rows 1–2 lose ~28 %. In the grid you can see it directly: in **column C** the window-frame rims
(Row 3) and the vertical/horizontal edge lines (Rows 1–2) are **crisp, thin and bright**; in
**column D** they are **dimmer, thicker and smeared** — the classic washed-out look. (Row 1 mean drops
from 20.55 in C to 12.54 in D because averaging also dilutes the response with surrounding zeros.)
So: **Max Pool preserves high-frequency peaks and thin lines; Average Pool low-pass filters them.**
(For the already-smooth Gaussian row the two nearly agree — 186.81 → 186.81 vs 184.08 — which proves
the difference only matters on high-frequency content.)

**Dark ink text on a pure white background (background ≈ 255, strokes ≈ 0):**
**Average Pooling preserves the strokes better.** In any 2 × 2 window that contains a thin dark stroke,
**Max Pooling selects the brightest pixel = 255 (background)**, so every window containing background
wins and the stroke pixels are erased — morphologically, a max filter *dilates* the white background and
eats the dark text. **Average Pooling** returns e.g. `(255+255+0+255)/4 ≈ 191` for a window with a
stroke, so the stroke survives as a visibly darker grey dip against the 255 background and the word
remains readable (at reduced contrast). Only if the text were pre-processed into *bright strokes on a
zero background* (an edge-detector output) would Max Pooling be the right choice.

---

### Question 3 — Parameter Explosion & Receptive Field Equivalence

**Percentage increase from Model A to Model B:**

```
(1750 − 710) / 710 × 100 = 1040 / 710 × 100 = 146.4789 % ≈ 146.48 %
```

Model B has **≈ 2.46×** the parameters of Model A — a **146.48 % increase** — caused only by the
kernel change 3×3 → 5×5 (the pooling type and the GAP/Dense tail are identical).

**Receptive-field equivalence:** a single 5 × 5 layer has receptive field **5**. Each 3 × 3 layer adds
`(3 − 1) = 2` pixels (stride 1), so **two stacked 3 × 3 layers give RF = 3 + 2 = 5 — exactly the same
field of view.** **2 layers are required.**

**Parameters saved** (weights + biases), replacing one 5×5 by two 3×3 with the same channel counts
(Cin = Cmid = Cout = C):

```
single 5×5 : (25C + 1)·C = 25C² + C
two 3×3    : 2 × (9C + 1)·C = 18C² + 2C
saved      = 7C² − C
```

* For **C = 10** (the setting of Model B's Conv2): `(250+1)·10 = 2510` vs `2 × (90+1)·10 = 1820`
  → **690 parameters saved = 27.49 % fewer** (and 27 % fewer multiply-accumulates: 25C² vs 18C²).
* For C = 5: 630 vs 460 → 170 saved (26.98 %); for C = 3: 228 vs 168 → 60 saved (26.32 %).
* General form with different channel counts:
  `(25·Cin+1)·Cout − [(9·Cin+1)·Cmid + (9·Cmid+1)·Cout]` — the saving grows as Cmid shrinks.

Beyond parameters, the stacked version inserts an **extra non-linearity (ReLU) between the two 3×3
convolutions**, which increases representational power for the same RF — this is precisely why VGG-style
nets use stacked 3×3 instead of large kernels.

---

### Question 4 — Operation Commutativity (Order Invariance)

**Answer: the resulting output tensor is IDENTICAL.** Conv → ReLU → MaxPool and Conv → MaxPool → ReLU
produce the same values everywhere.

**Proof with a 2 × 2 patch containing negatives** (the feature map right after convolution):

```
P = [ [ 2, −3],
      [ 1, −4] ]

Path 1 (ReLU → MaxPool):  ReLU(P) = [ [2, 0], [1, 0] ]  →  max = 2
Path 2 (MaxPool → ReLU):  max(P)  = max(2, −3, 1, −4) = 2  →  ReLU(2) = 2        ✔ equal
```

**General proof:** for any set of values S and any **non-decreasing** function f (ReLU is non-decreasing),

```
f( max S ) = max{ f(s) : s ∈ S }        because s ≤ max S  ⇒  f(s) ≤ f(max S)
```

ReLU is element-wise and monotone, so `ReLU(max(P)) = max(ReLU(P))` for **every** window — the two
orderings agree pixel-for-pixel, so the full output tensors are identical. (Same patch under Average
Pooling shows the rule fails there: ReLU-then-Avg = (2+0+1+0)/4 = **0.75**, Avg-then-ReLU =
max(0, −1) = **0** — *different*.)

**Computational efficiency implication:** since the outputs are provably equal, a framework may
**legally reorder** the cheaper-first way: pool **before** ReLU so the element-wise ReLU runs on
`H/2 × W/2` = **¼ as many elements (75 % fewer element-wise ops and memory writes)** and the
intermediate activation buffer is 4× smaller (less cache traffic, easier kernel fusion). The dominant
convolution cost is unchanged, so the swap is a free win — and it is only safe *because* ReLU is
monotone; with Average Pooling (or any non-max-compatible op) the swap would silently change results.

---

### Question 5 — Diagnosing Filter Redundancy (Debugging Scenario)

**Root cause 1 — Symmetric / degenerate weight initialization.**
If Layer-1's five filters start from identical (or near-identical) values — e.g. all zeros, a constant,
the same hand-written weights, or an init range so tiny (±0.001) that all five kernels are effectively
equal — then at epoch 0 all five feature maps are identical, each filter receives the *same* gradient,
and symmetry is **never broken**: after 20 epochs they have moved together and still produce nearly
identical maps.
**Fix:** draw every filter independently from a proper variance-scaled distribution — **Kaiming/He
normal** (`std = √(2/fan_in)`) for ReLU networks (Glorot/Xavier for tanh/sigmoid) — and assert at
initialization that the five kernels differ (check pairwise distance / plot them). Add a tiny
per-filter jitter if any shared/constant init exists in the code.

**Root cause 2 — Learning-rate pathology (too small, too large, or over-regularized).**
With an LR that is far too small the filters drift negligibly from their (near-)identical init within
20 epochs, so collapse persists; with an LR that is too large the updates oscillate/diverge and the
filters never settle into distinct specializations; an excessive, constant weight decay additionally
pulls all filters toward the same shrunk, symmetric solution.
**Fix:** tune the schedule — e.g. **Adam with lr ≈ 1e-3 + warmup + cosine/step decay** (or SGD+momentum
with an LR-finder sweep), lower/anneal the weight decay, clip gradients — and *monitor* per-filter
weight distance and activation diversity during training instead of only at the end.

**Root cause 3 (classic) — Dead ReLUs.**
If a filter's pre-activations are always negative (bad bias/initialization, or a too-large LR that
pushed weights into a killing regime), ReLU outputs are identically 0 → **zero gradient** → the filter
is frozen forever, and several filters that die the same way look identically empty.
**Fix:** He initialization, small positive bias init, **BatchNorm before ReLU**, switch to
**Leaky-ReLU**, lower the LR; watch the "% dead units" per layer during training.

*(Data-side bonus cause: too little input diversity or no augmentation lets all filters converge to the
same dominant pattern — fix with augmentation and a diverse dataset.)*

---

*All numbers above are regenerated by running `CNN_Visualization_Assignment.ipynb`
(Kernel → Restart & Run All) and are defendable in the viva.*
