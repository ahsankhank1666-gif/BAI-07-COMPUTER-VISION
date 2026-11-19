# Handwritten Implementations — Theory, Equations and Worked Examples

> **Course:** BAI-07 Computer Vision · **Student:** Muhammad Ahsan Khan (FA23-BAI-052)  
> **Companion files:** [`handwritten_demo.py`](handwritten_demo.py) and
> [`results/handwritten_step_by_step.txt`](results/handwritten_step_by_step.txt) — every number below is
> recomputed from the real dataset by that script.

---

## 0. The small sample used for all by-hand solutions

The first five houses of [`data/boston_housing.csv`](data/boston_housing.csv) are used so that every
arithmetic step is checkable by hand. Only 1–2 features are used in the manual examples; the Python
implementation uses all 13 features.

| House | `rm` (avg. rooms) | `lstat` (% lower status) | `medv` (target, $1000s) |
|---|---:|---:|---:|
| H1 | 6.575 | 4.98 | 24.0 |
| H2 | 6.421 | 9.14 | 21.6 |
| H3 | 7.185 | 4.03 | 34.7 |
| H4 | 6.998 | 2.94 | 33.4 |
| H5 | 7.147 | 5.33 | 36.2 |

---

## 1. Linear Regression

### 1.1 Working procedure

1. Assume the target is a linear combination of the inputs plus an intercept.
2. Build the design matrix $X$ (a column of ones for the intercept, then one column per feature).
3. Choose the coefficients that minimise the residual sum of squares (ordinary least squares).
4. Solve the normal equations (or use a numerically stable QR/SVD solver, as scikit-learn does).
5. Predict by the dot product of the coefficient vector with a new row.

### 1.2 Mathematical equations

$$
\hat{y}_i = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip} = \mathbf{x}_i^{T}\boldsymbol{\beta}
$$

$$
SSE(\boldsymbol{\beta}) = \sum_{i=1}^{n}\left(y_i - \mathbf{x}_i^{T}\boldsymbol{\beta}\right)^2,
\qquad
\frac{\partial SSE}{\partial \boldsymbol{\beta}} = 0
\;\Rightarrow\;
\hat{\boldsymbol{\beta}} = (X^{T}X)^{-1}X^{T}\mathbf{y}
$$

For a **single** feature this reduces to:

$$
b_1 = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^{n}(x_i-\bar{x})^2} = \frac{S_{xy}}{S_{xx}},
\qquad
b_0 = \bar{y} - b_1\bar{x}
$$

### 1.3 Step-by-step solution (feature `rm`, houses H1–H3)

**Step 1 — means**

$$
\bar{x} = \frac{6.575+6.421+7.185}{3} = 6.72700,
\qquad
\bar{y} = \frac{24.0+21.6+34.7}{3} = 26.76667
$$

**Step 2 — deviation products**

| House | $x_i-\bar x$ | $y_i-\bar y$ | $(x_i-\bar x)(y_i-\bar y)$ | $(x_i-\bar x)^2$ |
|---|---:|---:|---:|---:|
| H1 | −0.15200 | −2.76667 | +0.42053 | 0.02310 |
| H2 | −0.30600 | −5.16667 | +1.58100 | 0.09364 |
| H3 | +0.45800 | +7.93333 | +3.63347 | 0.20976 |

**Step 3 — sums and coefficients**

$$
S_{xy} = 5.63500,\quad S_{xx} = 0.32650,\quad
b_1 = \frac{5.63500}{0.32650} = 17.25859
$$

$$
b_0 = 26.76667 - 17.25859 \times 6.72700 = -89.33190
$$

$$
\boxed{\hat{y} = -89.33190 + 17.25859\,\text{rm}}
$$

**Step 4 — predictions:** H1 → 24.1434 (actual 24.0), H2 → 21.4855 (actual 21.6),
H3 → 34.6711 (actual 34.7). With two fitted coefficients and three points, least squares
minimises the sum of squared residuals; the small residuals reflect a near-linear trend.

**Step 5 — multivariate extension (all 5 houses, features `rm` and `lstat`)**

$$
X^{T}X =
\begin{bmatrix}
5.000 & 34.326 & 26.420\\
34.326 & 236.136 & 179.055\\
26.420 & 179.055 & 161.633
\end{bmatrix},
\qquad
X^{T}\mathbf{y} =
\begin{bmatrix}149.900\\1038.268\\747.927\end{bmatrix}
$$

$$
\hat{\boldsymbol{\beta}} = (-101.6966,\;19.1664,\;0.0180)
\quad\text{(verified identical to `LinearRegression` in scikit-learn)}
$$

For H1: $\hat y = -101.6966 + 19.1664(6.575) + 0.0180(4.98) = 24.4124$.

---

## 2. Decision Tree Regression

### 2.1 Working procedure

1. Start with all training samples in one node; its prediction is the mean of $y$ in that node.
2. For every feature and every candidate threshold $s$, split the node into two regions
   ($x_j \le s$ and $x_j > s$).
3. Measure the impurity (MSE) of each child and compute the variance reduction.
4. Keep the split with the largest reduction; recursively repeat until stopping criteria
   (max depth / min samples) are reached.
5. A new sample falls through the tree; the mean $y$ of its terminal leaf is the prediction.

### 2.2 Mathematical equations

$$
\bar{y}_m = \frac{1}{N_m}\sum_{i \in m} y_i,
\qquad
Q(m) = \frac{1}{N_m}\sum_{i \in m}(y_i-\bar{y}_m)^2
$$

$$
\text{Gain}(s) = Q(\text{parent}) -
\frac{N_L}{N}Q(\text{left}) - \frac{N_R}{N}Q(\text{right}),
\qquad
s^* = \arg\max_s \text{Gain}(s)
$$

$$
\hat{y}(\mathbf{x}) = \bar{y}_{m(\mathbf{x})}
$$

### 2.3 Step-by-step solution (feature `lstat`, houses H1–H4)

Candidate split: $s = 4.505$ (midpoint of 2.94 and 4.03).

* Left ($lstat \le 4.505$): H3, H4 → $y=[34.7, 33.4]$
* Right ($lstat > 4.505$): H1, H2 → $y=[24.0, 21.6]$

**Step 1 — parent impurity**

$$
\bar{y} = \frac{24.0+21.6+34.7+33.4}{4} = 28.4250
$$

$$
Q(\text{parent}) = \frac{(24-28.425)^2+(21.6-28.425)^2+(34.7-28.425)^2+(33.4-28.425)^2}{4}
= 32.5719
$$

**Step 2 — child impurities**

$$
\bar{y}_L = \frac{34.7+33.4}{2} = 34.0500,\quad
Q_L = \frac{(34.7-34.05)^2+(33.4-34.05)^2}{2} = 0.4225
$$

$$
\bar{y}_R = \frac{24.0+21.6}{2} = 22.8000,\quad
Q_R = \frac{(24-22.8)^2+(21.6-22.8)^2}{2} = 1.4400
$$

$$
\text{weighted child MSE} = \tfrac{2}{4}(0.4225) + \tfrac{2}{4}(1.4400) = 0.9313
$$

**Step 3 — gain and prediction**

$$
\text{Gain} = 32.5719 - 0.9313 = \mathbf{31.6406}
$$

Leaf predictions: H3, H4 → 34.05; H1, H2 → 22.80. A `DecisionTreeRegressor(max_depth=1)`
finds the identical threshold and leaf values — see the verification line in the log.

---

## 3. Random Forest Regression

### 3.1 Working procedure

1. Draw $B$ bootstrap samples (size $n$, sampled **with replacement**) from the training data.
2. Grow one CART regression tree on each bootstrap sample; at each split consider only a random
   subset of features (decorrelation).
3. About 37% of rows are left out of each tree ("out-of-bag" rows) and can be used for validation.
4. Average the predictions of all trees; averaging cancels out much of the variance of a single tree.

### 3.2 Mathematical equations

$$
\hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x})
$$

$$
\text{OOB-MSE} = \frac{1}{|\text{OOB}|}\sum_{i \in \text{OOB}}(y_i-\hat{f}_{\text{OOB},i})^2,
\qquad
\text{Gain}(s) = Q(\text{parent}) - \tfrac{N_L}{N}Q_L - \tfrac{N_R}{N}Q_R
$$

### 3.3 Step-by-step solution (two trees, five houses)

**Tree 1** — bootstrap indices `[0, 0, 2, 3, 3]`; out-of-bag: H2, H5.
Best stump: $s=4.505$, gain $=23.2067$, leaves $33.8333$ (left) and $24.0000$ (right).

**Tree 2** — bootstrap indices `[1, 1, 2, 4, 4]`; out-of-bag: H1, H4.
Best stump: $s=7.235$, gain $=47.7144$, leaves $35.7000$ (left) and $21.6000$ (right).

**Forest average:**

| House | Tree 1 | Tree 2 | Forest $\frac{T_1+T_2}{2}$ | Actual |
|---|---:|---:|---:|---:|
| H1 | 24.000 | 35.700 | **29.8500** | 24.0 |
| H2 | 24.000 | 21.600 | **22.8000** | 21.6 |
| H3 | 33.833 | 35.700 | **34.7667** | 34.7 |
| H4 | 33.833 | 35.700 | **34.7667** | 33.4 |
| H5 | 24.000 | 35.700 | **29.8500** | 36.2 |

The two trees disagree (H1: 24.0 vs 35.7); averaging pulls the extreme prediction toward the
centre and reduces variance. The full Python model averages **300** such trees.

---

## 4. Neural Network Regression (MLP)

### 4.1 Working procedure

1. Normalise inputs (essential when features have very different scales).
2. Initialise weights/biases; feed the input forward through hidden layers with a non-linearity.
3. Produce a **linear** output for regression (no sigmoid/softmax).
4. Compute a loss (MSE) and backpropagate its gradients with the chain rule.
5. Update weights with gradient descent / Adam; repeat for many epochs until convergence.

### 4.2 Mathematical equations

For layer $l$:

$$
z_j^{(l)} = \sum_i w_{ji}^{(l)} a_i^{(l-1)} + b_j^{(l)},\qquad
a_j^{(l)} = \mathrm{ReLU}(z_j^{(l)}) = \max(0, z_j^{(l)})
$$

Output (linear) and loss:

$$
\hat{y} = w_2 a_1 + b_2,\qquad
L = \tfrac{1}{2}(y-\hat{y})^2
$$

Backpropagation for the tiny 1-hidden-neuron net:

$$
\frac{\partial L}{\partial \hat y}=\hat y-y,\quad
\frac{\partial L}{\partial w_2}=(\hat y-y)a_1,\quad
\frac{\partial L}{\partial b_1}=(\hat y-y)w_2\,\mathbf{1}[z_1>0],\quad
\frac{\partial L}{\partial w_{1j}}=(\hat y-y)w_2\,\mathbf{1}[z_1>0]\,x_j
$$

$$
w \leftarrow w - \eta\,\frac{\partial L}{\partial w}
$$

### 4.3 Step-by-step solution (house H1, one ReLU neuron)

Initial values: $x=[6.575,\,4.98]$, $y=24.0$, $w_1=[0.5,\,-0.2]$, $b_1=-1.0$,
$w_2=0.8$, $b_2=1.0$, $\eta=0.001$.

**Forward pass**

$$
z_1 = 0.5(6.575) + (-0.2)(4.98) - 1.0 = 1.2915
$$

$$
a_1 = \max(0, 1.2915) = 1.2915,\qquad
\hat{y} = 0.8(1.2915)+1.0 = 2.0332
$$

$$
L = \tfrac12(24.0-2.0332)^2 = 241.2702
$$

**Backward pass**

$$
\frac{\partial L}{\partial \hat y} = -21.9668,\quad
\frac{\partial L}{\partial w_2} = -28.3701,\quad
\frac{\partial L}{\partial b_2} = -21.9668
$$

$$
\frac{\partial L}{\partial w_1} = [-115.5454,\,-87.5157],\qquad
\frac{\partial L}{\partial b_1} = -17.5734
$$

**Update**

$$
w_2^{new} = 0.8 - 0.001(-28.3701) = 0.828370,\quad
b_2^{new} = 1.021967
$$

$$
w_1^{new} = [0.615545,\,-0.112484],\quad b_1^{new} = -0.982427
$$

New prediction $=3.0967$ (closer to 24.0), new loss $=218.4737$. Repeating this over all
samples with Adam implements the full network. The actual model uses two hidden layers
(64 and 32 neurons) and converged after **314** iterations.

---

## 5. XGBoost Regression

### 5.1 Working procedure

1. Start with a constant prediction $F_0(x)$ (the mean of $y$ for squared-error loss).
2. At each boosting round, compute the residual (negative gradient) for every training row.
3. Fit a small tree $f_t$ to those residuals (depth 3 in the Python model).
4. Compute optimal leaf weights with L2 regularisation $\lambda$; shrink them by $\eta$.
5. Add the tree to the ensemble: $F_t = F_{t-1} + \eta f_t$. Repeat 400 times.

### 5.2 Mathematical equations

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(\mathbf{x}_i)
$$

$$
\mathcal{L}^{(t)} = \sum_i \left(y_i-\hat{y}_i^{(t-1)}-f_t(\mathbf{x}_i)\right)^2
 + \sum_j \left(\gamma T + \tfrac12 \lambda w_j^2\right)
$$

For squared error, $g_i = \hat y_i^{(t-1)}-y_i$ and $h_i = 1$, hence the optimal leaf weight and
split gain are:

$$
w_j^{*} = -\frac{G_j}{H_j+\lambda} = \frac{\sum_{i\in j}(y_i-\hat y_i)}{\#\{i\in j\}+\lambda},
\qquad
\text{Gain} = \tfrac12\left[
\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}
-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma
$$

### 5.3 Step-by-step solution (one boosting round, houses H1–H4)

**Step 1 — base score**

$$
F_0(x) = \bar{y} = \frac{24.0+21.6+34.7+33.4}{4} = 28.4250
$$

**Step 2 — residuals $r_i = y_i - F_0$**

| House | H1 | H2 | H3 | H4 |
|---|---:|---:|---:|---:|
| $r_i$ | −4.4250 | −6.8250 | +6.2750 | +4.9750 |

**Step 3 — fit depth-1 tree on residuals, split at $lstat=5.005$**

Left (H1, H3, H4): $G_L = +6.8250$, $H_L = 3$ →
$w_L = 6.825/(3+1) = \mathbf{+1.70625}$.

Right (H2): $G_R = -6.8250$, $H_R = 1$ →
$w_R = -6.825/(1+1) = \mathbf{-3.41250}$.

With $\lambda=1,\ \gamma=0$:

$$
\text{Gain} = \tfrac12\left[\frac{6.825^2}{4}+\frac{6.825^2}{2}-0\right] = 17.4677 > 0
\;\Rightarrow\; \text{keep the split}
$$

**Step 4 — update $F_1 = F_0 + \eta\,w$ with $\eta=0.3$**

| House | Leaf | $F_1$ prediction | Actual |
|---|---:|---:|---:|
| H1 | +1.7062 | 28.9369 | 24.0 |
| H2 | −3.4125 | 27.4013 | 21.6 |
| H3 | +1.7062 | 28.9369 | 34.7 |
| H4 | +1.7062 | 28.9369 | 33.4 |

Each subsequent tree fits the residuals that remain after this update (full model: 400 trees,
depth 3, $\eta=0.05$, `reg:squarederror`).

---

## 6. K-Nearest Neighbours Regression

### 6.1 Working procedure

1. Choose $k$ and a distance metric (Euclidean by default).
2. Scale features first so that large-range variables do not dominate the distance.
3. Compute the distance from the query point to every training point.
4. Select the $k$ closest training rows.
5. Predict the **average** of their target values (optionally distance-weighted).

### 6.2 Mathematical equations

$$
d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{p}(x_{ij}-x_j)^2}
\quad\text{(on standardized features in the real model)}
$$

$$
N_k(\mathbf{x}) = \text{indices of the } k \text{ smallest distances},\qquad
\hat{y}(\mathbf{x}) = \frac{1}{k}\sum_{i\in N_k(\mathbf{x})} y_i
$$

Weighted variant: $\hat{y} = \dfrac{\sum_{i\in N_k} y_i / d_i}{\sum_{i\in N_k} 1/d_i}$.

### 6.3 Step-by-step solution (query H1, $k=3$, raw `rm` and `lstat`)

$$
d = \sqrt{(rm-6.575)^2 + (lstat-4.98)^2}
$$

| Rank | House | $(rm, lstat)$ | $d^2$ | $d$ | In $k=3$? | $y$ |
|---:|---|---|---:|---:|---|---:|
| 1 | H5 | (7.147, 5.33) | 0.4497 | 0.6706 | yes | 36.2 |
| 2 | H3 | (7.185, 4.03) | 1.2746 | 1.1290 | yes | 34.7 |
| 3 | H4 | (6.998, 2.94) | 4.3405 | 2.0834 | yes | 33.4 |
| 4 | H2 | (6.421, 9.14) | 17.3293 | 4.1628 | no | 21.6 |

$$
\hat{y}(H1) = \frac{36.2 + 34.7 + 33.4}{3} = \mathbf{34.7667}
$$

Actual price $=24.0$, so the absolute error is $10.7667$. This example also illustrates why scaling
matters: in the full model all 13 features are standardised before distances are computed.

---

## 7. Summary of handwritten vs. programming parts

| Algorithm | By-hand example size | Equations demonstrated | Full Python model |
|---|---|---|---|
| Linear Regression | 3 houses, 1 feature (+5 houses, 2 features) | normal equation, OLS | all 13 features, 404 training rows |
| Decision Tree | 4 houses, 1 feature | variance reduction / MSE gain | CART, unlimited depth |
| Random Forest | 5 houses, 2 bootstrap trees | bootstrap + averaging | 300 bootstrap trees |
| Neural Network | 1 house, 1 hidden neuron | forward pass, ReLU, backprop, SGD | 64–32 ReLU MLP, Adam |
| XGBoost | 4 houses, 1 boosting round | residuals, leaf weights, gain, shrinkage | 400 trees, depth 3, $\eta=0.05$ |
| KNN | 1 query, 4 candidates, 2 features | Euclidean distance, mean of $k$ neighbours | $k=5$, 13 standardised features |

All six derivations use a tiny subset of the **same** dataset and the **same** target variable as the
programming part, so the handwritten and code results are directly comparable.
