## Introduction — Theoretical Fundamentals

This chapter lays the theoretical groundwork that underpins **AutoArimaExplorer**. It focuses on why we model, what ARIMA models are, how we select/diagnose them, and how we evaluate forecasts—especially under the practical constraints typical of financial time series (short memory, heavy tails, and time‐varying volatility).

---

### 1) The modeling goal

Given a univariate time series $\{y_t\}_{t=1}^T$, we want to:

1. **Explain** short-run serial dependence parsimoniously.
2. **Forecast** $y_{T+h}$ (and uncertainty around it) with stable, out-of-sample performance.
3. **Diagnose** whether remaining structure in the residuals violates standard modeling assumptions (e.g., autocorrelation).

AutoArimaExplorer automates this pipeline with robust fitting strategies, information-criterion selection, residual diagnostics, and rolling-origin validation.

---

### 2) ARIMA models at a glance

An **ARIMA(p, d, q)** model applies $d$ differences to make the series “close enough” to stationary, then fits an ARMA(p, q) to the differenced series.

* **Differencing**:

  $$
  \Delta^d y_t = (1 - B)^d y_t
  $$

  where $B$ is the backshift operator, $By_t = y_{t-1}$.

* **ARMA(p, q)** on $x_t = \Delta^d y_t$:

  $$
  \phi(B)\, x_t = \theta(B)\, \varepsilon_t,\quad \varepsilon_t \sim \text{i.i.d. }(0, \sigma^2)
  $$

  with $\phi(B) = 1 - \phi_1B - \dots - \phi_p B^p$ and $\theta(B) = 1 + \theta_1B + \dots + \theta_q B^q$.

**Interpretation**

* $p$: autoregressive memory (how many past values of $x_t$ matter).
* $d$: differencing order (how much trend we remove).
* $q$: moving-average memory (how many past shocks matter).

**Practical note in finance**
Prices are often modeled via **log-returns** (already close to stationary), for which $d=0$ is common; levels may require $d\ge 1$.

---

### 3) Stationarity, differencing, and trend

* **Weak stationarity** (constant mean/variance, ACF depends only on lag) is a working assumption for ARMA modeling.
* **Differencing** helps when trends or unit roots are present. Over-differencing injects moving-average structure and can inflate forecast variance.
* **Trend terms**: In integrated models, trend terms of order lower than $d$ are redundant (e.g., a constant is eliminated by first differencing). AutoArimaExplorer enforces sensible trend choices by $d$:

  * $d=0$: allow constant (“c”) or no constant (“n”).
  * $d=1$: allow drift (“t”) or none (“n”).
  * $d\ge 2$: no explicit trend (lower-order trends are differenced out).

---

### 4) Identification and parsimony

Classical Box–Jenkins identification uses the **ACF/PACF** to suggest $p, q$, but automated search commonly relies on **information criteria**:

* **BIC (Bayesian Information Criterion)**:

  $$
  \text{BIC} = -2\log L + k \log T
  $$

  with $k$ number of estimated parameters; **smaller is better**.
  BIC penalizes complexity more strongly than AIC, encouraging parsimonious models that typically forecast more robustly in practice.

AutoArimaExplorer:

* Fits a grid of $(p,d,q)$ with robust optimizers and constraints toggled.
* Selects by **minimum BIC**, optionally breaking ties by parsimony (smaller $p+q$).

---

### 5) Residual diagnostics (guardrails)

A fitted model should leave **white-noise residuals**:

* **No linear autocorrelation** in $\varepsilon_t$.
* (Often in finance) Variance can be time-varying; ARIMA won’t fix conditional heteroskedasticity.

We use **Ljung–Box** tests as guardrails:

* On residuals $\varepsilon_t$ → checks leftover linear dependence.
* Optionally on **squared residuals** $\varepsilon_t^2$ → crude check for short-run volatility clustering (ARCH-type effects).

Decision rule: **fail** (reject white noise) if any p-value at selected lags is below $\alpha$. Models that fail can be **filtered out** during selection.

---

### 6) Convergence and stability

Likelihood-based ARIMA fitting can struggle when:

* Roots are near the unit circle (non-stationary/invertible starts).
* The likelihood surface is flat or multi-modal.
* The sample is short relative to $p+q+d$.

AutoArimaExplorer mitigates this by:

* Trying **multiple estimation strategies** per candidate:

  * For $d=0$, **innovations MLE** (fast, stable) and state-space with L-BFGS.
  * For $d\ge 1$, state-space fits with/without stationarity/invertibility enforcement.
* Discarding pathological fits (non-finite BIC).
* Optionally **requiring convergence**.

---

### 7) How much data do we need?

A pragmatic heuristic (Box–Jenkins-style):

$$
T - d \;\ge\; \max\big( \text{base},\; \text{per\_param}\cdot(p+q+d+1)\big),
$$

with typical defaults like **base = 50**, **per\_param = 10**.
This avoids overfitting high-order models on very short samples.

---

### 8) Out-of-sample (OOS) validation

In-sample fit is not enough. We assess **forecast performance** via **rolling-origin** validation:

1. Choose a minimal training window.
2. For each fold $t$: fit on $[1..t]$, forecast $h$ steps ahead, compare to actuals.
3. Aggregate errors across folds.

**Metrics**

* **RMSE**: root mean squared error (penalizes large misses).
* **MAE**: mean absolute error (robust to outliers vs RMSE).
* **MAPE**: mean absolute percentage error (scale-free; beware near zero).
* **Winkler score** (for prediction intervals): balances **interval width** and **coverage**; **lower is better**.

AutoArimaExplorer can select the model with **lowest OOS score**, complementing BIC-based selection.

---

### 9) Prediction intervals (PIs) and Winkler score

For a nominal level $1-\alpha$ (e.g., 95%), a PI $[L_t, U_t]$ should contain the realization with probability $1-\alpha$.
The **Winkler score** penalizes both **width** and **misses**:

$$
W =
\begin{cases}
U - L, & \text{if } y \in [L, U] \\
(U - L) + \tfrac{2}{\alpha}(L - y), & \text{if } y < L \\
(U - L) + \tfrac{2}{\alpha}(y - U), & \text{if } y > U \\
\end{cases}
$$

Averaging $W$ across folds gives a scalar measure of **sharpness + calibration**.

---

### 10) When ARIMA isn’t enough

ARIMA captures **linear** serial dependence in the mean. It does **not** model:

* **Conditional heteroskedasticity** (GARCH-type effects).
* **Nonlinear dynamics** (thresholds, regime switches).
* **Seasonality** (unless explicitly modeled via SARIMA).

If Ljung–Box on squared residuals fails or volatility clustering is clear, consider **ARIMA + GARCH**, or robust forecasting of returns with a separate volatility model.

---

### 11) Notation used throughout

* $y_t$: original series (levels or log-levels).
* $x_t = \Delta^d y_t$: differenced (working) series.
* $p, d, q$: AR, integration, MA orders.
* $\varepsilon_t$: one-step-ahead residuals/innovations.
* $h$: forecast horizon; $T$: sample size.

---

### 12) How this theory informs AutoArimaExplorer

* **Robust fitting** across trends/constraints reflects stationarity/invertibility and estimation stability concerns.
* **BIC-first selection** embodies parsimony; **Ljung–Box guardrails** enforce whiteness.
* **Rolling OOS** closes the loop on predictive validity.
* **Data sufficiency checks** avoid overfitting with short samples.
