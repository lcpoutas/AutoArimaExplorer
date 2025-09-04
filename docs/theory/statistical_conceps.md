# Statistical Concepts & Diagnostics

This chapter unpacks the statistical machinery behind ARIMA modeling and behind the guardrails that **AutoArimaExplorer** uses. It’s deliberately rigorous: formulas, assumptions, and the “why” behind each diagnostic and selection rule. If you digest this chapter, you’ll understand every knob we turn.

---

## 1) Stationarity & Unit Roots

### 1.1 Weak stationarity

A process $\{x_t\}$ is (weakly) stationary if its first two moments do not depend on $t$:

* $\mathbb{E}[x_t] = \mu$ (constant)
* $\mathrm{Var}(x_t) = \gamma(0)$ (finite, constant)
* $\mathrm{Cov}(x_t, x_{t+h}) = \gamma(h)$ (depends only on lag $h$)

### 1.2 Unit roots & differencing

AR polynomial $\phi(B) = 1 - \phi_1 B - \dots - \phi_p B^p$.
A **unit root** occurs if $\phi(1)=0$ (i.e., root on the unit circle), implying a stochastic trend. Differencing removes it:

* If one unit root → use first difference $\nabla x_t = x_t - x_{t-1}$ (i.e., $d=1$).
* If two unit roots (rare) → $d=2$.

### 1.3 ADF and KPSS tests (complements)

**Augmented Dickey–Fuller (ADF):** tests $H_0$: **unit root** (non-stationary) vs $H_1$: stationary.
Regression (one version):

$$
\Delta x_t = \alpha + \beta t + \rho x_{t-1} + \sum_{i=1}^k \psi_i \Delta x_{t-i} + u_t.
$$

Rejecting $H_0$ (small p-value) supports stationarity.

**KPSS:** tests $H_0$: **level/trend stationarity** vs $H_1$: unit root.
Statistic:

$$
\mathrm{KPSS} = \frac{1}{n^2 \hat{\sigma}_u^2} \sum_{t=1}^{n} S_t^2,\quad
S_t = \sum_{i=1}^{t} \hat{u}_i,
$$

with $\hat{\sigma}_u^2$ a long-run variance estimator. Large values → reject stationarity.

**Practical rule:** Use ADF **and** KPSS together:

* ADF reject & KPSS not reject → stationary.
* ADF not reject & KPSS reject → non-stationary, difference.
* Mixed/inconclusive → inspect plots, ACF, and domain context.

---

## 2) Correlation Structure: ACF & PACF

### 2.1 Definitions

Autocovariance: $\gamma(h) = \mathrm{Cov}(x_t, x_{t+h})$.
Autocorrelation: $\rho(h) = \gamma(h)/\gamma(0)$.

Sample estimators for series $\{x_t\}_{t=1}^n$ with mean $\bar{x}$:

$$
\hat{\gamma}(h) = \frac{1}{n}\sum_{t=1}^{n-h} (x_t - \bar{x})(x_{t+h} - \bar{x}),\quad
\hat{\rho}(h) = \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)}.
$$

**Partial ACF (PACF)** at lag $k$ is the correlation between $x_t$ and $x_{t-k}$ after linearly removing the effect of intermediate lags $1,\dots,k-1$. Computationally via Durbin–Levinson / Yule–Walker recursions.

### 2.2 Sampling variability & bands

Under white-noise null, approximate 95% bands:

$$
\hat{\rho}(h) \approx \mathcal{N}\!\left(0, \frac{1}{n}\right) \;\Rightarrow\;
\text{bands} \approx \pm \frac{1.96}{\sqrt{n}}.
$$

Use with care: data-driven selection inflates false positives; treat ACF/PACF as **guides**, not proofs.

---

## 3) Model Adequacy: Ljung–Box

The Ljung–Box statistic aggregates sample autocorrelations up to lag $m$:

$$
Q(m) = n(n+2)\sum_{h=1}^{m} \frac{\hat{\rho}(h)^2}{n-h}.
$$

Under $H_0$ (no autocorrelation), $Q(m)$ is approximately $\chi^2$ with degrees of freedom $m - q$ if $q$ MA parameters were estimated (practical df adjustments vary).

* **On residuals:** Checks leftover autocorrelation → misspecification if rejected.
* **On squared residuals:** Checks short-range conditional heteroskedasticity (ARCH-like effects).

**AutoArimaExplorer guardrail:** A candidate must pass LB on residuals (and optionally on residuals²) at several lags to be considered adequate.

---

## 4) Information Criteria: AIC, AICc, BIC

Let $\hat{\ell}$ be the maximized log-likelihood and $k$ the number of estimated parameters (including variance).

* **AIC:** $\mathrm{AIC} = -2\hat{\ell} + 2k$
* **AICc (small-sample correction):**

$$
\mathrm{AICc} = \mathrm{AIC} + \frac{2k(k+1)}{n-k-1}
$$

* **BIC:** $\mathrm{BIC} = -2\hat{\ell} + k\log n$

**Guidance:**

* AIC/AICc often favor better **predictive** models; AICc when $n/k$ is small.
* BIC is more **parsimonious**; consistent for true model order under ideal conditions.
* We typically use **BIC** to control complexity and prevent overfitting, with a **drop threshold** (e.g., $\Delta \mathrm{BIC} \ge 2$) to justify extra parameters.

---

## 5) Estimation: MLE & State-Space Likelihood

### 5.1 ARIMA likelihood (invertible representation)

ARIMA($p,d,q$) can be written for the differenced series as an ARMA($p,q$) with i.i.d. innovations:

$$
\phi(B)\, y_t = \theta(B)\, \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0,\sigma^2),
$$

where $y_t = \nabla^d x_t$. Under Gaussianity, the log-likelihood follows from the joint normal density of residuals.

### 5.2 State-space & Kalman filter

ARIMA admits a linear Gaussian state-space form:

$$
\begin{aligned}
\text{State: } & \alpha_{t+1} = T \alpha_t + R \eta_t, \\
\text{Obs: }   & y_t = Z \alpha_t + \varepsilon_t,
\end{aligned}
$$

with $\eta_t,\varepsilon_t$ Gaussian. The Kalman filter/smoother yields the exact likelihood (up to initialization choices). This is robust and handles missing data naturally.

### 5.3 Constraints

* **Stationarity:** all roots of $\phi(z)=0$ outside unit circle.
* **Invertibility:** all roots of $\theta(z)=0$ outside unit circle.

Implementations enforce this via constrained optimization or parameter transforms (e.g., hyperbolic tangent reparameterization). AutoArimaExplorer lets you choose strict enforcement or relaxed attempts when the optimizer struggles.

---

## 6) Forecasts & Uncertainty

### 6.1 Point forecasts

For ARIMA($p,d,q$), $h$-step forecasts are linear projections based on the ARMA representation of $y_t=\nabla^d x_t$, then **integrated back** to the $x_t$ scale.

### 6.2 Forecast variance & intervals

For an ARMA($p,q$) with innovation variance $\sigma^2$, the $h$-step forecast error variance is

$$
\mathrm{Var}(\hat{y}_{t+h|t} - y_{t+h}) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2,
$$

where $\{\psi_j\}$ are the MA($\infty$) coefficients (impulse response). Intervals (Gaussian):

$$
\hat{y}_{t+h|t} \pm z_{1-\alpha/2}\,\sigma_h.
$$

After integrating to the $x$-scale, intervals widen appropriately.

### 6.3 Back-transform bias

If you model $\log x_t$, then

$$
\mathbb{E}[x_{t+h}| \mathcal{F}_t] \approx \exp\!\big(\mu_h + \tfrac{1}{2}\sigma_h^2\big),
$$

not $\exp(\mu_h)$. Use bias-correction when reporting mean forecasts on the original scale.

---

## 7) Error Metrics & Interval Quality

Let $y_i$ be actuals and $\hat{y}_i$ forecasts.

* **RMSE:** $\sqrt{\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2}$ (penalizes large errors)
* **MAE:** $\frac{1}{N}\sum_i |y_i - \hat{y}_i|$ (robust to outliers)
* **MAPE:** $\frac{100}{N}\sum_i \left|\frac{y_i - \hat{y}_i}{y_i}\right|$ (undefined at zeros; be careful)

**Prediction interval quality:**
For a nominal $(1-\alpha)$ interval $[L_i, U_i]$, the **Winkler score**:

$$
W_\alpha(y_i, L_i, U_i) =
\begin{cases}
U_i - L_i, & L_i \le y_i \le U_i,\\
(U_i - L_i) + \frac{2}{\alpha}(L_i - y_i), & y_i < L_i,\\
(U_i - L_i) + \frac{2}{\alpha}(y_i - U_i), & y_i > U_i.
\end{cases}
$$

Lower is better; it balances **narrowness** and **coverage**. AutoArimaExplorer supports Winkler-based OOS selection.

---

## 8) Transformations & Scaling

* **Log / Box–Cox:** stabilize variance, linearize growth. Box–Cox:

$$
x^{(\lambda)} =
\begin{cases}
\dfrac{x^\lambda - 1}{\lambda}, & \lambda \ne 0,\\[6pt]
\log x, & \lambda=0.
\end{cases}
$$

* **Standardization:** $(x - \mu)/\sigma$ improves optimizer stability; predictions are back-scaled.
* **Min–max \[0,1]:** helpful for comparability in OOS scoring across different $d$ choices (used optionally).

**Note:** AIC/BIC are invariant to scale changes that reparameterize $\sigma^2$ appropriately, but diagnostics like MAPE are **not**; be consistent when comparing models.

---

## 9) Model Selection Trade-offs

* **Parsimony vs fit:** More parameters always improve in-sample likelihood; information criteria penalize this.
* **Bias–variance:** Simple models may be biased but generalize better; complex models risk variance/instability.
* **Tie-breaking:** If BIC is effectively tied, prefer the model with smaller $p+q$; then smaller $p$, then $q$. AutoArimaExplorer uses this hierarchy.
* **Out-of-sample supremacy:** When diagnostics and BIC disagree with OOS performance, trust **OOS** (with enough folds).

---

## 10) Volatility, Misspecification & Breaks

* **Conditional heteroskedasticity (ARCH/GARCH):** If LB on squared residuals rejects, ARIMA’s intervals will be miscalibrated. Consider variance-stabilizing transforms, or model volatility explicitly (e.g., ARIMA + GARCH on residuals).
* **Structural breaks / regime changes:** ARIMA assumes time-invariant parameters. Breaks induce spurious autocorrelation and poor forecasts. Detect via change-point tests or rolling fits.
* **Seasonality:** Strong seasonal autocorrelation calls for seasonal differencing and SARIMA (or pre-decomposition like STL).
* **Exogenous effects (ARIMAX):** Shocks, promotions, holidays, weather, etc., create predictable structure not captured by pure ARIMA. Consider regressors.

---

## 11) How AutoArimaExplorer uses these concepts

* **Stationarity handling:** Tries $d=0,1,\dots$ (you control the grid). If $d=0$, works on log-returns; if $d\ge1$, works on standardized levels.
* **Estimation:** Robust MLE via state-space Kalman likelihood (and innovations MLE for $d=0$), with retries under different trend/constraint setups.
* **Diagnostics:** **Mandatory** convergence; **Ljung–Box guardrail** on residuals (and optional on squared residuals).
* **Selection:**

  * **In-sample:** **BIC** minimization with a **drop threshold** (e.g., $\Delta \mathrm{BIC}\ge2$).
  * **By-q:** best $p$ per $(d,q)$, then compare across $(d,q)$.
  * **Out-of-sample (rolling origin):** RMSE/MAE/MAPE or **Winkler** for intervals; the OOS winner can override purely in-sample picks.

---

## 12) References (suggested)

* Box, Jenkins, Reinsel & Ljung (2015): *Time Series Analysis: Forecasting and Control*.
* Brockwell & Davis (2016): *Introduction to Time Series and Forecasting*.
* Hyndman & Athanasopoulos (2021): *Forecasting: Principles and Practice*.
* Hamilton (1994): *Time Series Analysis*.
* Ljung & Box (1978): *On a Measure of Lack of Fit in Time Series Models*.
* Akaike (1974); Schwarz (1978): Original AIC/BIC papers.
