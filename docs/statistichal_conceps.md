awesome—here’s a deep, math-forward chapter of the **statistical and mathematical concepts** you need to fully understand the tool we built. It’s written as a standalone “theory” section you can drop into your docs.

---

# Statistical & Mathematical Concepts for ARIMA Modeling

> This chapter covers the core probability/stats background behind ARIMA-type models and the diagnostics/criteria we use in the tool. It includes definitions, assumptions, identification conditions, key tests, estimation methods, and forecasting math.

---

## 1) Time Series Basics

### 1.1 Stochastic Processes and Indexing

A (univariate) time series is a sequence $\{Y_t\}_{t\in\mathbb{Z}}$ of random variables indexed by discrete time $t$. We observe a single realization $y_1,\dots,y_T$.

The **backshift/lag operator** $B$ is defined by $B Y_t := Y_{t-1}$. Powers: $B^k Y_t = Y_{t-k}$.

### 1.2 Moments, Autocovariance, and ACF

* Mean: $\mu_t = \mathbb{E}[Y_t]$
* Autocovariance at lag $h$: $\gamma_Y(h) = \mathrm{Cov}(Y_t, Y_{t-h})$
* Autocorrelation function (ACF): $\rho_Y(h) = \gamma_Y(h)/\gamma_Y(0)$

### 1.3 Stationarity

* **Strict stationarity:** the joint distribution of $(Y_{t_1},\dots,Y_{t_k})$ is invariant to time shifts.
* **Weak (covariance) stationarity:**

  $$
  \mathbb{E}[Y_t]=\mu\ \text{(constant)},\quad \mathrm{Var}(Y_t)=\sigma^2\ \text{(finite, constant)},\quad \gamma_Y(h)\ \text{depends only on}\ h.
  $$

  ARMA modeling assumes weak stationarity (after appropriate transformations/differencing).

### 1.4 Wold Decomposition

Every zero-mean, weakly stationary process admits the **Wold representation**:

$$
Y_t = \sum_{j=0}^{\infty}\psi_j \varepsilon_{t-j},\quad \sum_{j=0}^\infty \psi_j^2<\infty,
$$

with $\{\varepsilon_t\}$ a white noise (WN) process: $\varepsilon_t\overset{iid}{\sim}(0,\sigma_\varepsilon^2)$. ARMA models are finite-order parametric approximations of this representation.

---

## 2) AR, MA, ARMA Models

### 2.1 AR(p)

$$
\phi(B)Y_t = \varepsilon_t,\quad \phi(B)=1-\phi_1 B - \dots - \phi_p B^p.
$$

**Stationarity condition:** all roots of $\phi(z)=0$ lie **outside** the unit circle ($|z|>1$).

### 2.2 MA(q)

$$
Y_t = \theta(B)\varepsilon_t,\quad \theta(B)=1+\theta_1 B+\dots+\theta_q B^q.
$$

**Invertibility condition:** all roots of $\theta(z)=0$ lie outside the unit circle (ensures a unique Wold representation).

### 2.3 ARMA(p,q)

$$
\phi(B)Y_t = \theta(B)\varepsilon_t.
$$

Combine the AR stationarity and MA invertibility conditions. The ACF/PACF patterns are classical identification heuristics:

* AR(p): PACF cuts off at lag $p$, ACF tails off.
* MA(q): ACF cuts off at lag $q$, PACF tails off.
* ARMA(p,q): both tail off.

---

## 3) ARIMA (Integrated ARMA)

### 3.1 Differencing

To remove low-frequency nonstationarity, difference $d$ times:

$$
\nabla^d Y_t := (1-B)^d Y_t = \sum_{k=0}^d \binom{d}{k}(-1)^k Y_{t-k}.
$$

Then model $X_t := \nabla^d Y_t$ as ARMA(p,q):

$$
\phi(B) X_t = \theta(B)\varepsilon_t \quad \Longleftrightarrow \quad \phi(B)(1-B)^d Y_t = \theta(B)\varepsilon_t.
$$

### 3.2 Drift/Trend Interpretation

Including a constant in the model for $X_t=\nabla^d Y_t$ corresponds to a polynomial trend of degree $d-1$ in levels $Y_t$ (e.g., with $d=1$, a constant ⇒ linear trend in $Y_t$).

---

## 4) Identification & Pre-Testing

### 4.1 Unit Root Tests (d)

* **ADF (Augmented Dickey–Fuller) test:**
  Tests $H_0:$ unit root (non-stationary) vs $H_1:$ stationary. Regression (with optional constant/trend):

  $$
  \Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^k \delta_i \Delta Y_{t-i} + u_t.
  $$

  Rejecting $H_0:\gamma=0$ suggests stationarity (no unit root).

* **KPSS test:**
  Tests $H_0:$ stationary around (possibly) deterministic trend vs $H_1:$ unit root. Complementary to ADF.

* **PP (Phillips–Perron):**
  Non-parametric correction for serial correlation and heteroskedasticity.

*Practice:* choose the smallest $d$ that renders the series “approximately stationary” (over-differencing inflates noise and induces spurious MA dynamics).

### 4.2 Transformations for Variance Stabilization

* **Log transform:** $X_t=\log(Y_t + c)$ for $Y_t> -c$.
* **Box–Cox:** $X_t=\frac{Y_t^\lambda-1}{\lambda}$ for $\lambda\neq 0$ (and $ \log Y_t$ if $\lambda=0$).

---

## 5) Estimation

### 5.1 Likelihood (Exact/Conditional/State-Space)

Let $X_t=\nabla^d Y_t$.

* **Conditional Sum of Squares (CSS):** minimize $\sum_{t=m+1}^T \hat\varepsilon_t^2(\theta)$ given initial conditions. Fast but approximate.
* **Exact MLE:** under Gaussian errors, maximize the Gaussian likelihood using the exact ARIMA covariance; slower but efficient asymptotically.
* **State-space MLE (Kalman filter):** represent ARIMA as a linear Gaussian state-space model; evaluate likelihood with Kalman recursions; robust and flexible (works with missing values, time-varying settings).
* **Innovations MLE:** evaluates the likelihood via innovations algorithm using the implied MA($\infty$) representation.

**Numerics.** Quasi-Newton optimizers like **L-BFGS** are common. Near-unit roots, high orders, and AR/MA near-cancellation can cause ill-conditioning and non-convergence. Enforcing stationarity/invertibility during optimization stabilizes estimation.

---

## 6) Model Selection Criteria

Let $k$ be the number of estimated parameters (including variance), $\hat L$ the maximized likelihood.

* **AIC:** $\mathrm{AIC} = -2\log \hat L + 2k$
* **AICc:** small-sample corrected AIC

  $$
  \mathrm{AICc} = \mathrm{AIC} + \frac{2k(k+1)}{T - k - 1}.
  $$
* **BIC (Schwarz):** $\mathrm{BIC} = -2\log \hat L + k\log T$

Lower is better. **BIC** penalizes complexity more strongly; it tends to pick more parsimonious models (our default selector). A useful rule of thumb: $\Delta \mathrm{BIC} \gtrsim 2$ indicates meaningful improvement.

---

## 7) Diagnostics: Residual Whiteness & Heteroskedasticity

### 7.1 Portmanteau Tests (Ljung–Box)

After fitting, residuals $\hat\varepsilon_t$ should be serially uncorrelated. Define residual autocorrelations $\hat\rho(h)$ for $h=1,\dots,m$.

* **Ljung–Box statistic:**

  $$
  Q(m) = T(T+2)\sum_{h=1}^m \frac{\hat\rho(h)^2}{T-h}.
  $$

Under the null of no autocorrelation up to lag $m$, $Q(m)$ is approximately $\chi^2_{m-k}$ (degrees of freedom adjusted for estimated parameters $k$). Large $Q$ ⇒ small $p$-value ⇒ residual autocorrelation remains.

We also apply Ljung–Box to **squared residuals** $\hat\varepsilon_t^2$ to screen for **conditional heteroskedasticity** (ARCH effects).

### 7.2 Engle’s ARCH LM Test

Regress $\hat\varepsilon_t^2$ on its own lags:

$$
\hat\varepsilon_t^2 = \alpha_0 + \sum_{i=1}^m \alpha_i \hat\varepsilon_{t-i}^2 + u_t.
$$

Test $H_0: \alpha_1=\dots=\alpha_m=0$. A significant test suggests time-varying variance; consider ARIMA+GARCH.

---

## 8) Forecasting Mathematics

### 8.1 h-step Forecasts

For a stationary ARMA on $X_t$ with Wold coefficients $\{\psi_j\}$:

$$
X_{t+h|t} = \mathbb{E}[X_{t+h}\mid \mathcal{F}_t] = \sum_{j=h}^{\infty}\psi_j \varepsilon_{t+h-j}.
$$

Practically, AR recursions + estimated residuals yield forecasts. For ARIMA, forecasts are made on $X_t=\nabla^d Y_t$ and then **integrated back** by summation to obtain $Y_{t+h|t}$.

### 8.2 Forecast Error Variance & Intervals

Let $\sigma_\varepsilon^2$ be innovation variance and $\psi_j$ the MA($\infty$) coefficients of the fitted ARMA for $X_t$. The $h$-step forecast error variance for $X_t$ is

$$
\mathrm{Var}\big(X_{t+h}-X_{t+h|t}\big) = \sigma_\varepsilon^2 \sum_{j=0}^{h-1}\psi_j^2.
$$

For $Y_t$ (when $d>0$), the variance accumulates due to integration.

Assuming Gaussian errors, **(1-$\alpha$) prediction intervals** for $X_{t+h}$ are:

$$
\hat X_{t+h|t} \pm z_{1-\alpha/2}\,\hat\sigma_\varepsilon \sqrt{\sum_{j=0}^{h-1}\hat\psi_j^2}.
$$

Intervals for $Y_{t+h}$ follow by integration. In practice, software computes these via state-space or equivalent formulas.

### 8.3 Winkler Score (Interval Quality)

For a (1-$\alpha$) PI $[L,U]$ and realized value $y$:

$$
\text{Winkler}_\alpha = (U-L) + \frac{2}{\alpha}\,(L-y)\,\mathbf{1}\{y<L\} + \frac{2}{\alpha}\,(y-U)\,\mathbf{1}\{y>U\}.
$$

Lower is better; it rewards narrow intervals with correct coverage and penalizes misses.

---

## 9) Time-Series Cross-Validation (Rolling Origin)

Given a series $y_1,\dots,y_T$, horizon $h$, and initial training size $n_0$:

For folds $t=n_0, n_0+\text{step}, \dots, T-h$:

1. Fit on $y_{1:t}$.
2. Forecast $h$ steps: $\hat y_{t+1|t},\dots,\hat y_{t+h|t}$.
3. Compute error(s) vs $y_{t+1},\dots,y_{t+h}$.

Aggregate per model:

* **RMSE:** $\sqrt{\frac{1}{N}\sum (y-\hat y)^2}$
* **MAE:** $\frac{1}{N}\sum |y-\hat y|$
* **MAPE:** $\frac{100}{N}\sum \left|\frac{y-\hat y}{y}\right|$ (use epsilon for small $y$)
* **Winkler score:** average across horizons/folds.

This procedure respects time order and uses only information available at the forecast origin.

---

## 10) Numerical & Identifiability Issues

### 10.1 Stationarity/Invertibility Constraints

* Enforcing roots outside the unit circle stabilizes estimation and ensures a valid infinite-order representation.
* Relaxing constraints can improve local fit but risks degenerate/near-non-invertible solutions and poor forecast behavior.

### 10.2 Near Cancellations

AR and MA polynomials can nearly cancel (e.g., $\phi(B)\approx \theta(B)$), creating **flat likelihood surfaces** and unstable parameters.

### 10.3 Optimization Warnings

* *Non-stationary starting AR parameters*: optimizer resets to zeros or constrained region.
* *Non-invertible starting MA parameters*: same idea on the MA side.
* *Failed convergence*: increase iterations, switch optimizer, simplify model, or re-scale.

---

## 11) Residual Analysis in Practice

* **Whiteness in mean:** residual ACF within bands; Ljung–Box $p$-values “not small” at chosen lags.
* **Variance dynamics:** ACF of squared residuals; Ljung–Box on $\hat\varepsilon_t^2$; ARCH LM test.
* **Distributional shape:** heavy tails/asymmetry → consider robust intervals, bootstrap, or volatility modeling.

Passing mean-whiteness **does not** imply calibrated uncertainty if variance is time-varying.

---

## 12) Scaling and Transformations Used in the Tool

* **Min–Max scaling** (for OOS comparability across different $d$ choices):

  $$
  x'_t = \frac{x_t - \min(x)}{\max(x) - \min(x) + \varepsilon}.
  $$
* **Log-returns for $d=0$ ARMA** (financial data):
  Given levels $P_t$, we work with $r_t = \log(P_t+c) - \log(P_{t-1}+c)$ after ensuring positivity ($c>0$ if needed) and optionally standardize to unit scale.

---

## 13) Why BIC + Ljung–Box Guardrails?

* **BIC:** favors parsimonious models with genuine likelihood gains; reduces overfitting.
* **Ljung–Box guardrail:** ensures selected models actually remove linear dependence in residuals (our minimum standard for “adequate” mean dynamics).
* Optional squared-residual check adds a quick screen for short-run heteroskedasticity (flagging when ARIMA alone is insufficient).

---

## 14) Summary of Notation

* $Y_t$: original series (level)
* $X_t=\nabla^d Y_t$: differenced series
* $B$: backshift operator; $BY_t=Y_{t-1}$
* $\phi(B)$: AR polynomial; $\theta(B)$: MA polynomial
* $\varepsilon_t$: innovations (white noise)
* $\psi_j$: Wold coefficients (MA$(\infty)$)
* $\rho(h)$: autocorrelation at lag $h$
* $k$: number of parameters
* AIC, BIC: information criteria
* $Q(m)$: Ljung–Box statistic at lag $m$

---

## 15) Where Each Concept Lives in the Tool

* **Stationarity & differencing (Section 3)** → choice of $d$; using returns for $d=0$, levels for $d\ge1$.
* **AR/MA orders & constraints (Section 2, 10)** → grid exploration and constrained optimization.
* **Likelihood and BIC (Sections 5–6)** → estimation and primary selection criterion.
* **Ljung–Box tests (Section 7)** → post-fit guardrails in selection functions.
* **OOS validation (Section 9)** → rolling origin selection route.
* **Variance checks (Sections 7, 11)** → optional squared-residual diagnostics; suggest ARIMA+GARCH if needed.

---

### Further Reading (short list)

* Box, Jenkins, Reinsel & Ljung — *Time Series Analysis: Forecasting and Control*
* Brockwell & Davis — *Introduction to Time Series and Forecasting*
* Hyndman & Athanasopoulos — *Forecasting: Principles and Practice* (free online)

---

If you want, I can turn this into a `statistical-concepts.md` file with a clean table of contents and cross-links to your “ARIMA implementation” and “Tutorials” chapters.
