# Introduction

Forecasting with time series is, at its core, a game of exploiting *memory*. Most real-world signals—sales, loads, prices, churn—do not reset to zero at midnight; what happened yesterday leaks into what happens today. ARIMA-class models are the classical way to formalize that memory: compact, interpretable and (when used where they fit) remarkably hard to beat for short-horizon forecasting.

This chapter gives you the conceptual toolkit to read, question, and effectively use our **AutoArimaExplorer**. I’ll keep the math light here and focus on intuition; the statistical details and proofs live in the *Statistical Concepts* chapter.

---

## A mental model for time series

Think of a univariate time series $x_t$ as “structure + noise” observed at regular intervals. The structure we care about is **serial dependence**—today’s value being explained by its own past and by past “shocks.” The noise, or **innovation** $\varepsilon_t$, is what’s left once we’ve explained everything systematic.

Two complementary mechanisms capture this dependence:

* **Autoregression (AR):** *feedback from the past level.* If $x_t$ tends to move back toward a long-run mean after a shock, or if it carries inertia forward, that’s AR behavior.
* **Moving average (MA):** *echoes of past shocks.* A one-off shock rarely vanishes instantly; it ripples for a few periods. MA terms model those ripples.

An **ARMA(p,q)** model combines both: the current value depends on the last $p$ levels and the last $q$ shock echoes. This is a parsimonious way to capture short-run dynamics without overfitting.

---

## From ARMA to ARIMA

ARMA assumes the series is **stationary** in mean and variance—no drifting level, no deterministic trend. Many business and economic series aren’t. The standard cure is **differencing**: model the changes rather than the levels.

An **ARIMA(p,d,q)** model simply applies $d$ differences to the original series and then fits an ARMA on that differenced series. Intuitively:

* $d=0$: the series is already “stable” → ARMA is fine.
* $d=1$: first differences (like returns) are stable → ARMA on $\Delta x_t$.
* $d\ge 2$: more aggressive detrending if changes themselves trend.

Differencing restores stationarity in mean, letting AR/MA terms describe the short-run correlation structure cleanly. The art is to difference **enough** to kill the drift but **not so much** that you destroy meaningful signal (over-differencing can inject spurious MA structure and inflate forecast variance).

---

## Where ARIMA shines (and where it doesn’t)

**Strengths**

* **Short-horizon accuracy:** Excellent for 1–10 step forecasts when dynamics are primarily short-memory.
* **Interpretability:** Coefficients map to persistence and shock decay; diagnostics map to “is anything left in the residuals?”
* **Parsimony:** Small number of parameters; robust on modest sample sizes.
* **Speed & stability:** Mature estimation routines with well-understood behavior.

**Limitations & common data issues**

* **Strong seasonality:** Plain ARIMA does not encode seasonal cycles; you’d need SARIMA (seasonal ARIMA) or exogenous regressors.
* **Time-varying volatility:** Volatility clustering (ARCH/GARCH effects) violates constant-variance assumptions. ARIMA can forecast the *mean* well while mis-calibrating uncertainty.
* **Structural breaks and regime changes:** Coefficients are fixed; abrupt changes can degrade fit.
* **Missing data / irregular sampling:** Classical estimators expect regular, clean sampling; pre-processing matters.
* **Outliers:** A few large shocks can bias estimation and diagnostics; robust cleaning helps.
* **Long memory:** Persistent autocorrelation across long lags may require specialized models.

These are not deal breakers; they’re *design constraints*. They inform how we prepare data, how strict our diagnostics should be, and when to escalate beyond plain ARIMA.

---

## Residuals, diagnostics, and model adequacy (the high-level view)

A good ARIMA fit makes residuals behave like **white noise**: no leftover autocorrelation in the mean, and—if you also want well-behaved uncertainty—no short-run dependence in squared residuals either. Two practical signals:

* **Ljung–Box tests** on residuals: detect leftover autocorrelation (missed AR/MA structure).
* **Ljung–Box on squared residuals:** quick check for ARCH-type effects (time-varying variance).

Model selection balances **fit versus complexity**. Information criteria—most commonly **BIC**—penalize extra parameters; out-of-sample rolling tests verify that gains are real, not artifacts. We detail these tools in the *Statistical Concepts* chapter; here, it’s enough to remember: *lower BIC and cleaner residuals are better, but always validate on held-out data when the stakes are high.*

---

## How AutoArimaExplorer uses these ideas

AutoArimaExplorer operationalizes the theory above into a practical workflow:

* **Two views of the series:** it uses standardized levels for $d\ge 1$ and log-returns for $d=0$. This mirrors how ARIMA is intended to be used and stabilizes estimation.
* **Robust fitting:** multiple estimation “trials” (changes in trend specification and optimizer constraints) are attempted; only finite-BIC solutions that **converge** are considered viable.
* **Guardrails by design:** optional Ljung–Box checks on residuals (and squared residuals) act as sanity filters.
* **Three complementary selectors:**

  1. **Best-by-BIC** across the full grid.
  2. **Best-by-q** (choose $p$ within each $(d,q)$ slice) for a stepwise, parsimonious path.
  3. **Best-by-OOS** via rolling-origin evaluation (RMSE/MAE/MAPE or Winkler score for PI calibration).

The result is a model that is interpretable, statistically sound by default, and validated against the kinds of data problems you actually meet in practice.

---

## What you’ll take away

After this chapter you should be able to:

* Explain, in plain language, what AR, MA and ARIMA components do and when they’re appropriate.
* Recognize the assumptions ARIMA makes (and how differencing helps meet them).
* Identify when ARIMA is the right tool—and when to reach for seasonal terms, exogenous regressors, or volatility models.
* Read AutoArimaExplorer’s selections and diagnostics with a critical eye, knowing what each signal means.

If you want the equations, proofs, and test definitions, jump to *Statistical Concepts*. If you want to see how these ideas become code and APIs, head to the *Technical Concepts* section.
