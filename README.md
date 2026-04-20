---

## Part-by-Part Summary

### Part 1 — Diagnose: Three Errors in an ARIMA Pipeline

Three deliberate bugs were embedded across three cells in an ARIMA pipeline applied to FRED CPI (`CPIAUCNS`). Each represents a distinct and common modeling mistake.

**Bug 1 — Stationarity violation:**  
`ARIMA(2,0,1)` was fitted with `d=0` on raw CPI, which the ADF test confirmed is non-stationary (unit root). ARMA models require a stationary input — applying them to levels produces a spurious, unreliable model.

**Bug 2 — Seasonality omission:**  
Even after fixing `d=1`, plain `ARIMA(2,1,1)` was used with no seasonal terms. Monthly CPI has well-documented seasonal patterns (holiday spending, energy costs). The residual ACF showed significant spikes at lags 12 and 24, confirming seasonal autocorrelation was leaking into the residuals.

**Bug 3 — Missing residual diagnostic:**  
The Ljung-Box test was never run before producing forecasts. If residuals are autocorrelated, the model is misspecified and the 95% confidence intervals will be too narrow and unreliable. A correct pipeline always verifies residual whiteness before trusting the forecast.

---

### Part 2 — Fix: Correct SARIMA Pipeline

All three bugs were corrected in a single clean pipeline:

**Step 1 — Stationarity:** ADF confirmed `diff(CPI)` is stationary (p < 0.05), justifying `d=1`.

**Step 2 — Seasonal model selection:** `pm.auto_arima` with `seasonal=True, m=12` selected the best SARIMA order. The model was refit using `statsmodels SARIMAX` for full diagnostics.

**Step 3 — Ljung-Box diagnostic:** `acorr_ljungbox` was run at lags 12 and 24. All p-values > 0.05 confirmed residuals are white noise — the model fully absorbed the seasonal structure.

**Step 4 — Forecast:** 24-month CPI forecast with 95% confidence intervals, produced only after passing the diagnostic gate.

```python
# Key fix: use SARIMA with seasonal differencing
sarima_model = SARIMAX(cpi, order=(p, 1, q), seasonal_order=(P, 1, Q, 12))
sarima_result = sarima_model.fit(disp=False)

# Diagnostic gate — must pass before forecasting
lb = acorr_ljungbox(sarima_result.resid, lags=[12, 24], return_df=True)
assert (lb['lb_pvalue'] > 0.05).all(), "Residuals are not white noise"
```

**Verification:**
- ADF p-value on `diff(CPI)` < 0.05 ✅
- Ljung-Box p-values at lags 12 and 24 > 0.05 ✅
- Residual ACF at lag 12 < 0.1 ✅

---

### Part 3 — Extend: GARCH(1,1) on S&P 500

ARIMA models the **conditional mean** of a series. Financial returns exhibit **volatility clustering** — large moves (positive or negative) tend to follow large moves. The GARCH(1,1) model captures this by modeling the **conditional variance** as a function of past squared residuals and past variance:

$$\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

S&P 500 daily log returns from 2000–2024 were pulled via `yfinance`. The key constraint is:

$$\alpha_1 + \beta_1 < 1$$

which ensures the variance process is stationary (shocks decay rather than explode).

```python
garch_spec = arch_model(returns, mean='Constant', vol='Garch', p=1, q=1, dist='normal')
garch_result = garch_spec.fit(disp='off')
```

**Key findings:**
- `alpha[1] + beta[1]` is close to 1, indicating **high volatility persistence** — shocks to volatility decay slowly.
- The half-life formula `log(2) / -log(alpha + beta)` quantifies how many days it takes for a volatility shock to decay by half.
- Conditional volatility spikes are clearly visible around **Sep 11 (2001), Lehman Brothers (2008), COVID (2020), and the 2022 Bear Market**.

| Parameter | Interpretation |
|---|---|
| `omega` | Long-run baseline variance |
| `alpha[1]` | Sensitivity to recent shocks (ARCH term) |
| `beta[1]` | Persistence of past variance (GARCH term) |
| `alpha + beta` | Total volatility persistence (must be < 1) |

---

### Part 4 — Production Module: `forecast_evaluation.py`

A reusable, portfolio-grade Python module with full docstrings, type hints, and error handling.

| Function | Description |
|---|---|
| `compute_mase(actual, forecast, insample, m)` | Mean Absolute Scaled Error — scales MAE against a naive seasonal benchmark. MASE < 1 means the model beats the naive forecast. |
| `backtest_expanding_window(series, model_fn, min_train, horizon, step)` | Expanding-window backtest — fits the model at each origin, records errors and MASE per horizon step. |

```python
from forecast_evaluation import compute_mase, backtest_expanding_window

# Score a forecast
mase = compute_mase(actual, forecast, insample, m=12)

# Run a full expanding-window backtest
results = backtest_expanding_window(cpi, model_fn=sarima_model, min_train=120, horizon=12)
```

---

### Challenge — Block Bootstrap Forecast Intervals

Standard ARIMA confidence intervals assume normally distributed, i.i.d. residuals. When residuals exhibit volatility clustering or heavy tails, these intervals can be **too narrow**. The **moving block bootstrap** produces distribution-free forecast intervals by resampling overlapping blocks of residuals — preserving their autocorrelation and heteroskedasticity structure.

**Algorithm:**
1. Fit SARIMA, extract residuals and point forecasts
2. For each bootstrap iteration, resample overlapping residual blocks of length `block_size`
3. Add resampled residuals to the point forecast to simulate a future path
4. Compute 2.5th and 97.5th percentiles across all paths for the 95% interval

```python
boot_ci = block_bootstrap_forecast(sarima_result, horizon=24, n_bootstrap=500, block_size=6)
```

**Why block bootstrap?** Macro residuals are serially correlated. i.i.d. bootstrap shuffles all observations independently, destroying that structure and underestimating uncertainty. Sampling contiguous blocks preserves within-block autocorrelation.

**Key property:** The confidence band widens as the horizon increases, correctly reflecting growing forecast uncertainty over time.

---

