# heavy_model_estimation.py
"""
HEAVY (1, 1) — one‑file helper
==============================
Robust workflow to estimate Shephard & Sheppard (2010) HEAVY(1,1) models on a **single equity
or index** selected by **--ticker**.

Features
--------
* CSV formats accepted
  * *Vertical* (one ticker per file) — standard.
  * *Panel*  (many tickers per file) — pass `--ticker TICKER`; the loader finds a column among
    `SYM_ROOT`, `symbol`, `Ticker`, `ticker`, `SYMBOL` and keeps only those rows.
* RV source
  * same file via `--rv-col` **or**
  * separate wide panel via `--rv / --rv-ticker`.
* **Auto‑scaling** so mean(r²) ≈ mean(RV) (override with `--rv-scale` / `--ret-scale`).
* Diagnostics plot: τₜ vs RV, hₜ vs r², plus r² bars.

Example
-------
```bash
python heavy_model_estimation.py \
  --prices prices_panel.csv --date-col DATE --price-col CPrc --rv-col ivol_t \
  --ticker AAPL --auto-scale --plot aapl_diag.png
```
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True})

###############################################################################
# CSV HELPERS
###############################################################################

def _clean_numeric(s: pd.Series) -> pd.Series:
    """Strip thousands separators so pandas can coerce to float."""
    return s.astype(str).str.replace(r",(?=\d{3}(?:\.\d+)?$)", "", regex=True)


def _filter_by_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if ticker is None:
        return df
    possible_cols = [c for c in df.columns if c.lower() in {"symbol"}]
    if not possible_cols:
        return df  # assume file already single‑ticker
    col = possible_cols[0]
    return df[df[col].astype(str).str.upper() == ticker.upper()]


def _read_price_csv(path: str | Path, *, date_col: str, price_col: str, ticker: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = _filter_by_ticker(df, ticker)
    df[price_col] = pd.to_numeric(_clean_numeric(df[price_col]), errors="raise")
    for c in df.columns.difference([date_col]):
        df[c] = pd.to_numeric(_clean_numeric(df[c]), errors="ignore")
    df = df.set_index(date_col).sort_index()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def load_price_series(path: str | Path, *, date_col="Date", price_col="Price", ticker: Optional[str] = None) -> pd.Series:
    return _read_price_csv(path, date_col=date_col, price_col=price_col, ticker=ticker)[price_col]


def load_price_and_rv_series(path: str | Path, *, date_col, price_col, rv_col, ticker: Optional[str]) -> Tuple[pd.Series, pd.Series]:
    df = _read_price_csv(path, date_col=date_col, price_col=price_col, ticker=ticker)
    if rv_col not in df.columns:
        raise ValueError(f"RV column '{rv_col}' not found; available={list(df.columns)[:10]}")
    return df[price_col], df[rv_col].astype(float)


def load_rv_panel(path: str | Path, ticker: str) -> pd.Series:
    with open(path) as fh:
        first = fh.readline()
    has_date = bool(re.match(r"(?i)date", first.split(",")[0]))
    if has_date:
        df = pd.read_csv(path, parse_dates=[0]).set_index("Date")
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    else:
        df = pd.read_csv(path, index_col=0)
    if ticker not in df.columns:
        raise ValueError(f"Ticker '{ticker}' not in RV panel")
    return df[ticker].astype(float)

###############################################################################
# ALIGN SERIES
###############################################################################

def _align(a: pd.Series, b: pd.Series, by_order: bool):
    if by_order or not (isinstance(a.index, pd.DatetimeIndex) and isinstance(b.index, pd.DatetimeIndex)):
        m = min(len(a), len(b))
        return a.iloc[-m:], b.iloc[-m:]
    joined = a.to_frame("a").join(b.to_frame("b"), how="inner")
    return joined["a"], joined["b"]

###############################################################################
# NEGATIVE LOG‑LIKELIHOODS
###############################################################################

def _nll_h(params, rv, ret):
    ω, α, β = params
    if ω <= 0 or α < 0 or β < 0 or α + β >= 1:
        return 1e10
    h = np.empty_like(ret)
    h[0] =(ret ** 2).mean()
    for t in range(1, len(ret)):
        h[t] = ω + α * rv[t - 1] + β * h[t - 1]
    return 0.5 * np.sum(np.log(h) + ret**2 / h)


def _nll_tau(params, rv):
    ω, α, β = params
    if ω <= 0 or α < 0 or β < 0 or α + β >= 1:
        return 1e10
    τ = np.empty_like(rv)
    τ[0] = rv.mean()
    for t in range(1, len(rv)):
        τ[t] = ω + α * rv[t - 1] + β * τ[t - 1]
    return 0.5 * np.sum(np.log(τ) + rv / τ)

###############################################################################
# MODEL CLASS
###############################################################################

@dataclass
class HeavyParams:
    ω: float; α: float; β: float; ω_R: float; α_R: float; β_R: float
    def as_vector(self):
        return np.array([self.ω, self.α, self.β, self.ω_R, self.α_R, self.β_R])


class HeavyModel:
    def __init__(self, prices: pd.Series, rv: pd.Series, *, 
                 by_order=False, ret_scale: float | None = None,
                 rv_scale: float | None = None, auto_scale=False):
        
        ret = np.log(prices).diff().dropna()
        ret, rv = _align(ret, rv, by_order)

        self.rv_scale_factor = 1.0  # <--- NEW: Default

        if ret_scale:
            ret *= ret_scale
        if rv_scale:
            rv *= rv_scale
            self.rv_scale_factor = rv_scale  # <--- SAVE the manual rv_scale

        ratio = (ret**2).mean() / rv.mean()
        if auto_scale or ratio > 1e2 or ratio < 1e-2:
            factor = (ret**2).mean() / rv.mean()
            rv *= factor
            self.rv_scale_factor = factor  # <--- SAVE the auto-scale factor
            print(f"[auto-scale] multiplied RV by {factor:.2e}")

        self.returns, self.rv = ret, rv
        self.params: HeavyParams | None = None


    # -------------------------- fit
    def fit(self, verbose=True):
        ret_arr, rv_arr = self.returns.to_numpy(), self.rv.to_numpy()
        res_h = minimize(_nll_h, [1e-6, 0.1, 0.1], args=(rv_arr, ret_arr),
                         bounds=Bounds([1e-12, 0, 0], [np.inf, 1, 1]), method="L-BFGS-B")
        res_tau = minimize(_nll_tau, [1e-6, 0.1, 0.3], args=(rv_arr,),
                           bounds=Bounds([1e-12, 0, 0], [np.inf, 1, 1]), method="L-BFGS-B")
        if not (res_h.success and res_tau.success):
            warnings.warn("Optimization may not have converged")
        self.params = HeavyParams(*res_h.x, *res_tau.x)
        if verbose:
            print(self.params)
        return self.params

    # -------------------------- filtering
    def _filter(self):
        if self.params is None:
            raise RuntimeError("Call fit() first")
        ω, α, β, ω_R, α_R, β_R = self.params.as_vector()
        ret, rv = self.returns.to_numpy(), self.rv.to_numpy()
        h = np.empty_like(ret); τ = np.empty_like(rv)
        h[0] = ω/(1 - β); τ[0] = ω_R/(1 - β_R)
        for t in range(1, len(ret)):
            h[t] = ω + α * rv[t - 1] + β * h[t - 1]
            τ[t] = ω_R + α_R * rv[t - 1] + β_R * τ[t - 1]
        return pd.DataFrame({"h": h, "tau": τ}, index=self.returns.index)

    # -------------------------- plot
    def plot(self, save: str | None = None):
        s = self._filter()
        fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 3, 1]})
        # τ vs RV
        ax[0].plot(s.index, s["tau"], label="τₜ (pred RV)", color="green")
        ax[0].plot(self.rv.index, self.rv, label="RV (actual)", color="orange", alpha=0.7)
        ax[0].legend(); ax[0].set_ylabel("Variance")

        ax[1].plot(s.index, s["h"], label="hₜ (pred var(r²))", color="blue")
        ax[1].plot(self.returns.index, self.rv, label="RV (actual)", color="grey", alpha=0.6)
        ax[1].legend(); ax[1].set_ylabel("Variance")

        ax[2].bar(self.returns.index, self.returns**2, color="grey", alpha=0.4, width=1.0)
        ax[2].set_ylabel("r²"); ax[2].set_xlabel("Date")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=300)
            print(f"Saved → {save}")
        return fig

###############################################################################
# CLI
###############################################################################

def _cli():
    import argparse
    p = argparse.ArgumentParser("HEAVY(1,1) quick run")
    p.add_argument("--prices", required=True)
    p.add_argument("--date-col", default="Date")
    p.add_argument("--price-col", default="Price")
    p.add_argument("--rv-col")
    p.add_argument("--rv")
    p.add_argument("--rv-ticker")
    p.add_argument("--by-order", action="store_true")
    p.add_argument("--ret-scale", type=float)
    p.add_argument("--rv-scale", type=float)
    p.add_argument("--auto-scale", action="store_true")
    p.add_argument("--plot")
    p.add_argument("--ticker")
    args = p.parse_args()

    if args.rv_col:
        prices, rv = load_price_and_rv_series(
            args.prices, date_col=args.date_col,
            price_col=args.price_col, rv_col=args.rv_col, ticker = args.ticker
        )
    else:
        if not (args.rv and args.rv_ticker):
            raise SystemExit("Either supply --rv-col or both --rv and --rv-ticker")
        prices = load_price_series(args.prices, date_col=args.date_col, price_col=args.price_col)
        rv = load_rv_panel(args.rv, args.rv_ticker)

    mdl = HeavyModel(
        prices, rv ** 2, by_order=args.by_order,
        ret_scale=args.ret_scale, rv_scale=args.rv_scale,
        auto_scale=args.auto_scale,
    )
    mdl.fit(verbose=True)
    mdl.plot(save=args.plot)

if __name__ == "__main__":
    _cli()
