from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from arch import arch_model

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True})

###############################################################################
# CSV HELPERS
###############################################################################

def _clean_numeric(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r",(?=\d{3}(?:\.\d+)?$)", "", regex=True)

def _filter_by_ticker(df: pd.DataFrame, ticker: Optional[str]) -> pd.DataFrame:
    if ticker is None:
        return df
    ticker_cols = {"sym_root", "symbol", "ticker", "sym", "symbolroot"}
    cols = [c for c in df.columns if c.lower() in ticker_cols]
    if not cols:
        return df  # single-ticker file
    col = cols[0]
    return df[df[col].astype(str).str.upper() == ticker.upper()]

def _read_price_csv(
    path: str | Path,
    *,
    date_col: str,
    price_col: str,
    rv_col: str,
    ticker: Optional[str],
) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = _filter_by_ticker(df, ticker)
    df[price_col] = pd.to_numeric(_clean_numeric(df[price_col]), errors="raise")
    df[rv_col] = pd.to_numeric(_clean_numeric(df[rv_col]), errors="raise")
    df = df.set_index(date_col).sort_index()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df

def load_price_and_rv_series(
    path: str | Path,
    *,
    date_col: str,
    price_col: str,
    rv_col: str,
    ticker: Optional[str],
) -> Tuple[pd.Series, pd.Series]:
    df = _read_price_csv(path, date_col=date_col, price_col=price_col, rv_col=rv_col, ticker=ticker)
    return df[price_col], df[rv_col]

###############################################################################
# GARCH USING ARCH PACKAGE
###############################################################################

@dataclass
class GarchArchModel:
    returns: pd.Series
    rv: pd.Series
    result: Optional[any] = None
    scale_factor: float = 1.0  # NEW: Save scaling factor

    @classmethod
    def from_prices(cls, prices: pd.Series, rv: pd.Series, auto_scale: bool = True):
        returns = np.log(prices).diff().dropna()
        returns, rv = cls._align(returns, rv)

        scale_factor = 1.0
        if auto_scale:
            r2_mean = (returns**2).mean()
            rv_mean = rv.mean()
            if rv_mean > 0:
                scale_factor = r2_mean / rv_mean
                rv = rv * scale_factor
                print(f"[auto-scale] RV × {scale_factor:.2e}")
            else:
                print("Warning: RV mean is zero, skipping auto-scale.")

        return cls(returns=returns, rv=rv, scale_factor=scale_factor)

    @staticmethod
    def _align(a: pd.Series, b: pd.Series):
        joined = a.to_frame("a").join(b.to_frame("b"), how="inner")
        return joined["a"], joined["b"]

    def fit(self, verbose=True):
        model = arch_model(self.returns, vol='Garch', p=1, q=1, rescale=False)
        self.result = model.fit(disp="off")
        if verbose:
            print(self.result.summary())
        return self.result

    def predict_volatility(self) -> pd.Series:
        if self.result is None:
            raise RuntimeError("Call fit() first")
        h = self.result.conditional_volatility**2
        h = h   # SCALE prediction as well
        return h

    def plot(self, save: Optional[str] = None):
        h = self.predict_volatility()
        r2 = self.returns**2
        rv = self.rv.loc[h.index]

        fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 3, 1]})

        ax[0].plot(h.index, h, label="hₜ (predicted volatility)", color="blue")
        ax[0].plot(rv.index, rv, label="RV (realized volatility)", color="orange", alpha=0.7)
        ax[0].legend()
        ax[0].set_ylabel("Variance")

        ax[1].plot(h.index, h, label="hₜ (pred r²)", color="green")
        ax[1].plot(rv.index, rv, label="RV (actual)", color="grey", alpha=0.6)
        ax[1].legend()
        ax[1].set_ylabel("Variance")

        ax[2].bar(r2.index, r2, color="grey", alpha=0.4, width=1.0)
        ax[2].set_ylabel("r²")
        ax[2].set_xlabel("Date")

        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=300)
            print(f"Saved → {save}")
        else:
            plt.show()

        return fig

###############################################################################
# MAIN ENTRY POINT
###############################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit GARCH(1,1) using arch and plot against RV.")
    parser.add_argument("--prices", type=str, required=True)
    parser.add_argument("--date-col", type=str, default="DATE")
    parser.add_argument("--price-col", type=str, default="CPrc")
    parser.add_argument("--rv-col", type=str, default="ivol_t")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    prices, rv = load_price_and_rv_series(
        path=args.prices,
        date_col=args.date_col,
        price_col=args.price_col,
        rv_col=args.rv_col,
        ticker=args.ticker,
    )

    model = GarchArchModel.from_prices(prices, rv ** 2)
    model.fit()
    model.plot(save=args.plot)
