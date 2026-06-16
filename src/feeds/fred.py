"""
FedClient — stateful FRED rate curve helper.

Fetches USD interest rate pillars from FRED, bootstraps a zero-coupon discount
curve, and exposes interpolated discount factors at arbitrary tenors.

Data is daily; the client caches for 3 hours to avoid redundant HTTP calls.

Credentials
-----------
Set the FRED_API_KEY environment variable, or pass ``api_key`` directly:
    client = FedClient(api_key="your_key")
"""

from __future__ import annotations

import math
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

try:
    from fredapi import Fred
except ImportError as exc:  # pragma: no cover
    raise ImportError("fredapi is required: pip install fredapi") from exc


# ---------------------------------------------------------------------------
# Pillars: FRED series → tenor in years
# ---------------------------------------------------------------------------
_PILLARS: dict[str, float] = {
    "SOFR":   1 / 365,    # overnight SOFR
    "DGS1MO": 1 / 12,     # 1-month T-bill
    "DGS3MO": 3 / 12,     # 3-month T-bill
    "DGS6MO": 6 / 12,     # 6-month T-bill
    "DGS1":   1.0,         # 1-year CMT
    "DGS2":   2.0,
    "DGS3":   3.0,
    "DGS5":   5.0,
    "DGS10":  10.0,
}

# These tenors are zero-coupon (T-bills / SOFR) — no coupon stripping needed.
_TBILL_SERIES = frozenset({"SOFR", "DGS1MO", "DGS3MO", "DGS6MO"})

_CACHE_TTL = timedelta(hours=3)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _interp_df(tenor: float, curve: list[tuple[float, float]]) -> float:
    """
    Log-linearly interpolate (or flat-forward extrapolate) a discount factor
    from a sorted list of (tenor_years, df) pillar pairs.
    """
    if not curve or tenor <= 0:
        return 1.0

    tenors = np.array([t for t, _ in curve])
    log_dfs = np.array([math.log(max(df, 1e-12)) for _, df in curve])

    if tenor <= tenors[-1]:
        return math.exp(float(np.interp(tenor, tenors, log_dfs)))

    # Flat zero-rate extrapolation beyond the last pillar
    z_last = -log_dfs[-1] / tenors[-1]
    return math.exp(-z_last * tenor)


# ---------------------------------------------------------------------------
# FedClient
# ---------------------------------------------------------------------------

class FedClient:
    """
    Stateful USD rate-curve client backed by FRED.

    Key methods
    -----------
    get_rates()                          → DataFrame[series_id, tenor_years, rate]
    bootstrap_discount_factors(as_of)    → DataFrame[maturity_date, discount_factor]
    get_discount_factor(tenor_years)     → float  (scalar pointwise query)
    get_zero_rate(tenor_years)           → float  (continuously compounded)
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("FRED_API_KEY")
        if not key:
            raise ValueError(
                "A FRED API key is required. Pass api_key= or set FRED_API_KEY."
            )
        self._fred = Fred(api_key=key)
        self._rates: Optional[dict[str, float]] = None
        self._fetched_at: Optional[datetime] = None
        self._curve: Optional[list[tuple[float, float]]] = None  # [(tenor, df)]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    @property
    def _cache_stale(self) -> bool:
        if self._fetched_at is None:
            return True
        return datetime.now(timezone.utc) - self._fetched_at > _CACHE_TTL

    # ------------------------------------------------------------------
    # Raw rates
    # ------------------------------------------------------------------
    def _rates_dict(self, force_refresh: bool = False) -> dict[str, float]:
        """
        Latest decimal rate per pillar (internal raw form).

        Results are cached for 3 hours; FRED data is only published daily so
        polling more frequently is wasteful.
        """
        if not force_refresh and not self._cache_stale and self._rates is not None:
            return self._rates

        rates: dict[str, float] = {}
        for series_id in _PILLARS:
            series = self._fred.get_series(series_id)
            rates[series_id] = float(series.dropna().iloc[-1]) / 100.0

        self._rates = rates
        self._fetched_at = datetime.now(timezone.utc)
        self._curve = None  # invalidate bootstrapped curve
        return rates

    def get_rates(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Latest available rate for each pillar, as decimals (FRED's 4.50 → 0.045).

        Results are cached for 3 hours; FRED data is only published daily so
        polling more frequently is wasteful.

        Returns
        -------
        DataFrame with columns ``series_id``, ``tenor_years``, ``rate``,
        sorted by ascending tenor.
        """
        rates = self._rates_dict(force_refresh)
        return (
            pd.DataFrame(
                [(sid, _PILLARS[sid], rates[sid]) for sid in _PILLARS],
                columns=["series_id", "tenor_years", "rate"],
            )
            .sort_values("tenor_years")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Bootstrapped curve
    # ------------------------------------------------------------------
    def _bootstrap_dict(self, as_of: Optional[date] = None) -> dict[date, float]:
        """
        Bootstrap a zero-coupon discount curve from FRED par yields (internal).

        For T-bill tenors (SOFR / DGS1MO / DGS3MO / DGS6MO):
            Simple-interest zero coupon:  P(0,T) = 1 / (1 + r·T)

        For bond tenors (DGS1 and longer, semi-annual coupon):
            Bootstrap via coupon stripping:
                P(0,T) = (1 − c/2 · Σ P(0,tᵢ)) / (1 + c/2)
            where the Σ runs over all intermediate semi-annual coupon dates.
            Discount factors for dates not yet bootstrapped are interpolated
            log-linearly from the already-known pillars.

        Returns
        -------
        dict mapping each pillar maturity date → discount factor.
        """
        rates = self._rates_dict()
        today = as_of or date.today()

        sorted_pillars = sorted(_PILLARS.items(), key=lambda kv: kv[1])
        bootstrapped: list[tuple[float, float]] = []  # built up as we go
        result: dict[date, float] = {}

        for series_id, tenor_years in sorted_pillars:
            r = rates[series_id]

            if series_id in _TBILL_SERIES:
                # Zero-coupon instrument — exact by construction
                df = 1.0 / (1.0 + r * tenor_years)

            else:
                # Semi-annual coupon bond: strip coupons
                c2 = r / 2.0                            # half-coupon rate
                n = round(tenor_years * 2)              # number of semi-annual periods
                # Sum present value of all coupons except the final one
                coupon_pv = sum(
                    _interp_df(k / 2.0, bootstrapped)
                    for k in range(1, n)
                )
                df = (1.0 - c2 * coupon_pv) / (1.0 + c2)

            df = min(max(df, 1e-8), 1.0)               # numeric guard
            bootstrapped.append((tenor_years, df))
            maturity_date = today + timedelta(days=round(tenor_years * 365))
            result[maturity_date] = df

        return result

    def bootstrap_discount_factors(
        self, as_of: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Bootstrap a zero-coupon discount curve from FRED par yields.

        See ``_bootstrap_dict`` for the coupon-stripping methodology.

        Returns
        -------
        DataFrame with columns ``maturity_date`` and ``discount_factor``,
        sorted by ascending maturity.
        """
        dfs = self._bootstrap_dict(as_of)
        return pd.DataFrame(
            sorted(dfs.items()), columns=["maturity_date", "discount_factor"]
        )

    def _get_curve(self) -> list[tuple[float, float]]:
        """Return sorted [(tenor, df)] for the bootstrapped curve, with caching."""
        if self._curve is not None:
            return self._curve

        today = date.today()
        dfs = self._bootstrap_dict(today)
        pairs = []
        for series_id, tenor_years in _PILLARS.items():
            mat = today + timedelta(days=round(tenor_years * 365))
            if mat in dfs:
                pairs.append((tenor_years, dfs[mat]))
        self._curve = sorted(pairs)
        return self._curve

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------
    def get_discount_factor(self, tenor_years: float) -> float:
        """Interpolated discount factor P(0, T) for an arbitrary tenor."""
        return _interp_df(tenor_years, self._get_curve())

    def get_zero_rate(self, tenor_years: float) -> float:
        """
        Continuously compounded zero rate for a given tenor (in years).
        z(T) = −ln P(0,T) / T
        """
        if tenor_years <= 0:
            return 0.0
        df = self.get_discount_factor(tenor_years)
        return -math.log(max(df, 1e-12)) / tenor_years
