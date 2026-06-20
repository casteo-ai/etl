"""
Smoke test for the Interactive Brokers client.

Connects to a local TWS / IB Gateway session (read-only) and pulls recent
1-minute EUR/USD candles. EUR/USD MIDPOINT data is free, so this works without
any market-data subscription — a good first connectivity check.

Prerequisites
-------------
* TWS (or IB Gateway) running, with API enabled
  (Global Configuration -> API -> Settings -> "Enable ActiveX and Socket Clients").
* PORT below must match that TWS session: paper TWS 7497, live TWS 7496,
  Gateway 4002 (paper) / 4001 (live).

Run from the project root:

    python main.py
"""

from __future__ import annotations

import pandas as pd

from src.feeds.interactive_brokers import InteractiveBrokersClient

# Match this to your running TWS session (live TWS defaults to 7496).
PORT = 7496
# Must be unique per simultaneous API connection.
CLIENT_ID = 11


def main() -> None:
    with InteractiveBrokersClient(port=PORT, client_id=CLIENT_ID, connect=True) as client:
        print(f"Connected to TWS on port {PORT}: {client.is_connected}\n")

        # Pull the last 5 days of 1-minute EUR/USD midpoint candles.
        candles = client.fetch_1min_candles(
            symbol="EUR/USD",
            security_type="fx",
            what_to_show="MIDPOINT",
            use_rth=False,
            start=pd.Timestamp.now("UTC") - pd.Timedelta(days=5),
        )

        print(f"EUR/USD 1m candles over last 5 days: {len(candles)}")
        if not candles.empty:
            print(f"window: {candles.index.min()}  ->  {candles.index.max()}\n")
            print(candles.tail(5).to_string())
        else:
            print("No candles returned (the FX market is closed on weekends).")


if __name__ == "__main__":
    main()
