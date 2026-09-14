# code/data/fx_converter.py
"""
FX Converter: Fixed dated exchange rate lookup and normalization.
Reads dataset/exchange_rates.csv and accurately converts foreign currency amounts.
"""

from typing import Dict, Tuple, Optional
import pandas as pd
from code.config import config


class FXConverter:
    def __init__(self, rates_csv_path=None):
        self.rates_path = rates_csv_path or (config.dataset_dir / "exchange_rates.csv")
        self.rates_map: Dict[Tuple[str, str, str], float] = {}
        self._load_rates()

    def _load_rates(self):
        """Loads and indexes fixed dated exchange rates from exchange_rates.csv."""
        if not self.rates_path.exists():
            return
        df = pd.read_csv(self.rates_path)
        for row in df.itertuples(index=False):
            key = (str(row.rate_date).strip(), str(row.from_currency).strip(), str(row.to_currency).strip())
            self.rates_map[key] = float(row.rate)

    def convert(self, amount: float, from_currency: str, to_currency: str, date: str) -> float:
        """
        Converts an amount from one currency to another on a specific settlement date.
        If currencies are identical or amount is zero, returns original amount.
        """
        from_curr = from_currency.strip().upper()
        to_curr = to_currency.strip().upper()
        if from_curr == to_curr or amount == 0:
            return amount

        # 1. Direct dated rate lookup (exact date and direction)
        key = (date, from_curr, to_curr)
        if key in self.rates_map:
            return amount * self.rates_map[key]

        # 2. Check inverse pair on the same date (1 / rate)
        inv_key = (date, to_curr, from_curr)
        if inv_key in self.rates_map:
            rate = self.rates_map[inv_key]
            if rate > 0:
                return amount / rate

        # 3. Fallback: Find closest available date for this currency pair
        matching_rates = [
            (k[0], v) for k, v in self.rates_map.items() if k[1] == from_curr and k[2] == to_curr
        ]
        if matching_rates:
            matching_rates.sort(key=lambda x: abs(pd.to_datetime(x[0]) - pd.to_datetime(date)))
            return amount * matching_rates[0][1]

        # 4. Fallback: Check matching inverse rates on closest date
        matching_inv = [
            (k[0], v) for k, v in self.rates_map.items() if k[1] == to_curr and k[2] == from_curr
        ]
        if matching_inv:
            matching_inv.sort(key=lambda x: abs(pd.to_datetime(x[0]) - pd.to_datetime(date)))
            return amount / matching_inv[0][1]

        raise ValueError(f"No exchange rate found for {from_curr} -> {to_curr} around date {date}")
