import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import logging
from typing import Tuple

class DynamicFactorNowcaster:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.model = None
        self.results = None

    def build_model(self, factors=1, factor_order=2):
        logging.info(f"Building Dynamic Factor Model ({factors} factors, order {factor_order})...")

        self.model = sm.tsa.DynamicFactorMQ(
            self.data, 
            factors=factors, 
            factor_orders=factor_order
        )

    def fit(self):
        logging.info("Fitting model via Expectation-Maximization and Kalman Filter...")
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        self.results = self.model.fit(disp=False)
        logging.info("Model fitting complete.")

    def get_nowcast(self, steps=3) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.results.forecast(steps=steps)
        return forecast

    def extract_latent_factor(self) -> pd.Series:
        if self.results is None:
            raise ValueError("Model not fitted.")
        return self.results.factors.smoothed.iloc[:, 0]

    def plot_nowcast(self, series_name='GDPC1', tail=24):
        if self.results is None:
            raise ValueError("Model not fitted.")
        
        fig, ax = plt.subplots(figsize=(12, 6))

        actual = self.data[series_name].dropna().tail(tail)
        fitted = self.results.fittedvalues[series_name].tail(tail * 3)
        
        ax.plot(actual.index, actual, 'ko', label='Actual Releases')
        ax.plot(fitted.index, fitted, 'b-', label='Kalman Filter Estimate (Nowcast)')
        
        ax.set_title(f"Real-Time Nowcast Tracking: {series_name}")
        ax.set_ylabel("Growth Rate (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()