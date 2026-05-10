from data_pipeline import DataFetcher
from nowcast_engine import DynamicFactorNowcaster
import logging

def main():
    pipeline = DataFetcher()
    df_stationary = pipeline.get_processed_data()

    nowcaster = DynamicFactorNowcaster(df_stationary)
    nowcaster.build_model(factors=1, factor_order=2)
    nowcaster.fit()

    print("\n--- Model Summary ---")

    print("\n--- Latest GDP Estimates (Including Unreleased Months) ---")
    latest_estimates = nowcaster.results.fittedvalues['GDPC1'].tail(6)
    print(latest_estimates)

    forecast = nowcaster.get_nowcast(steps=3)
    print("\n--- Next Quarter Projection ---")
    print(forecast['GDPC1'])

    nowcaster.plot_nowcast('GDPC1', tail=12)

if __name__ == "__main__":
    main()