import pandas as pd
import numpy as np
import pandas_datareader.data as web
import logging
from config import INDICATORS, START_DATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataFetcher:
    def __init__(self, start_date=START_DATE, indicators=INDICATORS):
        self.start_date = start_date
        self.indicators = indicators

    def fetch_fred_data(self) -> pd.DataFrame:
        logging.info("Fetching data from FRED API...")
        series_ids = list(self.indicators.keys())
        try:
            df = web.DataReader(series_ids, 'fred', self.start_date)
            df = df.resample('ME').last()
            logging.info("Data fetched and aligned successfully.")
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying stationarity transformations...")
        df_transformed = pd.DataFrame(index=df.index)
        
        for series, params in self.indicators.items():
            if params['transform'] == 5:
                periods = 3 if params['freq'] == 'Q' else 1
                df_transformed[series] = np.log(df[series]).diff(periods=periods) * 100
            else:
                df_transformed[series] = df[series]

        df_transformed = df_transformed.dropna(how='all')
        return df_transformed

    def get_processed_data(self) -> pd.DataFrame:
        raw_data = self.fetch_fred_data()
        return self.transform_data(raw_data)