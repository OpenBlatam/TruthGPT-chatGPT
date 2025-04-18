import time
import pandas as pd
import dask.dataframe as dd

COLUMNS_DATA = {
    16: {'threshold': 1, 'operator': 'gt'},
    12: {'threshold': 0.03, 'operator': 'gt'},
    27: {'threshold': 3, 'operator': 'gt'},
    21: {'threshold': 0, 'operator': 'lt'}
}

CHUNK_SIZE = 50000
COLUMN_LIMIT = 50
INPUT_FILE_NAME = 'input_file.csv'
OUTPUT_FILE_NAME = 'output_file.csv'

class DataProcessor:
    op_map = {'gt': '>', 'lt': '<'}

    @staticmethod
    def adjust_values(df, column, threshold, operator):
        df.loc[df[column].map(lambda x: eval(f'{x}{DataProcessor.op_map[operator]}{threshold}')), column] = 'Adjusted Value'
        return df


class PandasDataProcessor(DataProcessor):

    @staticmethod
    def process():
        for i, chunk in enumerate(pd.read_csv(INPUT_FILE_NAME, chunksize=CHUNK_SIZE, usecols=range(COLUMN_LIMIT))):
            for column, data in COLUMNS_DATA.items():
                df = DataProcessor.adjust_values(chunk, column, data['threshold'], data['operator'])
            if i == 0:
                df.to_csv(OUTPUT_FILE_NAME, index=False)
            else:
                df.to_csv(OUTPUT_FILE_NAME, mode='a', header=False, index=False)


class DaskDataProcessor(DataProcessor):

    @staticmethod
    def process():
        dask_df = dd.read_csv(INPUT_FILE_NAME, assume_missing=True, usecols=range(COLUMN_LIMIT))
        for column, data in COLUMNS_DATA.items():
            dask_df = dask_df.map_partitions(DataProcessor.adjust_values, column=column, threshold=data['threshold'], operator=data['operator'])

        dask_df.to_csv(OUTPUT_FILE_NAME, single_file=True, index=False)


if __name__ == '__main__':
    start_time = time.perf_counter()
    PandasDataProcessor.process()
    pandas_time = time.perf_counter() - start_time
    print("Pandas execution time:", pandas_time)

    start_time = time.perf_counter()
    DaskDataProcessor.process()
    dask_time = time.perf_counter() - start_time
    print("Dask execution time:", dask_time)