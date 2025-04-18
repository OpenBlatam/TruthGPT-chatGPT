import pandas as pd

# Constants
COLUMN_A = 16  # 0-indexed in pandas
COLUMN_B = 12
COLUMN_C = 27
COLUMN_D = 21

def adjust_values_in_data_frame(df):
    conditions = [
        (COLUMN_A, 1),
        (COLUMN_B, 0.03),
        (COLUMN_C, 3),
        (COLUMN_D, 0, 'less')
    ]

    for column, value, comparison in conditions:
        if comparison == 'less':
            condition_met = df.iloc[:, column] < value
        else:
            condition_met = df.iloc[:, column] > value

        df.loc[condition_met, df.columns[column]] = 'Adjusted_Value'
    return df

if __name__ == '__main__':
    file_path = r"/path/to/my/csv/file.csv"
    # Assume that csv file has no header, otherwise you will have to make adjustments
    df = pd.read_csv(file_path, header=None)
    adjusted_df = adjust_values_in_data_frame(df)

# If you would like to write it back to the CSV:
    adjusted_df.to_csv(file_path, index=False)