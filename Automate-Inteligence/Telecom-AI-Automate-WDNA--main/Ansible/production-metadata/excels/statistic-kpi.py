import os
import pandas as pd
import numpy as np


def main():
    # getting data
    file_path = r"C:\Users\AW474Y\OneDrive - AT&T Mexico\Escritorio\rutina\4.xls"

    if os.path.isfile(file_path):
        try:
            excel_data = pd.read_excel(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} could not be found. "
                                    "Please check the file path and try again.")
    else:
        print(f"The file {file_path} does not exist. Please check the file path again.")
        return

    # save as csv
    excel_data.to_csv('Huawei-4G.csv', index=False)
    csv_data = pd.read_csv('Huawei-4G.csv')

    # unique values
    unique_values = csv_data['C2'].unique()

    # splitting csv based on unique values in C2 column
    for val in unique_values:
        split_data = csv_data[csv_data['C2'] == val]

        # Processing the data similar to the 'ProcessWorksheet' subroutine
        split_data = split_data.drop(['A', 'B'], axis=1)

        # A placeholder for the 'statisticalThreshold' value. Replace this code with actual value.
        statisticalThreshold = "moving_average"

        umbral = split_data.at[4, 'B']
        upper = split_data.at[8, 'B']
        lower = split_data.at[9, 'B']

        mean = np.mean(split_data['B'])

        split_data.at[2, 'A'] = 'Mean'
        split_data.at[2, 'B'] = mean

        split_data.at[3, 'A'] = 'Lower'
        split_data.at[3, 'B'] = lower

        split_data.at[4, 'A'] = 'Upper'
        split_data.at[4, 'B'] = upper

        # Save the processed data to separate csv files
        split_data.to_csv(f'Huawei-4G_{val}.csv', index=False)


if __name__ == '__main__':
    main()