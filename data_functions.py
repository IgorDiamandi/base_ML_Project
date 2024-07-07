import numpy as np
from scipy.stats import zscore

def get_rmse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean() ** 0.5


def handle_outliers_iqr(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

    return df


# Function to handle outliers using Z-score method
def handle_outliers_zscore(df, columns, threshold=3):
    for column in columns:
        z_scores = zscore(df[column])
        abs_z_scores = np.abs(z_scores)
        filtered_entries = abs_z_scores < threshold
        df = df[filtered_entries]
    return df


def mean_without_extremums(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    mean_value = filtered_df[column].mean()
    return mean_value


def replace_outliers_with_mean(df, columns, threshold=3):
    for column in columns:
        mean_value = mean_without_extremums(df, column)
        z_scores = zscore(df[column])
        outliers = np.abs(z_scores) > threshold
        print(f'Column: {column}')
        print(f'Outliers - Mean without extremums: {mean_value}')
        print(f'Outliers - {df.loc[outliers, column]}')
        df.loc[outliers, column] = mean_value

    return df


def remove_columns_with_many_nulls(df, threshold=0.5):
    null_percentage = df.isnull().mean()
    columns_to_remove = null_percentage[null_percentage > threshold].index
    df_cleaned = df.drop(columns=columns_to_remove)

    return df_cleaned


def replace_null_with_mean(df, columns):
    for column in columns:
        if column in df.columns:
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
        else:
            print(f"Column '{column}' not found in DataFrame")
    return df


