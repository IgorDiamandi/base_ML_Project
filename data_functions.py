import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


# Function to retrieve statistic parameters from numerical columns of the train dataframe
def compute_statistics(df):
    numeric_df = df.select_dtypes(include='number')

    mean_values = numeric_df.mean()
    iqr_values = numeric_df.quantile(0.75) - numeric_df.quantile(0.25)
    zscore_values = (numeric_df.mean() / numeric_df.std()).mean()

    # Mean without extremes (assuming extremes are values outside 1.5*IQR)
    def mean_without_extremes(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        filtered_series = series[(series >= Q1 - 1.5 * IQR) & (series <= Q3 + 1.5 * IQR)]
        return filtered_series.mean()

    mean_no_extremes_values = numeric_df.apply(mean_without_extremes)

    statistics_df = pd.DataFrame({
        'mean': mean_values,
        'iqr': iqr_values,
        'zscore': [zscore_values] * len(numeric_df.columns),
        'mean_without_extremes': mean_no_extremes_values
    })

    statistics_df = statistics_df.T
    return statistics_df


def get_stat_value(method, column, statistics_df):
    if method not in statistics_df.index:
        raise ValueError(f"Method {method} not found in statistics DataFrame")
    return statistics_df.loc[method, column]


def replace_outliers(df, column, method, statistics_df):
    stat_value = get_stat_value(method, column, statistics_df)

    # Define the threshold for identifying outliers
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with the statistical method value
    df[column] = df[column].apply(lambda x: stat_value if x < lower_bound or x > upper_bound else x)

    return df


def replace_nans(df, column, method, statistics_df):
    stat_value = get_stat_value(method, column, statistics_df)
    df[column] = df[column].fillna(stat_value)

    return df


def get_rmse(y, y_pred):
    return mean_squared_error(y, y_pred) ** 0.5


def split_product_class_series(series):
    equipment_type = []
    details = []

    for item in series:
        if pd.isna(item):
            equipment_type.append(None)
            details.append(None)
        else:
            split_item = item.split(' - ', 1)
            equipment_type.append(split_item[0])
            details.append(split_item[1] if len(split_item) > 1 else None)

    return pd.Series(equipment_type), pd.Series(details)

def replace_nan_with_string(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].apply(type).nunique() > 1:
                df[col] = df[col].fillna('Missing').astype(str)
            df[col] = le.fit_transform(df[col])