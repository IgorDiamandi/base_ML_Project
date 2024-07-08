import zipfile
from datetime import datetime
import numpy as np
from scipy.stats import zscore
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DTYPE_SPEC = {
    'fiModelDescriptor': 'str',
    'Hydraulics_Flow': 'str',
    'Track_Type': 'str',
    'Undercarriage_Pad_Width': 'str'
}


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


# Function to handle outliers using 6-Sigma method
def handle_outliers_six_sigma(df, columns):
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        df.loc[outliers, column] = mean

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
        #print(f'Column: {column}')
        #print(f'Outliers - Mean without extremums: {mean_value}')
        #print(f'Outliers - {df.loc[outliers, column]}')
        df.loc[outliers, column] = mean_value

    return df


def remove_columns_with_many_nulls(df, threshold=0.5):
    null_percentage = df.isnull().mean()
    columns_to_remove = null_percentage[null_percentage > threshold].index
    df_cleaned = df.drop(columns=columns_to_remove)

    return df_cleaned


def replace_null_with_mean(df, columns):
    for column in columns:
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)

    return df


def load_and_extract_data(zip_path, extract_path):
    print('Loading data...')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.printdir()
        z.extractall(extract_path)
        with z.open('train.csv') as f:
            df = pd.read_csv(f, dtype=DTYPE_SPEC, low_memory=False)

    return df


def handle_missing_values(features):
    numeric_cols = features.select_dtypes(include=['number']).columns
    non_numeric_cols = features.select_dtypes(exclude=['number']).columns
    features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())
    features[non_numeric_cols] = features[non_numeric_cols].fillna(features[non_numeric_cols].mode().iloc[0])

    return features


def encode_categorical_variables(features):
    # Convert date columns to separate year, month, and day columns
    date_cols = [col for col in features.columns if 'date' in col.lower()]
    for col in date_cols:
        features[col] = pd.to_datetime(features[col], errors='coerce')
        features[col + '_year'] = features[col].dt.year
        features[col + '_month'] = features[col].dt.month
        features[col + '_day'] = features[col].dt.day
        features.drop(columns=[col], inplace=True)

    # Identify categorical columns
    categorical_cols = features.select_dtypes(include=['object', 'category']).columns

    # Apply Label Encoding to categorical columns
    le = LabelEncoder()
    for col in categorical_cols:
        features[col] = le.fit_transform(features[col])

    return features


def preprocess_data(df, dropped_coumns):
    df = df.set_index('SalesID')

    # Converting MachineId to Sales count indicator
    machine_id_counts = df['MachineID'].value_counts().reset_index()
    machine_id_counts.columns = ['MachineID', 'MachineID_Count']
    df = df.merge(machine_id_counts, on='MachineID', how='left').drop('MachineID', axis=1)
    # Converting YearMade to MachineAge and cleaning up corrupted values
    df['MachineAge'] = datetime.now().year - df['YearMade']
    df = df.drop(['YearMade'], axis=1)
    df = replace_outliers_with_mean(df, ['MachineAge'], 1)

    # Handle outliers and missing values
    df = handle_outliers_iqr(df, ['SalePrice', 'MachineHoursCurrentMeter'])
    df = replace_null_with_mean(df, ['MachineHoursCurrentMeter'])

    # Dropping irrelevant columns
    df = df.drop(dropped_coumns, axis=1)
    df = remove_columns_with_many_nulls(df, 0.7)

    return df

def preprocess_validation_data(df, dropped_coumns):
    #df = df.set_index('SalesID')

    # Converting MachineId to Sales count indicator
    machine_id_counts = df['MachineID'].value_counts().reset_index()
    machine_id_counts.columns = ['MachineID', 'MachineID_Count']
    df = df.merge(machine_id_counts, on='MachineID', how='left').drop('MachineID', axis=1)
    # Converting YearMade to MachineAge and cleaning up corrupted values
    df['MachineAge'] = datetime.now().year - df['YearMade']
    df = df.drop(['YearMade'], axis=1)
    df = replace_outliers_with_mean(df, ['MachineAge'], 1)

    # Handle outliers and missing values
    df = handle_outliers_iqr(df, ['MachineHoursCurrentMeter'])
    df = replace_null_with_mean(df, ['MachineHoursCurrentMeter'])

    # Dropping irrelevant columns
    df = df.drop(dropped_coumns, axis=1)
    df = remove_columns_with_many_nulls(df, 0.7)

    return df
