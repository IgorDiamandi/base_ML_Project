import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def compute_statistics(df):
    numeric_df = df.select_dtypes(include='number')

    mean_values = numeric_df.mean()
    iqr_values = numeric_df.quantile(0.75) - numeric_df.quantile(0.25)
    zscore_values = (numeric_df.mean() / numeric_df.std()).mean()

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

    return statistics_df.T


def get_stat_value(method, column, statistics_df):
    if method not in statistics_df.index:
        raise ValueError(f"Method {method} not found in statistics DataFrame")
    return statistics_df.loc[method, column]


def replace_outliers(dfs, column, method, statistics_df):
    stat_value = get_stat_value(method, column, statistics_df)

    def replace_outlier_values(df, column, lower_bound, upper_bound, stat_value):
        df[column] = df[column].apply(lambda x: stat_value if x < lower_bound or x > upper_bound else x)
        return df

    updated_dfs = []
    for df in dfs:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        updated_df = replace_outlier_values(df.copy(), column, lower_bound, upper_bound, stat_value)
        updated_dfs.append(updated_df)

    return updated_dfs


def replace_nans(dfs, column, method, statistics_df):
    stat_value = get_stat_value(method, column, statistics_df)
    for df in dfs:
        df[column] = df[column].fillna(stat_value)
    return dfs


def get_rmse(y_log, y_pred_log):
    y = np.expm1(y_log)
    y_pred = np.expm1(y_pred_log)

    return mean_squared_error(y, y_pred) ** 0.5

'''
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
'''


def apply_one_hot_encoder(dfs, columns_to_encode, excluded_columns=[]):
    one_hot_encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', one_hot_encoder, columns_to_encode)
        ],
        remainder='passthrough'
    )

    # Fit and transform the first dataframe to get the feature names
    X_encoded = [preprocessor.fit_transform(dfs[0])]
    feature_names = preprocessor.get_feature_names_out()

    # Transform the remaining dataframes
    for df in dfs[1:]:
        X_encoded.append(preprocessor.transform(df))

    # Create DataFrames from the encoded arrays
    encoded_dfs = [pd.DataFrame(X.toarray(), columns=feature_names) for X in X_encoded]

    # Remove excluded columns
    for df in encoded_dfs:
        df.drop(columns=excluded_columns, errors='ignore', inplace=True)

    return encoded_dfs


def unite_sparse_columns(df, columns_to_unite, new_column_name):
    columns_to_unite = [col for col in columns_to_unite if col != new_column_name]

    existing_columns = [col for col in columns_to_unite if col in df.columns]
    missing_columns = [col for col in columns_to_unite if col not in df.columns]

    if missing_columns:
        print(f"Warning: The following columns are missing and will be ignored: {missing_columns}")

    if not existing_columns:
        raise ValueError("None of the specified columns to unite exist in the DataFrame.")

    # Replace "None or Unspecified" with np.nan
    df[existing_columns] = df[existing_columns].replace("None or Unspecified", np.nan)

    df[new_column_name + '_non_null_count'] = df[existing_columns].notnull().sum(axis=1)
    df[new_column_name + '_any_non_null'] = df[existing_columns].notnull().any(axis=1).astype(int)

    if df[existing_columns].apply(lambda col: col.map(lambda x: isinstance(x, (int, float)))).all().all():
        df[new_column_name + '_sum'] = df[existing_columns].sum(axis=1, skipna=True)

    if df[existing_columns].apply(lambda col: col.map(lambda x: isinstance(x, str))).all().all():
        df[new_column_name + '_mode'] = df[existing_columns].mode(axis=1)[0]

    df = df.drop(columns=existing_columns)

    return df


def missing_values_imputation(df, target_column, feature_columns):
    df_notnull = df.dropna(subset=[target_column])
    df_null = df[df[target_column].isnull()]

    if df_null.empty:
        return df

    X = df_notnull[feature_columns]
    y = df_notnull[target_column]
    X_null = df_null[feature_columns]

    categorical_features = [col for col in feature_columns if df[col].dtype == 'object']
    numerical_features = [col for col in feature_columns if df[col].dtype != 'object']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=0, criterion='entropy'))
    ])

    model.fit(X, y)
    df.loc[df[target_column].isnull(), target_column] = model.predict(X_null)

    return df


def split_fiProductClassDesc(df, column_name):
    def parse_fiProductClassDesc(value):
        match = re.match(r'^(.*) - (\d+\.\d+) to (\d+\.\d+) (.*)$', value)
        if match:
            category, low_value, high_value, characteristic = match.groups()
            return category, float(low_value), float(high_value), characteristic
        else:
            return None, None, None, None

    parsed_values = df[column_name].apply(parse_fiProductClassDesc)

    df['Category'] = parsed_values.apply(lambda x: x[0])
    df['LowValue'] = parsed_values.apply(lambda x: x[1])
    df['HighValue'] = parsed_values.apply(lambda x: x[2])
    df['Characteristic'] = parsed_values.apply(lambda x: x[3])

    # Calculate the average of LowValue and HighValue
    df['AverageValue'] = (df['LowValue'] + df['HighValue']) / 2

    unique_characteristics = df['Characteristic'].dropna().unique()
    for characteristic in unique_characteristics:
        df[f'post_fi_{characteristic}'] = df.apply(
            lambda row: row['AverageValue'] if row['Characteristic'] == characteristic else None,
            axis=1
        )

    df = df.drop(columns=['LowValue', 'HighValue', 'Characteristic', 'AverageValue'])

    return df


def apply_lable_encoding(dfs, columns):
    encoders = {col: LabelEncoder() for col in columns}

    # Combine unique values for each column across all DataFrames
    for col in columns:
        unique_values = pd.concat([df[col] for df in dfs]).unique()
        encoders[col].fit(unique_values)

    # Encode the columns in all DataFrames
    for df in dfs:
        for col in columns:
            df[col] = df[col].map(lambda s: '<unknown>' if s not in encoders[col].classes_ else s)

    # Add '<unknown>' to the encoder classes and transform the columns
    for col in columns:
        encoders[col].classes_ = np.append(encoders[col].classes_, '<unknown>')

    for df in dfs:
        for col in columns:
            df[col] = encoders[col].transform(df[col])

    return dfs


#region Taina's functions
def categorize_stick_length(value):
    if pd.isna(value):
        return 'None or Unspecified'
    if value == 'None or Unspecified':
        return value
    if isinstance(value, str):
        if value.startswith(('6', '7', '8', '9')):
            return '<10'
        elif value.startswith(('10', '11', '12', '13', '14')):
            return '10 to 15'
        else:
            return 'above 15'
    return 'None or Unspecified'

def fill_product_size(row, size_mapping):
    if pd.isna(row['ProductSize']):
        return size_mapping.get(row['fiProductClassDesc'], 'None or Unspecified')
    else:
        return row['ProductSize']

def fill_missing_with_mode(train, valid, columns):
  for column in columns:
        if column in train.columns:
            # Compute the mode from the train DataFrame
            mode_value = train[column].mode()[0]
            # Fill missing values in both train and valid DataFrames with the mode from the train DataFrame
            train[column].fillna(mode_value, inplace=True)
            if column in valid.columns:
                valid[column].fillna(mode_value, inplace=True)
            else:
                print(f"Column {column} not found in valid DataFrame.")
        else:
            print(f"Column {column} not found in train DataFrame.")
  return train, valid


def handle_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)
    return df


#endregion