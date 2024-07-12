import pandas as pd
import re
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
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
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Missing').astype(str)
    return df


def calculate_target_encoding(df, target_column, columns_to_encode):
    target_means = {}
    global_mean = df[target_column].mean()

    for col in columns_to_encode:
        target_means[col] = df.groupby(col)[target_column].mean().to_dict()
        target_means[col]['__global_mean__'] = global_mean

    return target_means


def apply_target_encoding(df, target_means, columns_to_encode):
    for col in columns_to_encode:
        df[col + '_encoded'] = df[col].map(target_means[col]).fillna(target_means[col]['__global_mean__'])

    return df


def target_encode_train_test_valid(df_train, df_test, df_valid, target_column, columns_to_encode):
    target_means = calculate_target_encoding(df_train, target_column, columns_to_encode)

    df_train_encoded = apply_target_encoding(df_train.copy(), target_means, columns_to_encode)
    df_test_encoded = apply_target_encoding(df_test.copy(), target_means, columns_to_encode)
    df_valid_encoded = apply_target_encoding(df_valid.copy(), target_means, columns_to_encode)

    return df_train_encoded, df_test_encoded, df_valid_encoded


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def apply_one_hot_encoder(df_train, df_test, df_valid, columns_to_encode, n_features=10):
    # Function to convert DataFrame columns to the required format for FeatureHasher
    def convert_to_iterables(df, columns):
        return df[columns].astype(str).values.tolist()

    # Convert categorical columns to required format
    df_train_hashed = convert_to_iterables(df_train, columns_to_encode)
    df_test_hashed = convert_to_iterables(df_test, columns_to_encode)
    df_valid_hashed = convert_to_iterables(df_valid, columns_to_encode)

    # Use FeatureHasher for categorical columns
    hasher = FeatureHasher(n_features=n_features, input_type='string')

    # Transform the data
    X_train_hashed = hasher.fit_transform(df_train_hashed)
    X_test_hashed = hasher.transform(df_test_hashed)
    X_valid_hashed = hasher.transform(df_valid_hashed)

    # Combine the hashed features with the rest of the features
    df_train_encoded = df_train.drop(columns=columns_to_encode).reset_index(drop=True)
    df_test_encoded = df_test.drop(columns=columns_to_encode).reset_index(drop=True)
    df_valid_encoded = df_valid.drop(columns=columns_to_encode).reset_index(drop=True)

    # Convert sparse matrix to dense and create DataFrames
    X_train_encoded_df = pd.concat([df_train_encoded, pd.DataFrame(X_train_hashed.toarray())], axis=1)
    X_test_encoded_df = pd.concat([df_test_encoded, pd.DataFrame(X_test_hashed.toarray())], axis=1)
    X_valid_encoded_df = pd.concat([df_valid_encoded, pd.DataFrame(X_valid_hashed.toarray())], axis=1)

    # Ensure all column names are strings
    X_train_encoded_df.columns = X_train_encoded_df.columns.astype(str)
    X_test_encoded_df.columns = X_test_encoded_df.columns.astype(str)
    X_valid_encoded_df.columns = X_valid_encoded_df.columns.astype(str)

    # Ensure columns are in the same order
    common_columns = X_train_encoded_df.columns
    X_test_encoded_df = X_test_encoded_df[common_columns]
    X_valid_encoded_df = X_valid_encoded_df[common_columns]

    return X_train_encoded_df, X_test_encoded_df, X_valid_encoded_df

def unite_sparse_columns(df, columns_to_unite, new_column_name):
    columns_to_unite = [col for col in columns_to_unite if col != new_column_name]

    existing_columns = [col for col in columns_to_unite if col in df.columns]
    missing_columns = [col for col in columns_to_unite if col not in df.columns]

    if missing_columns:
        print(f"Warning: The following columns are missing and will be ignored: {missing_columns}")

    if not existing_columns:
        raise ValueError("None of the specified columns to unite exist in the DataFrame.")

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
