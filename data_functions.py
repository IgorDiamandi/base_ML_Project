import pandas as pd
import re
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


#import category_encoders as ce

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


# function to apply target encoding to selected columns
def apply_target_encoding(df_train, df_test, df_valid, target_column, columns_to_encode):
    X_train = df_train.drop(target_column, axis=1)
    y_train = df_train[target_column]

    target_encoder = TargetEncoder(columns_to_encode)
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(df_test.drop(target_column, axis=1))
    X_valid_encoded = target_encoder.transform(df_valid.drop(target_column, axis=1))

    X_train_encoded = pd.concat([X_train_encoded, y_train.reset_index(drop=True)], axis=1)
    X_test_encoded = pd.concat([X_test_encoded, df_test[target_column].reset_index(drop=True)], axis=1)
    X_valid_encoded = pd.concat([X_valid_encoded, df_valid[target_column].reset_index(drop=True)], axis=1)

    return X_train_encoded, X_test_encoded, X_valid_encoded;


def apply_one_hot_encoder(df_train, df_test, df_valid, columns_to_encode):
    one_hot_encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
    preprocessor = ColumnTransformer(transformers=[
        ('onehot', one_hot_encoder, columns_to_encode),
    ],
        remainder='passthrough'
    )
    X_train_encoded = preprocessor.fit_transform(df_train)
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=preprocessor.get_feature_names_out())
    X_Test_encoded = preprocessor.transform(df_test)
    X_Test_encoded_df = pd.DataFrame(X_Test_encoded, columns=preprocessor.get_feature_names_out())
    X_valid_encoded = preprocessor.transform(df_valid)
    X_valid_encoded_df = pd.DataFrame(X_valid_encoded, columns=preprocessor.get_feature_names_out())

    return X_train_encoded_df, X_Test_encoded_df, X_valid_encoded_df;


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

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=0))
    ])

    model.fit(X, y)

    df.loc[df[target_column].isnull(), target_column] = model.predict(X_null)

    return df


def split_fiProductClassDesc(df, column_name):
    # Function to parse each value in the fiProductClassDesc column
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

    unique_characteristics = df['Characteristic'].dropna().unique()
    for characteristic in unique_characteristics:
        df[f'post_fi_{characteristic}'] = df.apply(
            lambda row: f"{row['LowValue']} to {row['HighValue']}" if row['Characteristic'] == characteristic else None,
            axis=1
        )

    df = df.drop(columns=['LowValue', 'HighValue', 'Characteristic'])

    return df