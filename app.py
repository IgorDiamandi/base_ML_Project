import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_functions import compute_statistics, replace_outliers, \
    unite_sparse_columns, missing_values_imputation, \
    split_fiProductClassDesc, apply_one_hot_encoder, replace_nans, apply_lable_encoding
from model_functions import train_and_evaluate_model

#region Constants
LEVEL_OF_PARALLELISM = -1
NUMBER_OF_TREES = 100
TREE_DEPTH = [19]
MIN_SAMPLES_SPLIT = 3
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = 0.8

DONT_ENCODE_THEM_COLUMNS = ['ModelID', 'Income', 'AgeAtSale', 'SalesID', 'MachineHoursCurrentMeter']

KILL_THEM_COLUMNS = ['uControl_any_non_null', 'uMounting_any_non_null', 'uHydraulics_non_null_count',
                     'uControl_non_null_count', 'fiProductClassDesc', 'post_fi_Metric Tons',
                     'post_fi_Ft Standard Digging Depth', 'post_fi_Lb Operating Capacity']

SIZE_FIT_COLUMNS = ['fiModelDesc', 'post_fi_Horsepower',
                    'Drive_System', 'Stick_Length', 'Undercarriage_Pad_Width',
                    'post_fi_Metric Tons']

USAGE_FIT_COLUMNS = ['AgeAtSale', 'MachineHoursCurrentMeter']

DUPLICATE_COLUMNS = ['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor']

LABEL_ENCODING_COLUMNS = ['fiModelDesc', 'Enclosure', 'post_fi_Horsepower']

COLUMN_GROUPS = {
    'uBladeStick': ['Blade_Extension', 'Stick_Length', 'Stick'],
    'uTrack': ['Pad_Type', 'Grouser_Type', 'Grouser_Tracks'],
    'uMounting': ['Backhoe_Mounting', 'Forks', 'Pushblock', 'Ripper', 'Scarifier', 'Thumb'],
    'uControl': ['Travel_Controls', 'Ride_Control', 'Transmission',
                 'Pattern_Changer', 'Tip_Control', 'Coupler', 'Coupler_System'],
    'uHydraulics': ['Turbocharged', 'Hydraulics', 'Hydraulics_Flow'],
    'uDrive': ['Differential_Type', 'Drive_System']
}

DTYPE_SPEC = {
    13: 'str',
    39: 'str',
    40: 'str',
    41: 'str'
}
#endregion

#region Load CSV
df_state_demographics = pd.read_csv('Data\\state_demographics.csv')
df = pd.read_csv('Data\\train.csv', dtype=DTYPE_SPEC)
df_valid = pd.read_csv('Data\\valid.csv', dtype=DTYPE_SPEC)
#endregion

#region Adding External Data
# adding external states demographics data to both datasets
df = pd.merge(df, df_state_demographics, on='state', how='left')
df_valid = pd.merge(df_valid, df_state_demographics, on='state', how='left')
#endregion

#region Product size handling
df = split_fiProductClassDesc(df, 'fiProductClassDesc')
df_valid = split_fiProductClassDesc(df_valid, 'fiProductClassDesc')

df = missing_values_imputation(df, 'ProductSize', SIZE_FIT_COLUMNS)
df_valid = missing_values_imputation(df_valid, 'ProductSize', SIZE_FIT_COLUMNS)
#endregion

#region Merging bad columns to one bad column
# grouping columns with technical details
for new_column, columns in COLUMN_GROUPS.items():
    df = unite_sparse_columns(df, columns, new_column)
    df_valid = unite_sparse_columns(df_valid, columns, new_column)
#endregion

#region Convert dates to periods and categories
df['saleyear'] = pd.to_datetime(df['saledate']).dt.year
df_valid['saleyear'] = pd.to_datetime(df_valid['saledate']).dt.year

df['salemonth'] = pd.to_datetime(df['saledate']).dt.month
df_valid['salemonth'] = pd.to_datetime(df_valid['saledate']).dt.month

df['saledayofweek'] = pd.to_datetime(df['saledate']).dt.dayofweek
df_valid['saledayofweek'] = pd.to_datetime(df_valid['saledate']).dt.dayofweek

df['AgeAtSale'] = df['saleyear'] - df['YearMade']
df_valid['AgeAtSale'] = df_valid['saleyear'] - df_valid['YearMade']

# drop unnecessary columns
df.drop(['saledate', 'YearMade', 'saleyear'], axis=1, inplace=True)
df_valid.drop(['saledate', 'YearMade', 'saleyear'], axis=1, inplace=True)
#endregion

#region Usage_band handling
df = missing_values_imputation(df, 'UsageBand', USAGE_FIT_COLUMNS)
df_valid = missing_values_imputation(df_valid, 'UsageBand', USAGE_FIT_COLUMNS)
#endregion

df.to_csv('Data\\df_PreSplit.csv', index=False)

#region convert 'MachineID' column into 'TimesAppearing' column
df['TimesAppearing'] = df['MachineID'].map(df['MachineID'].value_counts())
df.drop('MachineID', axis=1, inplace=True)
df_valid['TimesAppearing'] = df_valid['MachineID'].map(df_valid['MachineID'].value_counts())
df_valid.drop('MachineID', axis=1, inplace=True)
#endregion

#region Replace NaN with 0 in 'ID' columns
df[['datasource', 'auctioneerID']] = df[['datasource', 'auctioneerID']].fillna(0.0)
df_valid[['datasource', 'auctioneerID']] = df_valid[['datasource', 'auctioneerID']].fillna(0.0)
#endregion

#region Dropping
# drop unimportant columns
df.drop(KILL_THEM_COLUMNS, axis=1, inplace=True)
df_valid.drop(KILL_THEM_COLUMNS, axis=1, inplace=True)

# drop duplicated columns
df.drop(DUPLICATE_COLUMNS, axis=1, inplace=True)
df_valid.drop(DUPLICATE_COLUMNS, axis=1, inplace=True)
#endregion

#region Split train and test data + log normalization of the target column
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['SalePrice']), df['SalePrice'], test_size=0.3, random_state=42)

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
#endregion

#region Get and use statistics dataframe
statistics_df = compute_statistics(X_train)

X_train, X_test, df_valid = replace_outliers([X_train, X_test, df_valid],
                                             'MachineHoursCurrentMeter', 'iqr', statistics_df)

X_train, X_test, df_valid = replace_outliers([X_train, X_test, df_valid],
                                             'AgeAtSale', 'iqr', statistics_df)

X_train, X_test, df_valid = replace_nans([X_train, X_test, df_valid],
                                         'MachineHoursCurrentMeter',
                                         'mean_without_extremes', statistics_df)
#endregion

#region Applying encodings
# label encoding
X_train, X_test, df_valid = apply_lable_encoding([X_train, X_test, df_valid],
                                                 LABEL_ENCODING_COLUMNS)

# one-hot encoding
X_train_transformed, X_test_transformed, X_valid_transformed = (
    apply_one_hot_encoder([X_train, X_test, df_valid],
                          X_train.select_dtypes(exclude=['number']).columns.tolist(),
                          LABEL_ENCODING_COLUMNS))
#endregion

#region Train model
model, results_df = train_and_evaluate_model(X_train_transformed, X_test_transformed, y_train, y_test, TREE_DEPTH,
                                             LEVEL_OF_PARALLELISM, NUMBER_OF_TREES, MAX_FEATURES, MIN_SAMPLES_SPLIT,
                                             MIN_SAMPLES_LEAF)
#endregion

X_train_transformed.to_csv('Data\\X_train_transformed.csv', index=False)
X_test_transformed.to_csv('Data\\X_test_transformed.csv', index=False)
X_valid_transformed.to_csv('Data\\X_valid_transformed.csv', index=False)

#region Feature importance
#print(results_df)
feature_importance = model.feature_importances_
feature_names = X_train_transformed.columns
importance_list = [(importance, feature) for feature, importance in zip(feature_names, feature_importance)]
importance_list.sort(reverse=True)
print('-------Feature Importance-------')
for importance, feature in importance_list:
    print(f"{feature}: {importance}")
#endregion

#region Use Model
y_valid_log_pred = model.predict(X_valid_transformed)
y_valid_pred = np.expm1(y_valid_log_pred);

# Create the prediction DataFrame with only 'SalesID' and 'Predicted_SalePrice'
df_predictions = pd.DataFrame({
    'SalesID': df_valid['SalesID'].astype(int),  # Ensure SalesID is included in df_valid
    'SalePrice': y_valid_pred
})

df_predictions.to_csv('valid_predictions_0.2.csv', index=False)
#endregion
