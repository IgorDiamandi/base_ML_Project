import pandas as pd
from sklearn.model_selection import train_test_split
from data_functions import compute_statistics, replace_outliers, \
    apply_one_hot_encoder, unite_sparse_columns, missing_values_imputation, \
    split_fiProductClassDesc
from model_functions import train_and_evaluate_model

# Constants
LEVEL_OF_PARALLELISM = -1
NUMBER_OF_TREES = 100
TREE_DEPTH = [17]
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = 0.8

KILL_THEM_COLUMNS = ['uControl_any_non_null', 'uMounting_any_non_null', 'uHydraulics_non_null_count',
                     'uControl_non_null_count', 'fiProductClassDesc']

SIZE_FIT_COLUMNS = ['fiModelDesc', 'post_fi_Horsepower', 'Drive_System', 'Stick_Length', 'Undercarriage_Pad_Width']

USAGE_FIT_COLUMNS = ['age', 'MachineHoursCurrentMeter']

DUPLICATE_COLUMNS = ['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor']
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

df = pd.read_csv('Data\\train.csv', dtype=DTYPE_SPEC)
df_valid = pd.read_csv('Data\\valid.csv', dtype=DTYPE_SPEC)

# product size handling
df, df_valid = split_fiProductClassDesc([df,df_valid], 'fiProductClassDesc')


df, df_valid = missing_values_imputation([df,df_valid], 'ProductSize', SIZE_FIT_COLUMNS)

# grouping columns with technical details
for new_column, columns in COLUMN_GROUPS.items():
    df, df_valid = unite_sparse_columns([df, df_valid], columns, new_column)

# drop unimportant columns
df.drop(KILL_THEM_COLUMNS, axis=1, inplace=True)
df_valid.drop(KILL_THEM_COLUMNS, axis=1, inplace=True)

# drop duplicated columns
df.drop(DUPLICATE_COLUMNS, axis=1, inplace=True)
df_valid.drop(DUPLICATE_COLUMNS, axis=1, inplace=True)

# convert dates to periods
df['saleyear'] = pd.to_datetime(df['saledate']).dt.year
df_valid['saleyear'] = pd.to_datetime(df_valid['saledate']).dt.year

df['salemonth'] = pd.to_datetime(df['saledate']).dt.month
df_valid['salemonth'] = pd.to_datetime(df_valid['saledate']).dt.month

df['saledayofweek'] = pd.to_datetime(df['saledate']).dt.dayofweek
df_valid['saledayofweek'] = pd.to_datetime(df_valid['saledate']).dt.dayofweek

df['age'] = 2024 - df['YearMade']
df_valid['age'] = 2024 - df_valid['YearMade']

df['AgeAtSale'] = df['saleyear'] - df['YearMade']
df_valid['AgeAtSale'] = df_valid['saleyear'] - df_valid['YearMade']

# drop unnecessary columns
df.drop(['saledate', 'YearMade'], axis=1, inplace=True)
df_valid.drop(['saledate', 'YearMade'], axis=1, inplace=True)

# usage_band handling
df, df_valid = missing_values_imputation([df,df_valid], 'UsageBand', USAGE_FIT_COLUMNS)

df.to_csv('Data\\df_PreSplit.csv', index=False)

# convert 'MachineID' column into 'TimesAppearing' column
df['TimesAppearing'] = df['MachineID'].map(df['MachineID'].value_counts())
df.drop('MachineID', axis=1, inplace=True)
df_valid['TimesAppearing'] = df_valid['MachineID'].map(df_valid['MachineID'].value_counts())
df_valid.drop('MachineID', axis=1, inplace=True)

# replace NaN with 0 in 'ID' columns
df[['datasource', 'auctioneerID']] = df[['datasource', 'auctioneerID']].fillna(0.0)
df_valid[['datasource', 'auctioneerID']] = df_valid[['datasource', 'auctioneerID']].fillna(0.0)

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['SalePrice']), df['SalePrice'], test_size=0.3, random_state=42)

# get statistics dataframe using compute_statistics function from the X_train
statistics_df = compute_statistics(X_train)

# Use replace_outliers function to replace outliers
X_train, X_test, df_valid = replace_outliers([X_train, X_test, df_valid],
                                             'MachineHoursCurrentMeter', 'iqr', statistics_df)

X_train, X_test, df_valid = replace_outliers([X_train, X_test, df_valid],
                                             'AgeAtSale', 'iqr', statistics_df)

X_train, X_test, df_valid = replace_outliers([X_train, X_test, df_valid],
                                             'age', 'iqr', statistics_df)

X_train, X_test, df_valid = replace_outliers([X_train, X_test, df_valid],
                                             'MachineHoursCurrentMeter', 'iqr', statistics_df)


X_train.to_csv('Data\\X_train.csv', index=False)


# Apply one-hot encoding
X_train_transformed, X_test_transformed, X_valid_transformed = (
    apply_one_hot_encoder([X_train, X_test, df_valid],
                          X_train.select_dtypes(exclude=['number']).columns.tolist()))

# Train model
model,results_df = train_and_evaluate_model(X_train_transformed, X_test_transformed, y_train, y_test, TREE_DEPTH,
                                 LEVEL_OF_PARALLELISM, NUMBER_OF_TREES, MAX_FEATURES, MIN_SAMPLES_SPLIT,
                                 MIN_SAMPLES_LEAF)

X_train_transformed.to_csv('Data\\X_train_transformed.csv', index=False)
X_test_transformed.to_csv('Data\\X_test_transformed.csv', index=False)
X_valid_transformed.to_csv('Data\\X_valid_transformed.csv', index=False)

#print(results_df)

# Print feature importance order in descending order
#feature_importance = model.feature_importances_
#feature_names = X_train_transformed.columns
#importance_list = [(importance, feature) for feature, importance in zip(feature_names, feature_importance)]
#importance_list.sort(reverse=True)
#print('-------Feature Importance-------')
#for importance, feature in importance_list:
#    print(f"{feature}: {importance}")

# Use Model
y_valid_pred = model.predict(X_valid_transformed)

# Create the prediction DataFrame with only 'SalesID' and 'Predicted_SalePrice'
df_predictions = pd.DataFrame({
    'SalesID': df_valid['SalesID'].astype(int),  # Ensure SalesID is included in df_valid
    'SalePrice': y_valid_pred
})

df_predictions.to_csv('valid_predictions_0.0005.csv', index=False)