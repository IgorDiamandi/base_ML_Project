import pandas as pd
from sklearn.model_selection import train_test_split
from data_functions import compute_statistics, replace_outliers, replace_nans, \
    replace_nan_with_string, apply_one_hot_encoder, unite_sparse_columns, missing_values_imputation, \
    split_fiProductClassDesc
from model_functions import train_and_evaluate_model

# Constants
LEVEL_OF_PARALLELISM = -1
NUMBER_OF_TREES = 100
TREE_DEPTH = 8
MIN_SAMPLES_SPLIT = 3
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = 0.9

KILL_THEM_COLUMNS = ['uControl_any_non_null', 'uMounting_any_non_null', 'uHydraulics_non_null_count',
                     'uControl_non_null_count', 'fiProductClassDesc']

SIZE_FIT_COLUMNS = ['fiModelDesc', 'post_fi_Horsepower', 'Drive_System', 'Stick_Length', 'Undercarriage_Pad_Width',
                    'Pad_Type', 'Differential_Type']

DUPLICATE_COLUMNS = ['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor']
COLUMN_GROUPS = {
    'uBladeStick': ['Blade_Extension', 'Stick_Length', 'Stick',], #'Blade_Type' 'Blade_Width'
    'uTrack': ['Pad_Type', 'Grouser_Type', 'Grouser_Tracks'], #'Track_Type' 'Undercarriage_Pad_Width'
    'uMounting': ['Backhoe_Mounting', 'Forks', 'Pushblock', 'Ripper', 'Scarifier', 'Thumb'],
    'uControl': ['Travel_Controls', 'Ride_Control', 'Transmission',
                 'Pattern_Changer', 'Tip_Control', 'Coupler', 'Coupler_System'], #'Steering_Controls'
    'uHydraulics': ['Turbocharged'], #'Hydraulics' 'Hydraulics_Flow'
    'uDrive': ['Differential_Type', 'Drive_System']
    }


DTYPE_SPEC = {
    13: 'str',
    39: 'str',
    40: 'str',
    41: 'str'}

df = pd.read_csv('Data\\train.csv', dtype=DTYPE_SPEC)
df_valid = pd.read_csv('Data\\valid.csv', dtype=DTYPE_SPEC)


# product size struggle
df = split_fiProductClassDesc(df, 'fiProductClassDesc')
df_valid = split_fiProductClassDesc(df_valid, 'fiProductClassDesc')

df = missing_values_imputation(df, 'ProductSize', SIZE_FIT_COLUMNS)
df_valid = missing_values_imputation(df_valid, 'ProductSize', SIZE_FIT_COLUMNS)



# grouping columns with technical details
for new_column, columns in COLUMN_GROUPS.items():
    df = unite_sparse_columns(df, columns, new_column)
    df_valid = unite_sparse_columns(df_valid, columns, new_column)

# drop unimportant columns
df.drop(KILL_THEM_COLUMNS, axis=1, inplace=True)
df_valid.drop(KILL_THEM_COLUMNS, axis=1, inplace=True)

# drop duplicated columns
df.drop(DUPLICATE_COLUMNS, axis=1, inplace=True)
df_valid.drop(DUPLICATE_COLUMNS, axis=1, inplace=True)

# convert dates to periods
df['saleyear'] = pd.to_datetime(df['saledate']).dt.year
df_valid['saleyear'] = pd.to_datetime(df_valid['saledate']).dt.year
df['age'] = 2024 - df['YearMade']
df_valid['age'] = 2024 - df_valid['YearMade']
df['AgeAtSale'] = df['saleyear'] - df['YearMade']
df_valid['AgeAtSale'] = df_valid['saleyear'] - df_valid['YearMade']
# convert 'YearMade' column to 'Age'
df.drop(['saledate', 'saleyear', 'YearMade'], axis=1, inplace=True)
df_valid.drop(['saledate', 'saleyear', 'YearMade'], axis=1, inplace=True)

df.to_csv('Data\\df_PreSplit.csv', index=False)

# split 'fiProductDesc' column
#df['ProductClassName'],df['ProductClassCharacteristic'] = split_product_class_series(df['fiProductClassDesc'])
#df_valid['ProductClassName'],df_valid['ProductClassCharacteristic'] = split_product_class_series(df_valid['fiProductClassDesc'])

# convert 'MachineID' column into 'TimesAppearing' column
df['TimesAppearing'] = df['MachineID'].map(df['MachineID'].value_counts())
df.drop('MachineID', axis=1, inplace=True)
df_valid['TimesAppearing'] = df_valid['MachineID'].map(df_valid['MachineID'].value_counts())
df_valid.drop('MachineID', axis=1, inplace=True)

# replace NaN with 0 in 'ID' columns
df[['datasource', 'auctioneerID']] = df[['datasource', 'auctioneerID']].fillna(0.0)
df_valid[['datasource', 'auctioneerID']] = df_valid[['datasource', 'auctioneerID']].fillna(0.0)

#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['SalePrice']), df['SalePrice'], test_size=0.3, random_state=42)

# get statistics dataframe using compute_statistics function from the X_train
statistics_df = compute_statistics(X_train)

""" Use replace_outliers function in order to replace outliers 
in the 'MachineHoursCurrentMeter' column using statistics_df and IQR method """
X_train = replace_outliers(X_train, 'MachineHoursCurrentMeter', 'mean_without_extremes', statistics_df)
X_test = replace_outliers(X_test, 'MachineHoursCurrentMeter', 'mean_without_extremes', statistics_df)
df_valid = replace_outliers(df_valid, 'MachineHoursCurrentMeter', 'mean_without_extremes', statistics_df)

""" Use replace_outliers function in order to replace outliers 
in the 'Age' columns using statistics_df and mean method """
X_train = replace_outliers(X_train, 'AgeAtSale', 'mean_without_extremes', statistics_df)
X_test = replace_outliers(X_test, 'AgeAtSale', 'mean_without_extremes', statistics_df)
df_valid = replace_outliers(df_valid, 'AgeAtSale', 'mean_without_extremes', statistics_df)

""" Use replace_nans function in order to replace NaN values
in the 'MachineHoursCurrentMeter' column using statistics_df and IQR method """
X_train = replace_nans(X_train, 'MachineHoursCurrentMeter', 'iqr', statistics_df)
X_test = replace_nans(X_test, 'MachineHoursCurrentMeter', 'iqr', statistics_df)
X_valid = replace_nans(df_valid, 'MachineHoursCurrentMeter', 'iqr', statistics_df)

# replace NaN with 'Missing' in textual columns
replace_nan_with_string(X_train)
replace_nan_with_string(X_test)
replace_nan_with_string(df_valid)

#X_train_target, X_test_transformed, X_valid_transformed=(
#    apply_target_encoding(X_train, X_test, X_valid, 'SalePrice', TARGET_ENCODE_COLUMNS))

X_train_transformed, X_test_transformed, X_valid_transformed=(
    apply_one_hot_encoder(X_train, X_test, df_valid, X_train.select_dtypes(exclude=['number']).columns.tolist()))

# Train model
model = train_and_evaluate_model(X_train_transformed, X_test_transformed, y_train, y_test, [TREE_DEPTH],
                                 LEVEL_OF_PARALLELISM, NUMBER_OF_TREES, MAX_FEATURES, MIN_SAMPLES_SPLIT,
                                MIN_SAMPLES_LEAF)

X_train_transformed.to_csv('Data\\X_train_transformed.csv', index=False)
X_test_transformed.to_csv('Data\\X_test_transformed.csv', index=False)
X_valid_transformed.to_csv('Data\\X_valid_transformed.csv', index=False)

# print features importance order it descending
feature_importance = model.feature_importances_
feature_names = X_train.columns
importance_list = [(importance, feature) for feature, importance in zip(feature_names, feature_importance)]
importance_list.sort(reverse=True)
print('-------FeatureImportance-------')
for importance, feature in importance_list:
    print(f"{feature}: {importance}")

# Use Model
y_valid_pred = model.predict(X_valid_transformed)
# Create the prediction DataFrame with only 'SalesID' and 'Predicted_SalePrice'
df_predictions = pd.DataFrame({
    'SalesID': X_valid_transformed['remainder__SalesID'].astype(int),
    'SalePrice': y_valid_pred
})
df_predictions.to_csv('valid_predictions_0.04.csv', index=False)