import pandas as pd
from sklearn.model_selection import train_test_split
from data_functions import split_product_class_series, compute_statistics, replace_outliers, replace_nans, \
    replace_nan_with_string
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from model_functions import train_and_evaluate_model

# Constants
LEVEL_OF_PARALLELISM = -1
NUMBER_OF_TREES = 100
TREE_DEPTH = 10
MIN_SAMPLES_SPLIT = 4
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = 0.5


DTYPE_SPEC = {
    13: 'str',
    39: 'str',
    40: 'str',
    41: 'str'}

df = pd.read_csv('Data\\train.csv', dtype=DTYPE_SPEC)
df_valid = pd.read_csv('Data\\valid.csv', dtype=DTYPE_SPEC)

df.drop(['Thumb', 'Blade_Width'], axis=1, inplace=True)
df_valid.drop(['Thumb','Blade_Width'], axis=1, inplace=True)

# convert 'YearMade' column to 'Age'
df['Age'] = 2024 - df['YearMade']
df['AgeAtLastSale'] = (pd.to_datetime(df['saledate']) - pd.to_datetime(df['YearMade'].clip(lower=1900), format='%Y')).dt.days
df.drop('YearMade', axis=1, inplace=True)

df_valid['Age'] = 2024 - df_valid['YearMade']
df_valid['AgeAtLastSale'] = (pd.to_datetime(df_valid['saledate']) - pd.to_datetime(df_valid['YearMade'].clip(lower=1900), format='%Y')).dt.days
df_valid.drop('YearMade', axis=1, inplace=True)

# split 'fiProductDesc' column
df['ProductClassName'],df['ProductClassCharacteristic'] = split_product_class_series(df['fiProductClassDesc'])
df_valid['ProductClassName'],df_valid['ProductClassCharacteristic'] = split_product_class_series(df_valid['fiProductClassDesc'])

# convert 'MachineID' column into 'TimesAppearing' column
df['TimesAppearing'] = df['MachineID'].map(df['MachineID'].value_counts())
df.drop('MachineID', axis=1, inplace=True)
df_valid['TimesAppearing'] = df_valid['MachineID'].map(df_valid['MachineID'].value_counts())
df_valid.drop('MachineID', axis=1, inplace=True)

# remove duplicated columns: 'ProductGroupDesc'
df.drop(['ProductGroupDesc','ProductClassName'], axis=1, inplace=True)
df_valid.drop(['ProductGroupDesc','ProductClassName'], axis=1, inplace=True)

# replace NaN with 0 in 'ID' columns
df[['datasource', 'auctioneerID']] = df[['datasource', 'auctioneerID']].fillna(0.0)
df_valid[['datasource', 'auctioneerID']] = df_valid[['datasource', 'auctioneerID']].fillna(0.0)

#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['SalePrice']), df['SalePrice'], test_size=0.3, random_state=100)

# get statistics dataframe using compute_statistics function from the X_train
statistics_df = compute_statistics(X_train)

""" Use replace_outliers function in order to replace outliers 
in the 'MachineHoursCurrentMeter' column using statistics_df and IQR method """
X_train = replace_outliers(X_train, 'MachineHoursCurrentMeter', 'mean_without_extremes', statistics_df)
X_test = replace_outliers(X_test, 'MachineHoursCurrentMeter', 'mean_without_extremes', statistics_df)
df_valid = replace_outliers(df_valid, 'MachineHoursCurrentMeter', 'mean_without_extremes', statistics_df)

""" Use replace_outliers function in order to replace outliers 
in the 'Age' columns using statistics_df and mean method """
X_train = replace_outliers(X_train, 'Age', 'mean_without_extremes', statistics_df)
X_test = replace_outliers(X_test, 'Age', 'mean_without_extremes', statistics_df)
df_valid = replace_outliers(df_valid, 'Age', 'mean_without_extremes', statistics_df)
X_train = replace_outliers(X_train, 'AgeAtLastSale', 'mean_without_extremes', statistics_df)
X_test = replace_outliers(X_test, 'AgeAtLastSale', 'mean_without_extremes', statistics_df)
df_valid = replace_outliers(df_valid, 'AgeAtLastSale', 'mean_without_extremes', statistics_df)

""" Use replace_nans function in order to replace NaN values
in the 'MachineHoursCurrentMeter' column using statistics_df and IQR method """
X_train = replace_nans(X_train, 'MachineHoursCurrentMeter', 'iqr', statistics_df)
X_test = replace_nans(X_test, 'MachineHoursCurrentMeter', 'iqr', statistics_df)
df_valid = replace_nans(df_valid, 'MachineHoursCurrentMeter', 'iqr', statistics_df)

# replace NaN with 'Missing' in textual columns
replace_nan_with_string(X_train)
replace_nan_with_string(X_test)
replace_nan_with_string(df_valid)

# convert categorical values to lables in all 3 dataframes
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
# Create the column transformer with OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='error'), categorical_cols)
    ],
    remainder='passthrough'
)
# Fit the preprocessor on X_train and transform all dataframes
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
X_valid_transformed = preprocessor.transform(df_valid)

# Convert transformed arrays back to dataframes with proper column names
X_train_transformed = pd.DataFrame(X_train_transformed, columns=preprocessor.get_feature_names_out())
X_test_transformed = pd.DataFrame(X_test_transformed, columns=preprocessor.get_feature_names_out())
X_valid_transformed = pd.DataFrame(X_valid_transformed, columns=preprocessor.get_feature_names_out())

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
df_predictions.to_csv('valid_predictions.csv', index=False)