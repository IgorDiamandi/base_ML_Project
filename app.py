import pandas as pd
from sklearn.model_selection import train_test_split
from data_functions import split_product_class_series, compute_statistics, replace_outliers, replace_nans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from model_functions import train_and_evaluate_model

# Constants
LEVEL_OF_PARALLELISM = -1
NUMBER_OF_TREES = 20
TREE_DEPTH = 10
MIN_SAMPLES_SPLIT = 3
MIN_SAMPLES_LEAF = 3
MAX_FEATURES = 0.5
DATA_PATH = 'Data\\train.zip'
EXTRACT_PATH = 'Data'
OUTPUT_PATH = 'Data\\fixed.csv'
DROPPED_COLUMNS = ['']
IQR_COLUMNS = ['SalesPrice','MachineHoursCurrentMeter']
MEAN_COLUMNS = ['']
ZSCORE_COLUMNS = ['']
TARGET_COLUMN = 'SalePrice'

DTYPE_SPEC = {
    13: 'str',
    39: 'str',
    40: 'str',
    41: 'str'
    }

df = pd.read_csv('Data\\train.csv', dtype=DTYPE_SPEC)
df_valid = pd.read_csv('Data\\valid.csv', dtype=DTYPE_SPEC)

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

# replace NaN with -1 in 'ID' columns
df[['datasource', 'auctioneerID']] = df[['datasource', 'auctioneerID']].fillna(-1)
df_valid[['datasource', 'auctioneerID']] = df_valid[['datasource', 'auctioneerID']].fillna(-1)

#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['SalePrice']), df['SalePrice'], test_size=0.3, random_state=100)

# get statistics dataframe using compute_statistics function from the X_train
statistics_df = compute_statistics(X_train)

""" Use replace_outliers function in order to replace outliers 
in the 'MachineHoursCurrentMeter' column using statistics_df and IQR method """
X_train = replace_outliers(X_train, 'MachineHoursCurrentMeter', 'iqr', statistics_df)
X_test = replace_outliers(X_test, 'MachineHoursCurrentMeter', 'iqr', statistics_df)
df_valid = replace_outliers(df_valid, 'MachineHoursCurrentMeter', 'iqr', statistics_df)

""" Use replace_outliers function in order to replace outliers 
in the 'Age' column using statistics_df and mean method """
X_train = replace_outliers(X_train, 'Age', 'mean', statistics_df)
X_test = replace_outliers(X_test, 'Age', 'mean', statistics_df)
df_valid = replace_outliers(df_valid, 'Age', 'mean', statistics_df)

""" Use replace_nans function in order to replace NaN values
in the 'MachineHoursCurrentMeter' column using statistics_df and IQR method """
X_train = replace_nans(X_train, 'MachineHoursCurrentMeter', 'iqr', statistics_df)
X_test = replace_nans(X_test, 'MachineHoursCurrentMeter', 'iqr', statistics_df)
df_valid = replace_nans(df_valid, 'MachineHoursCurrentMeter', 'iqr', statistics_df)

# replace NaN with 'Missing' in textual columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        if X_train[col].apply(type).nunique() > 1:
            X_train[col] = X_train[col].fillna('Missing').astype(str)
        X_train[col] = le.fit_transform(X_train[col])

for col in X_test.columns:
    if X_test[col].dtype == 'object':
        if X_test[col].apply(type).nunique() > 1:
            X_test[col] = X_test[col].fillna('Missing').astype(str)
        X_test[col] = le.fit_transform(X_test[col])


for col in df_valid.columns:
    if df_valid[col].dtype == 'object':
        if df_valid[col].apply(type).nunique() > 1:
            df_valid[col] = df_valid[col].fillna('Missing').astype(str)
        df_valid[col] = le.fit_transform(df_valid[col])


# convert categorical values to lables in all 3 dataframes
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
# Create the column transformer with OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
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

# Use Model
y_valid_pred = model.predict(X_valid_transformed)

# Create the prediction DataFrame with only 'SalesID' and 'Predicted_SalePrice'
df_predictions = pd.DataFrame({
    'SalesID': X_valid_transformed['SalesID'],
    'SalePrice': y_valid_pred
})
df_predictions.to_csv('valid_predictions.csv', index=False)

# print features importance order it descending
feature_importance = model.feature_importances_
feature_names = X_train.columns
importance_list = [(importance, feature) for feature, importance in zip(feature_names, feature_importance)]
importance_list.sort(reverse=True)
for importance, feature in importance_list:
    print(f"{feature}: {importance}")