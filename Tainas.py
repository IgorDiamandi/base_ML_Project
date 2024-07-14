import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data_functions import (
    compute_statistics, replace_outliers, unite_sparse_columns, missing_values_imputation,
    split_fiProductClassDesc, apply_one_hot_encoder, replace_nans, apply_lable_encoding,
    categorize_stick_length, fill_missing_with_mode, handle_missing_values
)
from model_functions import train_and_evaluate_model

# Constants
LABLE_ENCODING_COLUMNS = ['state', 'fiModelDesc', 'ProductSize', 'ProductGroup', 'Enclosure', 'Transmission',
                          'Blade_Width', 'Hydraulics', 'Tire_Size', 'Undercarriage_Pad_Width', 'Blade_Type',
                          'Travel_Controls', 'Steering_Controls']

DTYPE_SPEC = {13: 'str', 39: 'str', 40: 'str', 41: 'str'}

COLUMNS_TO_TRANSFORM = ['Forks', 'Ride_Control', 'Turbocharged', 'Blade_Extension']
COLUMNS_TO_FILL = ['Blade_Width', 'Enclosure_Type', 'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control',
                   'Tire_Size', 'Coupler', 'Track_Type', 'Grouser_Type', 'Undercarriage_Pad_Width',
                   'Thumb', 'Pattern_Changer', 'Travel_Controls']
VARIABLES_TO_ENCODE = ['UsageBand', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 'Turbocharged',
                       'Blade_Extension', 'Enclosure_Type', 'Engine_Horsepower', 'Pushblock', 'Ripper',
                       'Scarifier', 'Tip_Control', 'Coupler', 'Track_Type', 'Stick_Length', 'Thumb',
                       'Pattern_Changer', 'Grouser_Type', 'Differential_Type']
IRRELEVANT_FEATURES = ['MachineHoursCurrentMeter', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries',
                       'fiModelDescriptor', 'fiModelSeries', 'saledate', 'ProductGroupDesc', 'Drive_System',
                       'Coupler_System', 'Hydraulics_Flow', 'Grouser_Tracks', 'Backhoe_Mounting', 'fiProductClassDesc']
COLUMNS_TO_FILL_WITH_MODE = ['auctioneerID', 'Enclosure', 'Pad_Type', 'Stick', 'Transmission', 'Hydraulics',
                             'Differential_Type', 'Steering_Controls']


# Functions
def preprocess_data(df, mode_value):
    df = df.copy()
    df['UsageBand'] = df['UsageBand'].fillna('Medium')
    df['YearMade'] = df['YearMade'].replace(1000, mode_value)
    df['saledate'] = pd.to_datetime(df['saledate'])
    df['SaleYear'] = df['saledate'].dt.year
    df['SaleMonth'] = df['saledate'].dt.month
    df['Age'] = df['SaleYear'] - df['YearMade']

    # Debugging statement to check if 'Age' column is created correctly
    if 'Age' not in df.columns:
        print("Error: 'Age' column not created")
    else:
        print(f"'Age' column created with {df['Age'].isnull().sum()} missing values")

    df['Age'] = df['Age'].apply(lambda x: mode_value if x < 0 else x)

    for column in COLUMNS_TO_TRANSFORM:
        df[column] = df[column].apply(lambda x: 'Yes' if x == 'Yes' else 'No')

    df['Transmission'] = df['Transmission'].replace('AutoShift', 'Autoshift')

    for column in COLUMNS_TO_FILL:
        df[column] = df[column].fillna('None or Unspecified')

    df['Tire_Size'] = df['Tire_Size'].str.replace('"', '').str.replace(' inch', '').str.strip()
    df['Stick_Length'] = df['Stick_Length'].apply(categorize_stick_length)

    df['Blade_Type'] = df['Blade_Type'].fillna('No')
    df['Engine_Horsepower'] = df['Engine_Horsepower'].fillna('No')

    return df


def fill_missing_with_mean(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df


def calculate_mode_value(df, column):
    return df.loc[df[column] != 1000, column].mode()[0]


def encode_variables(train, valid, variables):
    train_encoded = pd.get_dummies(train[variables], drop_first=True)
    valid_encoded = pd.get_dummies(valid[variables], drop_first=True)
    valid_encoded = valid_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    train = train.drop(columns=variables)
    valid = valid.drop(columns=variables)

    train = pd.concat([train, train_encoded], axis=1)
    valid = pd.concat([valid, valid_encoded], axis=1)

    return train, valid


def ensure_numeric(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col], _ = pd.factorize(df[col])
    return df


# Load data
train = pd.read_csv('Data\\train.csv', dtype=DTYPE_SPEC)
valid = pd.read_csv('Data\\valid.csv', dtype=DTYPE_SPEC)

train = train.set_index('SalesID')
valid = valid.set_index('SalesID')

# Calculate mode values
mode_value = calculate_mode_value(train, 'YearMade')

# Preprocess data
train = preprocess_data(train, mode_value)
valid = preprocess_data(valid, mode_value)

# Calculate mode value for Age after creating 'Age' column
mode_value_age = train.loc[train['Age'] >= 0, 'Age'].mode()[0]

# Drop irrelevant features
train = train.drop(columns=[col for col in IRRELEVANT_FEATURES if col in train.columns])
valid = valid.drop(columns=[col for col in IRRELEVANT_FEATURES if col in valid.columns])

# Fill missing values with mode
train, valid = fill_missing_with_mode(train, valid, COLUMNS_TO_FILL_WITH_MODE)

# Handle remaining missing values
train = handle_missing_values(train)
valid = handle_missing_values(valid)

# Apply label encoding
train_encoded, valid_encoded = apply_lable_encoding([train, valid], LABLE_ENCODING_COLUMNS)

# One-hot encode variables
train_encoded, valid_encoded = encode_variables(train_encoded, valid_encoded, VARIABLES_TO_ENCODE)

# Ensure all columns are numeric
train_encoded = ensure_numeric(train_encoded)
valid_encoded = ensure_numeric(valid_encoded)

# Split the data
X = train_encoded.drop(columns=['SalePrice'])
y = train_encoded['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# Evaluation
def RMSE(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


print(f'Train RMSE:', RMSE(y_train, y_train_pred))
print(f'Test RMSE:', RMSE(y_test, y_test_pred))

# Feature importance
feature_importance = model.feature_importances_
feature_names = X_train.columns
importance_list = sorted(zip(feature_importance, feature_names), reverse=True)
print('-------Feature Importance-------')
for importance, feature in importance_list:
    print(f"{feature}: {importance}")

# Validate
valid = valid.reset_index()
X_valid = valid[X.columns]
X_valid = fill_missing_with_mean(X_valid, X_valid.columns)
y_valid_pred = model.predict(X_valid)
y_valid_pred = pd.Series(y_valid_pred, index=X_valid.index, name='SalePrice')
print(y_valid_pred)
