from datetime import datetime
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from data_functions import handle_outliers_iqr, handle_outliers_zscore, replace_outliers_with_mean, \
    remove_columns_with_many_nulls, replace_null_with_mean
from model_functions import train_and_evaluate_model

# Constants
LEVEL_OF_PARALLELISM = 20
NUMBER_OF_TREES = 20
TREE_DEPTH = [20]
DATA_PATH = 'Data\\train.zip'
EXTRACT_PATH = 'Data'
OUTPUT_PATH = 'Data\\fixed.csv'
DROPPED_COLUMNS = ['Forks', 'Ride_Control', 'Transmission', 'Coupler']
DTYPE_SPEC = {
    'fiModelDescriptor': 'str',
    'Hydraulics_Flow': 'str',
    'Track_Type': 'str',
    'Undercarriage_Pad_Width': 'str'
}
TARGET_COLUMN = 'SalePrice'


def load_and_extract_data(zip_path, extract_path):
    print('Loading data...')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.printdir()
        z.extractall(extract_path)
        with z.open('train.csv') as f:
            df = pd.read_csv(f, dtype=DTYPE_SPEC, low_memory=False)
    return df


def preprocess_data(df):
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
    df = df.drop(DROPPED_COLUMNS, axis=1)
    df = remove_columns_with_many_nulls(df, 0.7)

    return df


def handle_missing_values(features):
    numeric_cols = features.select_dtypes(include=['number']).columns
    non_numeric_cols = features.select_dtypes(exclude=['number']).columns

    features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())
    features[non_numeric_cols] = features[non_numeric_cols].fillna(features[non_numeric_cols].mode().iloc[0])

    return features


def encode_categorical_variables(features):
    categorical_cols = features.select_dtypes(include=['object', 'category']).columns

    # Convert date columns to separate year, month, and day columns
    date_cols = [col for col in features.columns if 'date' in col.lower()]
    for col in date_cols:
        features[col] = pd.to_datetime(features[col], errors='coerce')
        features[col + '_year'] = features[col].dt.year
        features[col + '_month'] = features[col].dt.month
        features[col + '_day'] = features[col].dt.day
        features.drop(columns=[col], inplace=True)

    return features


df = load_and_extract_data(DATA_PATH, EXTRACT_PATH)
df = preprocess_data(df)
df.to_csv(OUTPUT_PATH, index=False)

target = df[TARGET_COLUMN]
features = df.drop(columns=[TARGET_COLUMN])
features = handle_missing_values(features)
features = encode_categorical_variables(features)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

train_and_evaluate_model(X_train, X_test, y_train, y_test, TREE_DEPTH, LEVEL_OF_PARALLELISM, NUMBER_OF_TREES)