import pandas as pd
from sklearn.model_selection import train_test_split
from data_functions import load_and_extract_data, preprocess_data, handle_missing_values, encode_categorical_variables
from model_functions import train_and_evaluate_model

# Constants
LEVEL_OF_PARALLELISM = -1
NUMBER_OF_TREES = 100
TREE_DEPTH = [16]
MIN_SAMPLES_SPLIT = 3
MIN_SAMPLES_LEAF = 3
MAX_FEATURES = 0.7
BOOTSTRAP = False
DATA_PATH = 'Data\\train.zip'
EXTRACT_PATH = 'Data'
OUTPUT_PATH = 'Data\\fixed.csv'
DROPPED_COLUMNS = ['ProductGroupDesc','Forks']
TARGET_COLUMN = 'SalePrice'

df = load_and_extract_data(DATA_PATH, EXTRACT_PATH)
df = preprocess_data(df, DROPPED_COLUMNS)

target = df[TARGET_COLUMN]
features = df.drop(columns=[TARGET_COLUMN])
features = encode_categorical_variables(features)

features.to_csv(OUTPUT_PATH, index=False)
features.info()
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=100)

model = train_and_evaluate_model(X_train, X_test, y_train, y_test, TREE_DEPTH, LEVEL_OF_PARALLELISM, NUMBER_OF_TREES,
                               MIN_SAMPLES_LEAF, MIN_SAMPLES_SPLIT, MAX_FEATURES, BOOTSTRAP)

# Use model on the validation data
df_valid = pd.read_csv('Data\\valid.csv')
df_valid = preprocess_data(df_valid, DROPPED_COLUMNS)
df_valid = encode_categorical_variables(df_valid)

X_valid = df_valid
y_valid_pred = model.predict(X_valid)

# Create the prediction DataFrame with only 'SalesID' and 'Predicted_SalePrice'
df_predictions = pd.DataFrame({
    'SalesID': df_valid['SalesID'],
    'SalePrice': y_valid_pred
})

# Save predictions
df_valid.to_csv('Data\\df_valid.csv', index=False)
df_predictions.to_csv('Data\\valid_predictions.csv', index=False)