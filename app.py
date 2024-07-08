import pandas as pd
from sklearn.model_selection import train_test_split
from data_functions import load_and_extract_data, preprocess_data, handle_missing_values, encode_categorical_variables, \
    preprocess_validation_data
from model_functions import train_and_evaluate_model

# Constants
LEVEL_OF_PARALLELISM = 20
NUMBER_OF_TREES = 20
TREE_DEPTH = [17]
MIN_SAMPLES_SPLIT = 3
MIN_SAMPLES_LEAF = 3
MAX_FEATURES = 0.8
DATA_PATH = 'Data\\train.zip'
EXTRACT_PATH = 'Data'
OUTPUT_PATH = 'Data\\fixed.csv'
DROPPED_COLUMNS = ['Forks', 'Ride_Control', 'Transmission', 'Coupler']
TARGET_COLUMN = 'SalePrice'

df = load_and_extract_data(DATA_PATH, EXTRACT_PATH)
df = preprocess_data(df, DROPPED_COLUMNS)
df.to_csv(OUTPUT_PATH, index=False)

target = df[TARGET_COLUMN]
features = df.drop(columns=[TARGET_COLUMN])
features = handle_missing_values(features)
features = encode_categorical_variables(features)

features.to_csv(OUTPUT_PATH, index=False)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=100)

model = train_and_evaluate_model(X_train, X_test, y_train, y_test, TREE_DEPTH, LEVEL_OF_PARALLELISM, NUMBER_OF_TREES,
                                 MIN_SAMPLES_LEAF, MIN_SAMPLES_SPLIT, MAX_FEATURES)

#Use model on the validation data
df_valid = pd.read_csv('Data\\valid.csv')
df_valid = preprocess_validation_data(df_valid, DROPPED_COLUMNS)
df_valid = handle_missing_values(df_valid)
df_valid = encode_categorical_variables(df_valid)

X_valid = df_valid
y_valid_pred = model.predict(X_valid)

# Create the prediction DataFrame with only 'SalesID' and 'Predicted_SalePrice'
df_predictions = pd.DataFrame({
    'SalesID': df_valid['SalesID'],
    'SalePrice': y_valid_pred
})

# Save predictions
df_predictions.to_csv('Data\\valid_predictions.csv', index=False)
