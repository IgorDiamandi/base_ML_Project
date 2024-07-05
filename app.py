from datetime import datetime
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from data_functions import get_rmse, handle_outliers_iqr, handle_outliers_zscore, replace_outliers_with_mean, \
    replace_null_with_mean, remove_columns_with_many_nulls

level_of_parallelism: int = 20
number_of_trees: int = 20
tree_depth = [19]

dtype_spec = {
    'Column13': 'str',
    'Column39': 'str',
    'Column40': 'str',
    'Column41': 'str'
}

print('Loading')

with zipfile.ZipFile('Data\\train.zip', 'r') as z:
    z.printdir()
    z.extractall()

    with z.open('train.csv') as f:
        df = pd.read_csv(f, dtype=dtype_spec, low_memory=False).sample(frac=0.2)

#region Converting MachineId to Sales count indicator
machine_id_counts = df['MachineID'].value_counts().reset_index()
machine_id_counts.columns = ['MachineID', 'MachineID_Count']
df = df.merge(machine_id_counts, on='MachineID', how='left').drop('MachineID', axis=1)
#endregion

#region Converting YearMade to MachineAge and cleaning up corrupted values (1000)
df['MachineAge'] = datetime.now().year - df['YearMade']
df = df.drop('YearMade', axis=1)
df = replace_outliers_with_mean(df, ['MachineAge'], 1)
#endregion

#region Handle Outliers
df = handle_outliers_iqr(df, ['SalePrice', 'MachineHoursCurrentMeter'])
df = replace_null_with_mean(df, ['MachineHoursCurrentMeter'])
#endregion

#region removing columns with manu nulls
remove_columns_with_many_nulls(df, 0.5)
#endregion

# Identifying features and target
target = 'SalePrice'
features = df.drop(columns=[target])
target = df[target]

# Handling missing values for numeric columns
numeric_cols = features.select_dtypes(include=['number']).columns
features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())

# Handling missing values for non-numeric columns
non_numeric_cols = features.select_dtypes(exclude=['number']).columns
features[non_numeric_cols] = features[non_numeric_cols].fillna(features[non_numeric_cols].mode().iloc[0])

# Encoding categorical variables
categorical_cols = features.select_dtypes(include=['object', 'category']).columns

# Converting date columns (example assuming you have date columns)
date_cols = [col for col in features.columns if 'date' in col.lower()]
for col in date_cols:
    features[col] = pd.to_datetime(features[col], errors='coerce')
    features[col + '_year'] = features[col].dt.year
    features[col + '_month'] = features[col].dt.month
    features[col + '_day'] = features[col].dt.day
    features.drop(columns=[col], inplace=True)

# Updating the list of numeric and categorical columns after date transformation
numeric_cols = features.select_dtypes(include=['number']).columns
categorical_cols = features.select_dtypes(include=['object', 'category']).columns

# Preprocessing dataset
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model pipeline
for number in tree_depth:
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            random_state=42,
            n_jobs=level_of_parallelism,
            n_estimators=number_of_trees,
            max_depth=number))
    ])

    print('Fitting')
    # Training the model
    model.fit(X_train, y_train)

    print('Testing')
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluation
    print(f'Tree depth - {number}')
    print(f'STD Test - {y_test.std()}')
    print(f'STD Train - {y_train.std()}')
    print(f'RMSE Test - {get_rmse(y_test, y_test_pred)}')
    print(f'RMSE Train - {get_rmse(y_train, y_train_pred)}')

    # Extracting the RandomForestRegressor from the pipeline to access feature importances
    regressor = model.named_steps['regressor']
    feature_importances = regressor.feature_importances_

    # Creating a DataFrame for feature importances
    feature_names = numeric_cols.tolist() + model.named_steps['preprocessor']\
                                          .transformers_[1][1].get_feature_names_out(categorical_cols).tolist()
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print("Feature Importances:")
    print(feature_importance_df)
