from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from data_functions import get_rmse
import pandas as pd


def create_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    return preprocessor


def train_and_evaluate_model(X_train, X_test, y_train, y_test, tree_depth, level_of_parallelism, number_of_trees):
    for depth in tree_depth:
        model = Pipeline(steps=[
            ('preprocessor', create_preprocessor(X_train)),
            ('regressor', RandomForestRegressor(
                random_state=100,
                n_jobs=level_of_parallelism,
                n_estimators=number_of_trees,
                max_depth=depth))
        ])

        print('Fitting the model...')
        model.fit(X_train, y_train)

        print('Testing the model...')
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        print(f'Tree depth - {depth}')
        print(f'STD Test - {y_test.std()}')
        print(f'STD Train - {y_train.std()}')
        print(f'RMSE Test - {get_rmse(y_test, y_test_pred)}')
        print(f'RMSE Train - {get_rmse(y_train, y_train_pred)}')

        # Extracting the RandomForestRegressor from the pipeline to access feature importances
        regressor = model.named_steps['regressor']
        feature_importances_tree = regressor.feature_importances_

        # Calculate permutation importances
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

        # Getting the feature names from the preprocessor
        preprocessor = model.named_steps['preprocessor']
        numeric_features = preprocessor.transformers_[0][2]
        categorical_features = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names = list(numeric_features) + list(categorical_features)

        # Check lengths of feature importances arrays
        if len(feature_names) != len(feature_importances_tree) or len(feature_names) != len(result.importances_mean):
            raise ValueError("Mismatch between feature names and feature importances lengths.")

        # Creating a DataFrame for feature importances
        feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'Permutation Importance': result.importances_mean,
            'Tree-based Importance': feature_importances_tree
        }).sort_values(by='Permutation Importance', ascending=False)

        print("Feature Importances:")
        print(feature_importances)