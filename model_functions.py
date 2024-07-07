from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from app import LEVEL_OF_PARALLELISM, NUMBER_OF_TREES
from data_functions import get_rmse


def create_preprocessor(features):
    numeric_cols = features.select_dtypes(include=['number']).columns
    categorical_cols = features.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    return preprocessor


def train_and_evaluate_model(X_train, X_test, y_train, y_test, tree_depth):
    for depth in tree_depth:
        model = Pipeline(steps=[
            ('preprocessor', create_preprocessor(X_train)),
            ('regressor', RandomForestRegressor(
                random_state=100,
                n_jobs=LEVEL_OF_PARALLELISM,
                n_estimators=NUMBER_OF_TREES,
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
