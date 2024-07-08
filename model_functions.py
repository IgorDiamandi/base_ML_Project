import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data_functions import get_rmse


def create_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    return preprocessor


def get_feature_names(preprocessor):
    numeric_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
    return numeric_features.tolist() + cat_features.tolist()


def train_and_evaluate_model(X_train, X_test, y_train, y_test, tree_depth, level_of_parallelism, number_of_trees,
                             min_samples_split, min_samples_leaf, max_features, bootstrap):
    for depth in tree_depth:
        preprocessor = create_preprocessor(X_train)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                random_state=100,
                n_jobs=level_of_parallelism,
                n_estimators=number_of_trees,
                max_depth=depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                max_features=max_features,
                bootstrap=bootstrap))
        ])

        print('Fitting the model...')
        model.fit(X_train, y_train)

        print('Testing the model...')
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        rmse_test = get_rmse(y_test, y_test_pred)
        rmse_train = get_rmse(y_train, y_train_pred)

        print(f'Tree depth - {depth}')
        print(f'STD Test - {y_test.std()}')
        print(f'STD Train - {y_train.std()}')
        print(f'RMSE Test - {rmse_test}')
        print(f'RMSE Train - {rmse_train}')

        # Feature importance check
        #regressor = model.named_steps['regressor']
        #feature_importances = regressor.feature_importances_
        #feature_names = get_feature_names(preprocessor)
        #feature_importance_df = pd.DataFrame({
        #    'Feature': feature_names,
        #    'Importance': feature_importances
        #}).sort_values(by='Importance', ascending=False)

        #print("Feature Importances:")
        #print(feature_importance_df)

    return model
