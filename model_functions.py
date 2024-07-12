from sklearn.ensemble import RandomForestRegressor
from data_functions import get_rmse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)


def train_and_evaluate_model(X_train, X_test, y_train, y_test, tree_depth, level_of_parallelism, number_of_trees,
                             max_features, min_samples_split, min_samples_leaf):
    results = []

    for depth in tree_depth:
        model = RandomForestRegressor(
            random_state=100,
            n_jobs=level_of_parallelism,
            n_estimators=number_of_trees,
            max_depth=depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        logging.info('Fitting the model...')
        model.fit(X_train, y_train)

        logging.info('Testing the model...')
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        rmse_test = get_rmse(y_test, y_test_pred)
        rmse_train = get_rmse(y_train, y_train_pred)

        result = {
            'Tree Depth': depth,
            'STD Test': y_test.std(),
            'STD Train': y_train.std(),
            'RMSE Test': rmse_test,
            'RMSE Train': rmse_train,
            'Test/Train Ratio': 1 - rmse_train / rmse_test
        }
        results.append(result)

        logging.info(f'Tree depth - {depth}')
        logging.info(f'STD Test - {y_test.std()}')
        logging.info(f'STD Train - {y_train.std()}')
        logging.info(f'RMSE Test - {rmse_test}')
        logging.info(f'RMSE Train - {rmse_train}')
        logging.info(f'Test/Train Ratio - {1 - rmse_train / rmse_test}')

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    return model, results_df
