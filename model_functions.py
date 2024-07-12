from sklearn.ensemble import RandomForestRegressor
from data_functions import get_rmse


def train_and_evaluate_model(X_train, X_test, y_train, y_test, tree_depth, level_of_parallelism, number_of_trees,
                             max_features, min_samples_split, min_samples_leaf):
    for depth in tree_depth:
        model = RandomForestRegressor(
                random_state=100,
                n_jobs=level_of_parallelism,
                n_estimators=number_of_trees,
                max_depth=depth,
                max_features=max_features,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf)


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

    return model