from sklearn.model_selection import train_test_split


def get_rmse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean() ** 0.5


def train(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    model.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test


