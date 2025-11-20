from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        self.model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            multi_class='multinomial'
        )
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X_test)