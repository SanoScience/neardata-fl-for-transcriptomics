from typing import NamedTuple
import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class SklearnDataset(NamedTuple):
    X_train: any
    y_train: any
    X_test: any
    y_test: any
    n_features: any
    n_classes: any


class LRGenotypesClient(fl.client.NumPyClient):
    def __init__(self, sklearn_dataset: SklearnDataset, client_id: str) -> None:
        super().__init__()
        self.model = LogisticRegression(
            penalty="l2",
            max_iter=1,
            warm_start=True,
        )
        self.dataset = sklearn_dataset
        self._set_initial_params()
        self.client_id = client_id

    def get_parameters(self, config=None):
        if self.model.fit_intercept:
            params = [
                self.model.coef_,
                self.model.intercept_,
            ]
        else:
            params = [
                self.model.coef_,
            ]
        return params

    def fit(self, params, config):
        self._set_model_params(params)
        for _ in range(config["epochs_num"]):
            self.model.fit(self.dataset.X_train, self.dataset.y_train)
        return self.get_parameters(), len(self.dataset.X_train), {}

    def evaluate(self, params, config):
        self._set_model_params(params)
        loss = log_loss(self.dataset.y_test, self.model.predict_proba(self.dataset.X_test))
        accuracy = self.model.score(self.dataset.X_test, self.dataset.y_test)
        return loss, len(self.dataset.X_test), {"client_id": self.client_id, "accuracy": accuracy, "loss": loss}

    def _set_initial_params(self):
        self.model.classes_ = np.array([i for i in range(self.dataset.n_classes)])

        self.model.coef_ = np.zeros((self.dataset.n_classes, self.dataset.n_features))
        if self.model.fit_intercept:
            self.model.intercept_ = np.zeros((self.dataset.n_classes,))

    def _set_model_params(self, params) -> None:
        self.model.coef_ = params[0]
        if self.model.fit_intercept:
            self.model.intercept_ = params[1]
        return self.model
