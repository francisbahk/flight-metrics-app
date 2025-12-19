from typing import Tuple

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.models.transforms.input import InputTransform
from gpytorch.mlls import ExactMarginalLogLikelihood


class SimpleGPProxyModel:
    def __init__(
        self,
        input_names: list,
        target_col: str,
        input_transform: InputTransform,
    ):
        """
        Initializes a Gaussian Process model.
        Args:
            env: Environment object.
            target_col: Name of the target column to optimize.
        """
        self.model = None
        self.input_transform = input_transform
        self.target_col = target_col
        self.input_names = input_names
        self.multi_task = False

    def fit(self, train_df: pd.DataFrame) -> SingleTaskGP:
        """
        Fits a Gaussian Process model to the training data.

        Args:
            train_df: DataFrame containing features and target column.

        Returns:
            SingleTaskGP model and fitted MinMaxScaler.
        """
        train_X = torch.tensor(train_df[self.input_names].values, dtype=torch.float64)
        train_Y = torch.tensor(
            train_df[self.target_col].values.reshape(-1, 1), dtype=torch.float64
        )

        self.model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=self.input_transform,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll=mll)
        return self.model

    def fit_transform(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits a Gaussian Process model to the training data.
        Returns posterior mean and variance on the test data.

        Args:
            train_df: DataFrame containing features and target column.
            test_df: DataFrame containing features for predictions.

        Returns:
            Tuple of (posterior_mean, posterior_variance).
        """
        self.fit(train_df)
        test_X = torch.tensor(test_df[self.input_names].values)
        posterior = self.model.posterior(test_X)
        mean, var = posterior.mean.detach().numpy(), posterior.variance.detach().numpy()
        return mean, var
