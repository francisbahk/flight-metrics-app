import io
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.test_functions.multi_objective import VehicleSafety
# from manifold.clients.python import ManifoldClient, StorageException  # Not used
from omegaconf import DictConfig
from pymoo.problems import get_problem
from scipy.stats import beta

from .btl_models import RewardNet

from .utils import sigmoid


class UtilityFunc(ABC):
    @abstractmethod
    def get_utility_from_y(self, y: np.ndarray) -> np.ndarray:
        """
        Returns the utility of an experiment given the outcomes y.

        Args:
            y: The outcome of the experiment.
        """
        pass

    @abstractmethod
    def get_utility_gradient(self, y: np.ndarray) -> np.ndarray:
        """
        Returns the gradient of the utility function with respect to the outcomes y.

        Args:
            y: The outcome of the experiment.
        """
        pass

    @abstractmethod
    def _get_goal_message(self) -> str:
        """
        Returns a message describing the optimization goal.
        """
        pass

    def get_goal_message(self) -> str:
        """
        Returns a message describing the optimization goal.
        """
        message = (
            "\n## Optimization Goal \nThe DM has provided the following information:\n"
        )
        message += self._get_goal_message()
        message += "\n"
        return message


class LinearUtility(UtilityFunc):
    def __init__(self, cfg: DictConfig):
        self.beta = cfg.beta
        self.bias = cfg.bias
        self.scale = cfg.scale
        self.contributions_available = False

    def get_utility_from_y(self, y):
        utilty = np.sum(self.beta * y, axis=1).reshape(-1, 1)
        utilty = (utilty - self.bias) / self.scale
        utilty = sigmoid(utilty)
        return utilty

    def get_utility_gradient(self, y):
        utility = np.sum(self.beta * y, axis=1).reshape(-1, 1)
        return sigmoid(utility) * (1 - sigmoid(utility)) * self.beta

    def _get_goal_message(self) -> str:
        beta_str = list(np.array(self.beta).round(2))
        ind = np.argpartition(self.beta, -3)[-3:]
        ind = np.sort(ind)
        top3 = [f"{beta_str[i]} * y_{i+1}" for i in ind]
        return f"- My goal is to increase all outcome metrics. I pay special attention to outcomes: {top3}."


class L1Utility(UtilityFunc):
    def __init__(self, cfg: DictConfig):
        self.opt_y = cfg.opt_y
        self.contributions_available = False

    def get_utility_from_y(self, y):
        utility = np.exp(-(np.linalg.norm(y - self.opt_y, ord=1, axis=1)))
        return utility.reshape(-1, 1)

    def get_utility_gradient(self, y):
        return np.where(y > self.opt_y, -1, 1)

    def _get_goal_message(self) -> str:
        opt_y_str = list(np.array(self.opt_y).round(2))
        return f"- My goal is to bring all the outcome metrics as close to {opt_y_str} as possible."


class BetaProductsUtility(UtilityFunc):
    def __init__(self, cfg: DictConfig):
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.contributions_available = True

    def get_utility_from_y(self, y):
        return np.prod(beta.cdf(y, self.alpha, self.beta), axis=1).reshape(-1, 1)

    def get_utility_gradient(self, y):
        cdf = beta.cdf(y, self.alpha, self.beta)
        pdf = beta.pdf(y, self.alpha, self.beta)
        prod_all = np.prod(cdf, axis=1, keepdims=True)
        grad = pdf * prod_all / (cdf + 1e-8)
        return grad

    def get_utility_contributions(self, y):
        return beta.cdf(y, self.alpha, self.beta)

    def _get_goal_message(self) -> str:
        return "- My goal is to bring all the outcome metrics as close to 1 as possible. Results are strongest only when every metric is high—if any metric is low, it significantly reduces the overall performance."


class PiecewiseLinearUtility(UtilityFunc):
    def __init__(self, cfg: DictConfig):
        self.beta_1 = np.array(cfg.beta_1)
        self.beta_2 = np.array(cfg.beta_2)
        self.t = np.array(cfg.t)
        self.bias = cfg.bias
        self.scale = cfg.scale
        self.n_obj = len(self.beta_1)
        self.contributions_available = True
        self.y_bounds_str = "\n\t".join(
            [
                "y_{i} in [{lower:.2f}, {upper:.2f}]".format(
                    i=i + 1, lower=cfg.y_lower[i], upper=cfg.y_upper[i]
                )
                for i in range(self.n_obj)
            ]
        )

    def get_utility_from_y(self, y):
        if y.ndim == 1:
            y = y.reshape(1, -1)
        h1 = self.beta_1 * y + (self.beta_2 - self.beta_1) * self.t
        h2 = self.beta_2 * y
        h = np.where(y < self.t, h1, h2)
        utility = h.sum(axis=1).reshape(-1, 1)
        utility = (utility - self.bias) / self.scale
        utility = sigmoid(utility)
        return utility

    def get_utility_gradient(self, y):
        grad = np.where(
            y < self.t,
            self.beta_1,
            self.beta_2,
        )
        return grad

    def get_avg_utility(self):
        rng = np.random.default_rng(0)
        y = rng.random((10000, self.n_obj))
        utility = self._get_utility_from_y(y)
        return utility.mean()

    def get_utility_contributions(self, y):
        if y.ndim == 1:
            y = y.reshape(1, -1)
        h1 = self.beta_1 * y + (self.beta_2 - self.beta_1) * self.t
        h2 = self.beta_2 * y
        h = np.where(y < self.t, h1, h2)
        return h

    def _get_goal_message(self) -> str:
        t_str = ", ".join(
            [f"y_{i+1} >= {x}" for i, x in enumerate(list(np.array(self.t).round(2)))]
        )
        message = f"- My goal is to achieve the following thresholds in each outcome: {t_str}. Improvements over the thresholds are always good, but less important than bringing the outcomes to their threshold values. The further away an outcome is from its threshold, the higher is its negative impact on the overall performance.\n"
        return message


class SimulEnvironment(ABC):
    def __init__(self, cfg: DictConfig):
        self.n_var = cfg.n_var
        self.n_obj = cfg.n_obj
        self.seed = cfg.seed
        self.cfg = cfg
        self.x_names = ["x_" + str(i + 1) for i in range(self.n_var)]
        self.y_names = ["y_" + str(i + 1) for i in range(self.n_obj)]

        if cfg.utility_func == "piecewise_linear":
            self.utility_func = PiecewiseLinearUtility(cfg)
        elif cfg.utility_func == "l1":
            self.utility_func = L1Utility(cfg)
        elif cfg.utility_func == "beta_products":
            self.utility_func = BetaProductsUtility(cfg)
        elif cfg.utility_func == "osy_sigmoid":
            self.utility_func = OSYSigmoidUtility(cfg)
        elif cfg.utility_func in ["vehicle_safety_llm", "carcab_llm"]:
            self.utility_func = BTLRewardUtility(cfg)
        elif cfg.utility_func.startswith("thermo"):
            self.utility_func = ThermoUtilityFunc(cfg)
        else:
            raise ValueError(f"Unknown utility function name {cfg.utility_func}.")

        self.contributions_available = self.utility_func.contributions_available

    @abstractmethod
    def get_outcomes(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the outcome of an experiment given the inputs x.

        Args:
            x: The input to the experiment.
        """
        pass

    def get_utility_from_y(self, y: np.ndarray) -> np.ndarray:
        """
        Returns the utility of an experiment given the outcomes y.

        Args:
            y: The outcome of the experiment.
        """
        return self.utility_func.get_utility_from_y(y)

    def get_utility(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the utility of an experiment given the inputs x.

        Args:
            x: The input to the experiment.
        """
        y = self.get_outcomes(x)
        return self.get_utility_from_y(y)

    def get_utility_gradient(self, y: np.ndarray) -> np.ndarray:
        """
        Returns the gradient of the utility function with respect to the outcomes y.

        Args:
            y: The outcome of the experiment.
        """
        return self.utility_func.get_utility_gradient(y)

    def get_utility_contributions(self, y: np.ndarray) -> np.ndarray:
        """
        Returns a measure of the contribution of each feature to the utility of an experiment.

        Args:
            x: The input to the experiment.
        """
        if self.contributions_available:
            return self.utility_func.get_utility_contributions(y)
        else:
            raise ValueError("No utility contributions available for this environment.")

    @abstractmethod
    def get_random_x(self, seed: int, N: int = 1) -> np.ndarray:
        """
        Returns a random input to the experiment.

        Args:
            N: The number of random inputs to generate.
        """
        pass

    def get_goal_message(self) -> str:
        """
        Returns a message describing the optimization goal.
        """
        return self.utility_func.get_goal_message()

    def get_prior_message(
        self, top_q: float = 0.1, prior_type: str = "promising_point"
    ) -> str:
        """
        Returns a message describing the DM's prior over the inputs.
        """
        X = self.get_random_x(self.seed, max(4**self.n_var, 50000))
        Y = self.get_outcomes(X)
        U = self.get_utility_from_y(Y)
        df = pd.DataFrame(X, columns=self.x_names)
        df["utility"] = U.squeeze()
        q = df.utility.quantile((1 - top_q))
        if prior_type == "point":
            promising_point = df[df.utility >= q].sample(1).iloc[0][self.x_names]
            promising_point = list(promising_point.round(2))
            return f"- Based on my experience, the following inputs should bring good results: {promising_point}."
        elif prior_type == "area":
            promising_points = df[df.utility >= q]
            lower = promising_points[self.x_names].quantile(0.25).values.round(2)
            upper = promising_points[self.x_names].quantile(0.75).values.round(2)
            message = "- Based on my experience, inputs within these ranges should bring good results:"
            for x, lower_x, upper_x in zip(self.x_names, lower, upper):
                message += f"\n\t{x} in [{lower_x}, {upper_x}]"
            return message
        else:
            raise ValueError(f"Unknown prior type {prior_type}.")

    @abstractmethod
    def get_input_transform(self) -> InputTransform:
        """
        Returns botorch input transform.
        """
        pass

    @abstractmethod
    def get_problem_bounds(self) -> np.ndarray:
        """
        Returns the bounds of the problem.
        """
        pass


class DTLZ2Environment(SimulEnvironment):
    def __init__(self, cfg: DictConfig):
        default_cfg: Dict[str, Union[int, str, list, float]] = {
            "n_var": 8,
            "n_obj": 4,
            "outcome_func": "dtlz2",
        }

        default_cfg.update(cfg)
        self.problem = "dtlz2"
        self.pf = get_problem(
            self.problem, n_var=default_cfg["n_var"], n_obj=default_cfg["n_obj"]
        )
        self.utility_func_name = cfg.utility_func
        if cfg.utility_func == "piecewise_linear":
            default_cfg.update(
                {
                    "beta_1": [4.0, 3.0, 2.0, 1.0],
                    "beta_2": [0.4, 0.3, 0.2, 0.1],
                    "t": [1.0, 0.8, 0.5, 0.5],
                    "bias": -2.0,
                    "scale": 1.7,
                    "y_lower": [0.0, 0.0, 0.0, 0.0],
                    "y_upper": [2.0, 2.0, 2.0, 2.0],
                }
            )

        elif cfg.utility_func == "l1":
            # opt_x = np.ones(int(default_cfg["n_var"])) * 0.5
            # opt_y = self.get_outcomes(opt_x).squeeze().tolist()
            # default_cfg.update({"opt_y": opt_y})
            default_cfg.update({"opt_y": [0.8, 1.0, 0.7, 1.25]})

        elif cfg.utility_func == "beta_products":
            default_cfg.update(
                {
                    "alpha": [0.5, 2.0, 2.0, 2.0],
                    "beta": [0.5, 1.0, 2.0, 5.0],
                }
            )

        # Override defaults with configs
        default_cfg.update(cfg)
        cfg = DictConfig(default_cfg)
        super().__init__(cfg)

    def get_outcomes(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the outcome of an experiment given the inputs x.

        Args:
            x: The input to the experiment.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y = self.pf.evaluate(x)
        if self.utility_func_name == "beta_products":
            y = (np.abs(y) / 2) ** 0.1
        return y

    def get_random_x(self, seed: int, N: int) -> np.ndarray:
        """
        Returns a random input to the experiment.

        Args:
            N: The number of random inputs to generate.
        """
        rng = np.random.default_rng(seed)
        x = np.array(rng.random((N, self.n_var)))
        return x

    def get_input_transform(self) -> InputTransform:
        bounds = torch.stack([torch.zeros(self.n_var), torch.ones(self.n_var)])
        return Normalize(d=self.n_var, bounds=bounds)

    def get_problem_bounds(self) -> np.ndarray:
        return np.array([[0.0] * self.n_var, [1.0] * self.n_var])


class VehicleSafetyEnvironment(SimulEnvironment):
    def __init__(self, cfg: DictConfig):
        self.problem = VehicleSafety(negate=True)
        default_cfg: Dict[str, Union[int, str, list, float]] = {
            "n_var": 5,
            "n_obj": 3,
            "outcome_func": "vehicle_safety",
        }

        if cfg.utility_func == "piecewise_linear":
            default_cfg.update(
                {
                    "beta_1": [2, 6, 8],
                    "beta_2": [1, 2, 2],
                    "t": [0.5, 0.8, 0.8],
                    "bias": 0.0,
                    "scale": 1.0,
                    "y_lower": [0.0, 0.0, 0.0],
                    "y_upper": [1.0, 1.0, 1.0],
                }
            )

        elif cfg.utility_func == "beta_products":
            default_cfg.update(
                {
                    "alpha": [0.5, 1.0, 1.5],
                    "beta": [1, 2, 3],
                }
            )

        elif cfg.utility_func == "vehicle_safety_llm":
            default_cfg.update(
                {
                    "model_path": "tree/language_bo/vehicle_safety_intrusion_model.pt",
                    "input_dim": 3,
                    "hidden_dim": 64,
                }
            )

        # Override defaults with configs
        default_cfg.update(cfg)
        cfg = DictConfig(default_cfg)

        self.x_bounds = self.problem.bounds.numpy()
        self.y_bounds = np.array(
            [
                [1.7040e03, 1.1708e01, 2.6192e-01],
                [1.6619e03, 6.2136e00, 4.2879e-02],
            ]
        )

        super().__init__(cfg)

        self.y_names = [
            "y_1 (mass)",
            "y_2 (int_acc)",
            "y_3 (intrusion)",
        ]

        if self.cfg.utility_func in ["piecewise_linear", "beta_products"]:
            self.y_names = [
                "y_1 (1 - mass)",
                "y_2 (1 - int_acc)",
                "y_3 (1 - intrusion)",
            ]

    def get_outcomes(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y = self.problem.evaluate_true(torch.tensor(x)).numpy()
        y = (y - self.y_bounds[0, :]) / (self.y_bounds[1, :] - self.y_bounds[0, :])
        if self.cfg.utility_func in ["piecewise_linear", "beta_products"]:
            y = 1 - y
        return y

    def get_random_x(self, seed: int, N: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = np.array(
            rng.uniform(
                low=self.x_bounds[0, :], high=self.x_bounds[1, :], size=(N, self.n_var)
            )
        )
        return x

    def get_input_transform(self) -> InputTransform:
        bounds = torch.stack(
            [torch.tensor(self.x_bounds[0, :]), torch.tensor(self.x_bounds[1, :])]
        )
        return Normalize(d=self.n_var, bounds=bounds)

    def get_problem_bounds(self) -> np.ndarray:
        return self.x_bounds

    def get_prior_message(self, top_q: float = 0.1, prior_type: str = "domain") -> str:
        if prior_type == "domain":
            if self.cfg.utility_func in ["piecewise_linear", "beta_products"]:
                message = "- y_1 measures the reduction in vehicle's mass, y_2 measures the reduction in integration of acceleration between two time points, y_3 measures the reduction in toe board intrusion in the offset-frontal crash."
            else:
                message = "- y_1 measures the mass of the vehicle, y_2 is an integration of acceleration between two time points, y_3 measures the toe board intrusion in the offset-frontal crash."
            message += "\n- The parameters x measure the thickness of five reinforced members around the frontal structure of a car, which can significantly affect the crash safety."
            return message
        else:
            return super().get_prior_message(top_q, prior_type)


class CarCabEnvironment(SimulEnvironment):
    def __init__(self, cfg: DictConfig):
        r"""RE9-7-1 car cab design from Tanabe & Ishibuchi (2020)"""
        default_cfg: Dict[str, Union[int, str, list, float]] = {
            "n_var": 7,
            "n_obj": 11,
            "outcome_func": "carcab",
        }

        self.utility_func_name = cfg.utility_func

        self.x_bounds = np.array(
            [
                [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4],
                [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2],
            ]
        )

        if cfg.utility_func == "piecewise_linear":
            n_obj = int(default_cfg["n_obj"])
            default_cfg.update(
                {
                    "beta_1": [
                        7.0,
                        6.75,
                        6.5,
                        6.25,
                        6.0,
                        5.75,
                        5.5,
                        5.25,
                        5.0,
                        4.75,
                        4.5,
                    ],
                    "beta_2": [
                        0.5,
                        0.4,
                        0.375,
                        0.35,
                        0.325,
                        0.3,
                        0.275,
                        0.25,
                        0.225,
                        0.2,
                        0.175,
                    ],
                    "t": [
                        0.64,
                        0.68,
                        0.96,
                        0.88,
                        1.06,
                        0.65,
                        0.84,
                        0.86,
                        0.58,
                        0.7,
                        0.53,
                    ],
                    "bias": 0.0,
                    "scale": 1.0,
                    "y_lower": [0.0 for _ in range(n_obj)],
                    "y_upper": [1.0 for _ in range(n_obj)],
                }
            )
        elif cfg.utility_func == "carcab_llm":
            default_cfg.update(
                {
                    "model_path": "tree/language_bo/carcab_const_model.pt",
                    "input_dim": 11,
                    "hidden_dim": 64,
                }
            )

        self.y_bounds = np.array(
            [
                [16.48, 0.35, 0.18, 0.16, 0.28, 20.25, 22.5, 28.05, 3.1, 7.88, 11.21],
                [45.23, 1.23, 0.34, 0.27, 0.59, 38.23, 42.02, 48.19, 5.07, 11.98, 18.4],
            ]
        )

        # Override defaults with configs
        default_cfg.update(cfg)
        cfg = DictConfig(default_cfg)

        super().__init__(cfg)

        self.y_names = [
            "y_1 (weight)",
            "y_2 (abdomen load)",
            "y_3 (viscous criterion V * Cu)",
            "y_4 (viscous criterion V * Cm)",
            "y_5 (viscous criterion V * Cl)",
            "y_6 (upper rib deflection)",
            "y_7 (middle rib deflection)",
            "y_8 (lower rib deflection)",
            "y_9 (pubic symphysis force)",
            "y_10 (vel. of B-Pillar at mid.)",
            "y_11 (Front door vel. at B-Pillar)",
        ]

        if cfg.utility_func == "piecewise_linear":
            self.y_names = [
                "y_1 (1 - weight)",
                "y_2 (1 - abdomen load)",
                "y_3 (1 - viscous criterion V * Cu)",
                "y_4 (1 - viscous criterion V * Cm)",
                "y_5 (1 - viscous criterion V * Cl)",
                "y_6 (1 - upper rib deflection)",
                "y_7 (1 - middle rib deflection)",
                "y_8 (1 - lower rib deflection)",
                "y_9 (1 - pubic symphysis force)",
                "y_10 (1 - vel. of B-Pillar at mid.)",
                "y_11 (1 - Front door vel. at B-Pillar)",
            ]

    def get_outcomes(self, x: np.ndarray) -> np.ndarray:
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x3 = x[:, 3]
        x4 = x[:, 4]
        x5 = x[:, 5]
        x6 = x[:, 6]
        x7 = np.ones_like(x1) * 0.345
        x8 = np.ones_like(x1) * 0.192
        x9 = np.zeros_like(x1)
        x10 = np.zeros_like(x1)

        y = np.zeros((x.shape[0], self.n_obj))

        y[:, 0] = (
            1.98
            + 4.9 * x0
            + 6.67 * x1
            + 6.98 * x2
            + 4.01 * x3
            + 1.75 * x4
            + 0.00001 * x5
            + 2.73 * x6
        )
        y[:, 1] = (
            1.16
            - 0.3717 * x1 * x3
            - 0.00931 * x1 * x9
            - 0.484 * x2 * x8
            + 0.01343 * x5 * x9
        )
        y[:, 2] = (
            0.261
            - 0.0159 * x0 * x1
            - 0.188 * x0 * x7
            - 0.019 * x1 * x6
            + 0.0144 * x2 * x4
            + 0.87570001 * x4 * x9
            + 0.08045 * x5 * x8
            + 0.00139 * x7 * x10
            + 0.00001575 * x9 * x10
        )
        y[:, 3] = (
            0.214
            + 0.00817 * x4
            - 0.131 * x0 * x7
            - 0.0704 * x0 * x8
            + 0.03099 * x1 * x5
            - 0.018 * x1 * x6
            + 0.0208 * x2 * x7
            + 0.121 * x2 * x8
            - 0.00364 * x4 * x5
            + 0.0007715 * x4 * x9
            - 0.0005354 * x5 * x9
            + 0.00121 * x7 * x10
            + 0.00184 * x8 * x9
            - 0.018 * x1 * x1
        )
        y[:, 4] = (
            0.74
            - 0.61 * x1
            - 0.163 * x2 * x7
            + 0.001232 * x2 * x9
            - 0.166 * x6 * x8
            + 0.227 * x1 * x1
        )
        y[:, 5] = (
            28.98
            + 3.818 * x2
            - 4.2 * x0 * x1
            + 0.0207 * x4 * x9
            + 6.63 * x5 * x8
            - 7.77 * x6 * x7
            + 0.32 * x8 * x9
        )
        y[:, 6] = (
            33.86
            + 2.95 * x2
            + 0.1792 * x9
            - 5.057 * x0 * x1
            - 11 * x1 * x7
            - 0.0215 * x4 * x9
            - 9.98 * x6 * x7
            + 22 * x7 * x8
        )
        y[:, 7] = 46.36 - 9.9 * x1 - 12.9 * x0 * x7 + 0.1107 * x2 * x9
        y[:, 8] = (
            4.72
            - 0.5 * x3
            - 0.19 * x1 * x2
            - 0.0122 * x3 * x9
            + 0.009325 * x5 * x9
            + 0.000191 * x10 * x10
        )
        y[:, 9] = (
            10.58
            - 0.674 * x0 * x1
            - 1.95 * x1 * x7
            + 0.02054 * x2 * x9
            - 0.0198 * x3 * x9
            + 0.028 * x5 * x9
        )
        y[:, 10] = (
            16.45
            - 0.489 * x2 * x6
            - 0.843 * x4 * x5
            + 0.0432 * x8 * x9
            - 0.0556 * x8 * x10
            - 0.000786 * x10 * x10
        )

        if self.utility_func_name == "piecewise_linear":
            y = (y - self.y_bounds[0, :]) / (self.y_bounds[1, :] - self.y_bounds[0, :])
            y = 1 - y  # minimize weight

        return y

    def get_random_x(self, seed: int, N: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = np.array(
            rng.uniform(
                low=self.x_bounds[0, :], high=self.x_bounds[1, :], size=(N, self.n_var)
            )
        )
        return x

    def get_input_transform(self) -> InputTransform:
        bounds = torch.stack(
            [torch.tensor(self.x_bounds[0, :]), torch.tensor(self.x_bounds[1, :])]
        )
        return Normalize(d=self.n_var, bounds=bounds)

    def get_problem_bounds(self) -> np.ndarray:
        return self.x_bounds

    def get_prior_message(self, top_q: float = 0.1, prior_type: str = "domain") -> str:
        if prior_type == "domain":
            message = "- A car is subjected to a side-impact crash test. The outcome veriables y meassure the following:"
            message += "\n- The effect of the side-impact on a dummy is measured in terms of head injury, load in abdomen, pubic symphysis force, viscous criterion (V * C), and rib deflections at the upper, middle and lower rib locations."
            message += "\n- The effect on the car are considered in terms of the vehicle's weight, the velocity of the B-Pillar at the middle point and the velocity of the front door at the B-Pillar."
            message += "\n- The parameters x describe some design aspects of the car. An increase in dimension of the car parameters may improve safety, but with a burden of an increased weight of the car."
            message += " These parameters are and their considered ranges are:"
            message += "\n\tx1: Thickness of B-Pillar inner [0.5, 1.5]"
            message += "\n\tx2: Thickness of B-Pillar reinforcement [0.45, 1.35]"
            message += "\n\tx3: Thickness of floor side inner [0.5, 1.5]"
            message += "\n\tx4: Thickness of cross members [0.5, 1.5]"
            message += "\n\tx5: Thickness of door beam [0.875, 2.625]"
            message += "\n\tx6: Thickness of door beltline reinforcement [0.4, 1.2]"
            message += "\n\tx7: Thickness of roof rail [0.4, 1.2]"
            if self.cfg.utility_func in ["piecewise_linear"]:
                message += "\n- NOTE: The presented values of outcomes y represent the reduction in mass, forces, velocities etc. So the goal is to increase y_1, ..., y_11, corresponding to lowering the vehicle's weight and minimizing the impact on the dummy and the car."
            return message
        else:
            return super().get_prior_message(top_q, prior_type)


class ThermoUtilityFunc(UtilityFunc):
    def __init__(self, cfg: DictConfig):
        self.L_ppd, self.U_ppd = cfg.ppd[0], cfg.ppd[1]
        self.L_dr, self.U_dr = cfg.dr[0], cfg.dr[1]
        self.L_dT, self.U_dT = cfg.dT[0], cfg.dT[1]
        self.L_rad, self.U_rad = cfg.rad[0], cfg.rad[1]
        self.min_tf, self.L_tf, self.U_tf, self.max_tf = (
            cfg.tf[0],
            cfg.tf[1],
            cfg.tf[2],
            cfg.tf[3],
        )
        self.contributions_available = True

    def _d_smaller_better(self, y, L, U, s=1.0):
        y = np.asarray(y)
        L = np.asarray(L)
        U = np.asarray(U)
        d = np.where(y <= L, 1.0, np.where(y >= U, 0.0, ((U - y) / (U - L)) ** s))
        return d

    def _d_target_range(self, y, low, high, min_low, max_high, s=1.0):
        y = np.asarray(y)
        low = np.asarray(low)
        high = np.asarray(high)
        min_low = np.asarray(min_low)
        max_high = np.asarray(max_high)
        d = np.where(
            (y >= low) & (y <= high),
            1.0,
            np.where(
                y < low,
                np.where(y <= min_low, 0.0, ((y - min_low) / (low - min_low)) ** s),
                np.where(y >= max_high, 0.0, ((max_high - y) / (max_high - high)) ** s),
            ),
        )
        return d

    def get_utility_from_y(self, y):
        contributions = self.get_utility_contributions(y)
        U = np.prod(contributions, axis=1) ** (1.0 / 5.0)
        return U

    def get_utility_contributions(self, y):
        ppd, dr, dT_vert, dT_pr, T_floor = [y[:, i] for i in range(5)]
        d_ppd = self._d_smaller_better(ppd, self.L_ppd, self.U_ppd, s=1.2)
        d_dr = self._d_smaller_better(dr, self.L_dr, self.U_dr, s=1.2)
        d_vert = self._d_smaller_better(dT_vert, self.L_dT, self.U_dT, s=1.0)
        d_rad = self._d_smaller_better(dT_pr, self.L_rad, self.U_rad, s=1.0)
        d_floor = self._d_target_range(
            T_floor, self.L_tf, self.U_tf, self.min_tf, self.max_tf, s=1.2
        )
        contributions = np.stack([d_ppd, d_dr, d_vert, d_rad, d_floor], axis=1)
        return contributions

    def get_utility_gradient(self, y):
        raise NotImplementedError

    def _get_goal_message(self) -> str:
        message = (
            "- My goal is to keep all metrics within my thermal comfort preferences."
        )
        return message


class ThermoEnvironment(SimulEnvironment):
    def __init__(self, cfg: DictConfig):
        default_cfg: Dict[str, Union[int, str, list, float]] = {
            "n_var": 8,
            "n_obj": 5,
            "outcome_func": "thermo",
        }

        default_cfg.update(cfg)
        self.utility_func_name = cfg.utility_func
        if cfg.utility_func == "thermo_A":
            # Co-working space user—light clothing, moderate tolerance
            default_cfg.update(
                {
                    "ppd": [0.0, 30.0],
                    "dr": [10.0, 35.0],
                    "dT": [3.0, 9.0],
                    "rad": [5.0, 22.0],
                    "tf": [16.0, 19.0, 26.0, 30.0],
                    "met": 1.0,
                    "clo": 0.61,
                }
            )
        elif cfg.utility_func == "thermo_B":
            # Summer athlete, light kit
            default_cfg.update(
                {
                    "ppd": [0.0, 24.0],
                    "dr": [30.0, 45.0],
                    "dT": [2.5, 6.0],
                    "rad": [4.0, 12.0],
                    "tf": [19.0, 20.0, 23.0, 25.0],
                    "met": 2.0,
                    "clo": 0.3,
                }
            )
        elif cfg.utility_func == "thermo_C":
            # Office winter, warm clothing
            default_cfg.update(
                {
                    "ppd": [0.0, 22.0],
                    "dr": [15.0, 30.0],
                    "dT": [2.5, 5.0],
                    "rad": [5.0, 14.0],
                    "tf": [19.0, 21.0, 24.0, 26.0],
                    "met": 1.2,
                    "clo": 1.4,
                }
            )

        self.met = default_cfg["met"]
        self.clo = default_cfg["clo"]

        # Override defaults with configs
        default_cfg.update(cfg)
        cfg = DictConfig(default_cfg)
        super().__init__(cfg)

        self.x_bounds = np.array(
            [
                [18.0, 30.0],  # Ta (°C)
                [18.0, 30.0],  # Tr (°C)
                [20.0, 80.0],  # RH (%)
                [0.00, 0.50],  # v (m/s)
                [0.05, 0.70],  # Tu (fraction)
                [0.0, 6.0],  # dT_vert (K)
                [0.0, 20.0],  # dT_pr (K)
                [16.0, 30.0],  # T_floor (°C)
                # [1.0, 2.0],  # Met (met)
                # [0.3, 1.2],  # Clo (clo)
            ]
        ).T

        self.x_names = [
            "x_1: Ta (°C)",
            "x_2: Tr (°C)",
            "x_3: RH (%)",
            "x_4: v (m/s)",
            "x_5: Tu (fraction)",
            "x_6: dT_vert (K)",
            "x_7: dT_pr (K)",
            "x_8: T_floor (°C)",
        ]

        self.y_names = [
            "y_1: PPD (% dissatisfied)",
            "y_2: DR (% draft risk)",
            "y_3: dT_vert (K)",
            "y_4: dT_pr (K)",
            "y_5: T_floor (°C)",
        ]

    def _pmv_ppd(self, ta, tr, vel, rh, met, clo, wme=None):
        """
        Vectorized PMV/PPD per ISO 7730 / ASHRAE 55.
        All inputs are np arrays of shape [bs].
        Returns: (pmv, ppd) arrays of shape [bs]
        """
        if wme is None:
            wme = np.zeros_like(met)
        icl = 0.155 * clo
        m = met * 58.15
        w = wme * 58.15
        mw = m - w
        pa = (rh / 100.0) * 10.0 * np.exp(16.6536 - 4030.183 / (ta + 235.0))
        fcl = np.where(icl <= 0.078, 1.0 + 1.29 * icl, 1.05 + 0.645 * icl)
        taa = ta + 273.0
        tra = tr + 273.0
        tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)
        p1 = icl * fcl
        p2 = p1 * 3.96
        p3 = p1 * 100.0
        p4 = p1 * taa
        p5 = 308.7 - 0.028 * mw + p2 * ((tra / 100.0) ** 4)
        xn = tcla / 100.0
        eps = 1e-4
        hcf = 12.1 * np.sqrt(np.maximum(vel, 0.0))
        for _ in range(200):
            hcn = 2.38 * np.abs(100.0 * xn - taa) ** 0.25
            hc = np.maximum(hcf, hcn)
            xn_new = (p5 + p4 * hc - p2 * (xn**4)) / (100.0 + p3 * hc)
            if np.all(np.abs(xn_new - xn) <= eps):
                xn = xn_new
                break
            xn = 0.5 * (xn + xn_new)
        tcl = 100.0 * xn - 273.0
        hl1 = 3.05e-3 * (5733 - 6.99 * mw - pa)
        hl2 = np.where(mw > 58.15, 0.42 * (mw - 58.15), 0.0)
        hl3 = 1.7e-5 * m * (5867 - pa)
        hl4 = 0.0014 * m * (34.0 - ta)
        hl5 = 3.96 * fcl * (xn**4 - (tra / 100.0) ** 4)
        hc = np.maximum(hcf, 2.38 * np.abs(tcl - ta) ** 0.25)
        hl6 = fcl * hc * (tcl - ta)
        ts = 0.303 * np.exp(-0.036 * m) + 0.028
        pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
        ppd = 100.0 - 95.0 * np.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
        ppd = np.clip(ppd, 0.0, 100.0)
        return pmv, ppd

    def _draft_rate(self, ta, vel, tu_fraction):
        """
        Vectorized Draft Rate DR (%) per ISO/ASHRAE.
        All inputs are np arrays of shape [bs].
        """
        vel = np.maximum(0.0, vel)
        base = np.maximum(vel - 0.05, 0.0)
        tu_percent = np.clip(tu_fraction * 100.0, 0.0, 100.0)
        dr = (34.0 - ta) * (base**0.62) * (0.37 * vel * tu_percent + 3.14)
        dr = np.clip(dr, 0.0, 100.0)
        return dr

    def get_outcomes(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorized outcome function y = f(x).
        x: np.ndarray of shape [bs, 10]
        Returns y: np.ndarray of shape [bs, 5]
        """
        Ta, Tr, RH, v, Tu, dT_vert, dT_pr, T_floor = [x[:, i] for i in range(8)]
        Met, Clo = self.met, self.clo
        _, ppd = self._pmv_ppd(Ta, Tr, v, RH, Met, Clo)
        dr = self._draft_rate(Ta, v, Tu)
        y = np.stack([ppd, dr, dT_vert, dT_pr, T_floor], axis=1)
        return y

    def get_random_x(self, seed: int, N: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = np.array(
            rng.uniform(
                low=self.x_bounds[0, :], high=self.x_bounds[1, :], size=(N, self.n_var)
            )
        )
        return x

    def get_input_transform(self) -> InputTransform:
        bounds = torch.stack(
            [torch.tensor(self.x_bounds[0, :]), torch.tensor(self.x_bounds[1, :])]
        )
        return Normalize(d=self.n_var, bounds=bounds)

    def get_problem_bounds(self) -> np.ndarray:
        return self.x_bounds

    def get_prior_message(self, top_q: float = 0.1, prior_type: str = "domain") -> str:
        if prior_type == "domain":
            message = (
                "The parameters x describe the following aspects of the environment:"
            )
            message += "\n\tx_1: Ta air temperature (°C)"
            message += "\n\tx_2: Tr mean radiant temperature (°C)"
            message += "\n\tx_3: RH relative humidity (%)"
            message += "\n\tx_4: v air speed (m/s)"
            message += "\n\tx_5: Tu turbulence intensity (fraction)"
            message += "\n\tx_6: dT_vert vertical air-temperature difference (K)"
            message += "\n\tx_7: dT_pr air-temperatur assymetry (K)"
            message += "\n\tx_8: T_floor floor surface temperature (°C)"

            message += (
                "\n- The outcome variables y determining the DM's satisfaction are:"
            )
            message += "\n\ty_1: PPD (%) whole body dissatisfaction based on the ISO 7730 standard"
            message += "\n\ty_2: DR (%) local draft risk at neck level"
            message += "\n\ty_3 = x_6: dT_vert (K)"
            message += "\n\ty_4 = x_7: dT_pr (K)"
            message += "\n\ty_5 = x_8: T_floor (°C)"

            return message

        else:
            return super().get_prior_message()
