import asyncio

import numpy as np
import pandas as pd

from .environments import SimulEnvironment

from .llm_utils import LLMClient
from .prompts import HF_PROMPTS
from .utils import extract_json_from_text


def get_prompt(example, prompt_type, env_kwargs):
    prompt_template = HF_PROMPTS[env_kwargs.name][prompt_type]
    prompt = prompt_template.format(**example)
    return prompt


def prepare_example(y: np.ndarray, env: SimulEnvironment):
    utility = env.get_utility_from_y(y)[0][0]
    grads = env.get_utility_gradient(y)[0]
    contributions = None
    if env.contributions_available and env.cfg.utility_func in [
        "beta_products",
        "piecewise_linear",
    ]:
        contributions = env.get_utility_contributions(y)[0].round(3).tolist()
    elif env.contributions_available and env.cfg.utility_func == "osy_sigmoid":
        contributions = env.get_utility_contributions(y)
        contributions = pd.DataFrame(contributions, index=[0])
        contributions.columns = [f"contrib(y_{i})" for i in range(1, 9)]
        contributions["contrib(y_1, y_2)"] = (
            contributions["contrib(y_1)"] + contributions["contrib(y_2)"]
        )
        columns = ["contrib(y_1, y_2)"] + [f"contrib(y_{i})" for i in range(3, 9)]
        contributions = contributions[columns]
        contributions = contributions.to_markdown(index=False)

    example = {
        "y": y[0].round(2).tolist(),
        "utility": np.round(utility, 3),
        "utility_gradient": grads.round(2).tolist(),
        "contributions": contributions,
        "env_kwargs": env.cfg,
    }
    return example


def _get_outcomes_markdown(
    y: np.ndarray, arm_index_ls: list, env: SimulEnvironment
) -> str:
    utility = env.get_utility_from_y(y).flatten()
    outcomes_df = pd.DataFrame(y, columns=env.y_names)
    outcomes_df["utility"] = utility
    outcomes_df["arm_index"] = arm_index_ls
    y_cols = env.y_names
    outcomes_df = outcomes_df[["arm_index"] + y_cols + ["utility"]]
    outcomes_df.columns = ["arm_index"] + env.y_names + ["utility"]

    if env.contributions_available:
        contributions = pd.DataFrame()
        contrib_cols = []
        if env.cfg.utility_func == "beta_products":
            contributions = env.get_utility_contributions(y)
            contrib_cols = [f"contrib_y_{i+1}" for i in range(env.n_obj)]
            contributions = pd.DataFrame(contributions, columns=contrib_cols)

        elif env.cfg.utility_func == "osy_sigmoid":
            contributions = env.get_utility_contributions(y)
            contributions = pd.DataFrame(contributions)
            contributions.columns = [f"contrib(y_{i+1})" for i in range(env.n_obj)]
            contributions["contrib(y_1, y_2)"] = (
                contributions["contrib(y_1)"] + contributions["contrib(y_2)"]
            )
            contrib_cols = ["contrib(y_1, y_2)"] + [
                f"contrib(y_{i})" for i in range(3, 9)
            ]
            contributions = contributions[contrib_cols]

        outcomes_df = pd.concat([outcomes_df, contributions], axis=1)
        outcomes_df = outcomes_df[
            ["arm_index"] + env.y_names + contrib_cols + ["utility"]
        ]
    else:
        outcomes_df = outcomes_df[["arm_index"] + env.y_names + ["utility"]]
    outcomes_markdown = outcomes_df.to_markdown(index=False)

    return outcomes_markdown


def get_human_answers(
    questions: list,
    exp_df: pd.DataFrame,
    env: SimulEnvironment,
    llm_client: LLMClient,
):
    prompt_template = HF_PROMPTS[env.cfg.name][env.cfg.utility_func + "_answers"]
    if len(exp_df) > 0:
        outcomes_markdown = _get_outcomes_markdown(
            y=exp_df[env.y_names].values,
            arm_index_ls=exp_df["arm_index"].to_list(),
            env=env,
        )
    else:
        outcomes_markdown = "No outcomes yet."
    questions_str = ""
    for i, q in enumerate(questions):
        questions_str += f"q{i+1}: {q}\n"
    n_questions = len(questions)
    prompt = prompt_template.format(
        outcomes_markdown=outcomes_markdown,
        questions_str=questions_str,
        n_questions=n_questions,
        env_kwargs=env.cfg,
        y_names=env.y_names,
    )
    response = None
    it = 0
    answers = []
    while response is None and it < 3:
        response = asyncio.run(
            llm_client.get_llm_response(prompt, kwargs={"max_tokens": 10000})
        )[0]
        a_dict = extract_json_from_text(response)
        if a_dict is None:
            print("JSON parsing error, raw response:")
            print(response)
        if a_dict is not None and len(a_dict) == n_questions:
            answers = []
            for a in a_dict.values():
                answers.append(a)
            answers = answers[:n_questions]
        else:
            response = None
        it += 1
    return answers


def get_pairwise_comparisons(
    arm_index_ls: list, exp_df: pd.DataFrame, env: SimulEnvironment
):
    """
    Gets pairwise comparisons for model performance results using the ground-truth utiliy.
    """

    exp_df = exp_df.copy()
    exp_df["true_utility"] = env.get_utility_from_y(
        exp_df[env.y_names].values
    ).flatten()
    comparisons = []
    for arm_idx_a, arm_idx_b in arm_index_ls:
        util_a = exp_df[exp_df["arm_index"] == arm_idx_a].iloc[0]["true_utility"]
        util_b = exp_df[exp_df["arm_index"] == arm_idx_b].iloc[0]["true_utility"]
        if util_a > util_b:
            comparisons.append(0)
        elif util_a < util_b:
            comparisons.append(1)
        else:
            comparisons.append(np.random.randint(2))
    return comparisons
