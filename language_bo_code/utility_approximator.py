import asyncio
import random
import re
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from pyre_extensions import assert_is_instance

from .environments import SimulEnvironment

from .gp_models import SimpleGPProxyModel
from .llm_utils import LLMClient
from .prompts import (
    SCALAR_UAPPROX,
    PAIRWISE_UAPROX,
    QA_SUMMARIZER,
    UAPPROX_PRIOR_LABEL,
    UAPPROX_QA_LABEL,
    UAPPROX_QUESTIONS_INIT,
    UAPPROX_QUESTIONS_PAIRWISE,
    UAPPROX_QUESTIONS_SCALAR,
)
from .utils import extract_json_from_text, extract_jsonl_from_text


def data_to_string(
    y_ls: list,
    feedback_ls: list,
    y_names: list,
    masked: bool = False,
    markdown: bool = False,
) -> str:
    """
    Converts experimental outcomes and feedback data to a formatted string representation.
    Creates numbered entries with outcomes and feedback.

    Args:
        y: Array containing experimental ourcomes.
        y_names: List of outcome names.
        feedback: List of feedback strings.
        masked: If True, uses generic labels (y1, y2, y3, ..) instead of metric names.

    Returns:
        Formatted string with numbered entries containing the performance data.
    """
    if masked:
        y_names = ["y_" + str(i + 1) for i in range(len(y_names))]

    if not markdown:

        def get_line_entry(y, y_names):
            return ", ".join(
                [
                    "{y_name} = {y_val:.2f}".format(y_name=y_name, y_val=y_val)
                    for y_name, y_val in zip(y_names, y)
                ]
            )

        return "\n\n".join(
            [
                "{i}. {y_line}, feedback = {feedback}".format(
                    i=i, y_line=get_line_entry(y, y_names), feedback=feedback
                )
                for i, (y, feedback) in enumerate(zip(y_ls, feedback_ls))
            ]
        )

    else:
        df = pd.DataFrame(np.concatenate([y_ls]))
        df.columns = y_names
        df["idx"] = df.index
        df["feedback"] = feedback_ls
        df = df[["idx"] + y_names + ["feedback"]]
        return df.to_markdown(index=False, tablefmt="github")


def exp_data_to_string(
    exp_df: pd.DataFrame,
    y_names: list,
    masked: bool = False,
    markdown: bool = False,
    start_idx: int = 0,
) -> str:
    """
    Converts experimental data to a formatted string representation.
    Creates numbered entries with outcomes.

    Args:
        exp_df: DataFrame containing experimental datapoints.
        y_names: List of outcome names.
        masked: If True, uses generic labels (y1, y2, y3, ..) instead of metric names.

    Returns:
        Formatted string with numbered entries containing the performance data.
    """
    if masked:
        y_names = ["y_" + str(i + 1) for i in range(len(y_names))]

    if not markdown:
        return "\n\n".join(
            [
                "{idx}. {y_line}".format(
                    idx=start_idx + i,
                    y_line=", ".join(
                        [
                            "{y_name} = {y_val:.2f}".format(y_name=y_name, y_val=y_val)
                            for y_name, y_val in zip(
                                y_names, exp_df.iloc[i][y_names].values
                            )
                        ]
                    ),
                )
                for i in range(len(exp_df))
            ]
        )
    else:
        df = exp_df.copy()
        df["idx"] = df.index + start_idx
        df = df[["idx"] + y_names]
        return df.to_markdown(index=False, tablefmt="github")


def get_joint_p_accept(text: str, batch_size: int) -> Dict[str, Union[None, list, str]]:
    """
    Extracts the list of p_accept values from text using regex patterns.
    Searches for patterns like 'p_accept: 0.5' or '"p_accept": 0.5'.

    Args:
        text: The text to search for p_accept value.
        batch_size: The number of examples in the batch.

    Returns:
        The extracted list of p_accept value as float, or None if not found and error string.
    """
    error = ""
    jsonl = extract_jsonl_from_text(text)
    if jsonl is None:
        p_accept_ls = None
        reasoning_ls = None
        error = "JSON parsing error"
    else:
        p_accept_ls = [float(json["p_accept"]) for json in jsonl]
        reasoning_ls = [json["reasoning"] for json in jsonl]
        if len(p_accept_ls) != batch_size:
            error = "Incorrect lengths"

    return {"p_accept": p_accept_ls, "reasoning": reasoning_ls, "error": error}


def _elicit_utilities(
    prompt: str,
    to_label_df: pd.DataFrame,
    env: SimulEnvironment,
    llm_client: LLMClient,
    num_responses: int,
):
    n_labels = len(to_label_df)
    # Sample until num_responses responses are obtained
    p_accept_ls = []
    reasoning_ls = []
    it = 0
    print("LLM labelling ...")
    print(prompt)
    while len(p_accept_ls) < num_responses and it < 10:
        num_samples = int((num_responses - len(p_accept_ls)) * (1 + 0.1 * (it + 1)))
        raw_responses_ls = asyncio.run(
            llm_client.get_batch_llm_responses(
                [prompt],
                num_responses=num_samples,
                kwargs={"max_tokens": 1024 + n_labels * 256},
            )
        )[0]
        for r in raw_responses_ls:
            if extract_jsonl_from_text(r) is None:
                print(r)
                print("JSON parsing error")
        results = [
            get_joint_p_accept(response, batch_size=n_labels)
            for response in raw_responses_ls
        ]

        # filter out failed responses
        this_p_accept_ls = [r["p_accept"] for r in results if r["error"] == ""]
        this_reasoning_ls = [r["reasoning"] for r in results if r["error"] == ""]
        print(f"{len(this_p_accept_ls)} / {num_samples}")
        p_accept_ls.extend(this_p_accept_ls)
        reasoning_ls.extend(this_reasoning_ls)
        it += 1

    # reshape
    p_accept_ls = np.array(p_accept_ls)[:num_responses, :].T.tolist()
    p_accept_ls = [np.array(p) for p in p_accept_ls]
    reasoning_ls = np.array(reasoning_ls)[:num_responses, :].T.tolist()
    reasoning_ls = [list(r) for r in reasoning_ls]

    to_label_df["p_accept"] = p_accept_ls
    to_label_df["reasoning"] = reasoning_ls
    to_label_df["true_utility"] = env.get_utility(to_label_df[env.x_names].values)

    # turn p_accept into columns of lists
    to_label_df["p_accept"] = to_label_df["p_accept"].apply(lambda x: x.tolist())

    def get_mean(values):
        if isinstance(values, float) or values is None:
            return values
        else:
            values = [v for v in values if v is not None]
            if len(values) == 0:
                return None
            else:
                return np.mean(values)

    def get_var(values):
        if isinstance(values, float) or values is None:
            return 0.0
        else:
            values = [v for v in values if v is not None]
            if len(values) == 0:
                return None
            else:
                return np.var(values)

    to_label_df["p_accept_mean"] = to_label_df["p_accept"].apply(get_mean)
    to_label_df["p_accept_var"] = to_label_df["p_accept"].apply(get_var)

    # Fill missing values with mean of p_accept_mean/_var
    to_label_df["p_accept_mean"] = to_label_df["p_accept_mean"].fillna(
        to_label_df["p_accept_mean"].mean()
    )
    to_label_df["p_accept_var"] = to_label_df["p_accept_var"].fillna(
        to_label_df["p_accept_var"].mean()
    )

    return to_label_df


def get_proxy_utilities(
    to_label_df: pd.DataFrame,
    context_df: pd.DataFrame,
    env: SimulEnvironment,
    llm_client: LLMClient,
    include_goals: bool = True,
    num_responses: int = 1,
) -> pd.DataFrame:
    # Flatten the feedbacks:
    feedbacks = []
    for f_dict in context_df["feedback"]:
        feedback_df = pd.DataFrame(f_dict, index=[0]).T.reset_index()
        feedbacks.append(feedback_df)
    feedback_df = pd.concat(feedbacks).reset_index(drop=True)
    feedback_df.columns = ["arm_index", "feedback"]

    # Format context data to markdown
    context_data = feedback_df.to_markdown(index=False)

    # Format experiment data to markdown
    experiment_data = to_label_df[["arm_index"] + env.y_names].to_markdown(index=False)

    if include_goals:
        goals = env.get_goal_message()
    else:
        goals = ""

    prompt = SCALAR_UAPPROX.format(
        y_names=env.y_names,
        optimization_goals=goals,
        context_data=context_data,
        experiment_data=experiment_data,
        idx0=to_label_df.iloc[0]["arm_index"],
        idx1=to_label_df.iloc[1]["arm_index"],
        idxn=to_label_df.iloc[len(to_label_df) - 1]["arm_index"],
    )

    to_label_df = _elicit_utilities(
        prompt=prompt,
        to_label_df=to_label_df,
        env=env,
        llm_client=llm_client,
        num_responses=num_responses,
    )

    return to_label_df


def get_gp_proxy_utilities(
    to_label_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    context_df: pd.DataFrame,
    env: SimulEnvironment,
) -> Tuple[pd.DataFrame, SingleTaskGP]:
    """
    Generates proxy utility values for experimental data using a Gaussian process model.
    Uses historical context data to predict acceptance probabilities for new examples.

    Args:
        exp_df: DataFrame containing experimental data with acc, time, mem columns.
        context_df: DataFrame containing historical context data for reference.
        env: The environment object containing utility function and other parameters.
    """
    model = SimpleGPProxyModel(
        input_names=env.y_names,
        target_col="feedback",
        input_transform=Normalize(d=env.n_obj),
    )
    feedbacks = []
    for f_dict in context_df["feedback"]:
        feedback_df = pd.DataFrame(f_dict, index=[0]).T.reset_index()
        feedbacks.append(feedback_df)
    feedback_df = pd.concat(feedbacks).reset_index(drop=True)
    feedback_df.columns = ["arm_index", "feedback"]
    gp_df = exp_df.set_index("arm_index").copy()
    gp_df = gp_df.loc[feedback_df["arm_index"].values, :]
    gp_df["feedback"] = feedback_df["feedback"].values
    pred_p_accept_mean, pred_p_accept_var = model.fit_transform(gp_df, to_label_df)
    to_label_df["p_accept"] = np.nan
    to_label_df["true_utility"] = env.get_utility(to_label_df[env.x_names].values)
    to_label_df.loc[:, "p_accept_mean"] = pred_p_accept_mean
    to_label_df.loc[:, "p_accept_var"] = pred_p_accept_var
    return to_label_df, model.model


def _fit_pairwise_gp_model(
    feedback_df: pd.DataFrame,
    train_df: pd.DataFrame,
    env: SimulEnvironment,
    input_type: str = "y",
) -> PairwiseGP:
    train_df.reset_index(inplace=True, drop=True)
    # map arm_idx to train_df idx
    index_map = dict(zip(train_df["arm_index"], train_df.index))
    train_comp = []
    for _, row in feedback_df.iterrows():
        arm_a, arm_b = row["arm_a"], row["arm_b"]
        feedback = row["feedback"]
        if feedback == 0:
            train_comp.append([index_map[arm_a], index_map[arm_b]])
        else:
            train_comp.append([index_map[arm_b], index_map[arm_a]])
    train_comp = torch.tensor(np.array(train_comp))
    if input_type == "y":
        train_Y = torch.tensor(train_df[env.y_names].values)
        model = PairwiseGP(
            train_Y,
            train_comp,
            input_transform=Normalize(d=train_Y.shape[-1]),
        )
    elif input_type == "x":
        train_X = torch.tensor(train_df[env.x_names].values)
        model = PairwiseGP(
            train_X, train_comp, input_transform=env.get_input_transform()
        )
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    mll = fit_gpytorch_mll(mll)
    return model


def fit_pairwise_gp_model(
    exp_df: pd.DataFrame,
    context_df: pd.DataFrame,
    env: SimulEnvironment,
) -> PairwiseGP:
    feedbacks = []
    for f_dict in context_df["feedback"]:
        feedback_df = pd.DataFrame(f_dict, index=[0]).T.reset_index()
        feedbacks.append(feedback_df)
    feedback_df = pd.concat(feedbacks).reset_index(drop=True)
    feedback_df.columns = ["arm_a", "arm_b", "feedback"]
    unique_arms = list(set(feedback_df["arm_a"]).union(set(feedback_df["arm_b"])))
    train_df = (
        exp_df[exp_df.arm_index.isin(unique_arms)].sort_values(by="arm_index").copy()
    )
    model = _fit_pairwise_gp_model(feedback_df, train_df, env)
    return model


def get_pairwise_gp_proxy_utilities(
    to_label_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    context_df: pd.DataFrame,
    env: SimulEnvironment,
) -> Tuple[pd.DataFrame, PairwiseGP]:
    """
    Generates proxy utility values for experimental data using a Pairwise GP model.
    Uses historical context data to predict acceptance probabilities for new examples.

    Args:
        exp_df: DataFrame containing experimental data with acc, time, mem columns.
        context_df: DataFrame containing historical context data for reference.
        env: The environment object containing utility function and other parameters.
        n_labels: Maximum number of labels to generate (default: np.inf for all).
    """
    model = fit_pairwise_gp_model(exp_df, context_df, env)
    posterior = model.posterior(torch.tensor(to_label_df[env.y_names].values))
    posterior = assert_is_instance(posterior, GPyTorchPosterior)
    mean = posterior.mean.detach().squeeze().numpy()
    var = posterior.variance.detach().squeeze().numpy()
    to_label_df["true_utility"] = env.get_utility(to_label_df[env.x_names].values)
    to_label_df.loc[:, "p_accept_mean"] = mean
    to_label_df.loc[:, "p_accept_var"] = var
    return to_label_df, model


def _get_hf_from_context(
    context_df: pd.DataFrame,
    env: SimulEnvironment,
    include_goals: bool = True,
):
    if include_goals:
        human_feedback = env.get_goal_message()
        human_feedback = human_feedback.replace(
            "\n## Optimization Goal \nThe DM has provided the following information:\n",
            "",
        ).rstrip("\n")
    else:
        human_feedback = ""

    if len(context_df) > 0:
        feedback = {}
        for f in context_df["feedback"]:
            feedback.update(f)
        questions_answers = ""
        for q, a in feedback.items():
            questions_answers += f"- Q: {q} A: {a}\n"

        human_feedback = human_feedback + "\n" + questions_answers

    return human_feedback


def _get_human_feedback(
    context_df: pd.DataFrame,
    env: SimulEnvironment,
    include_goals: bool,
    summarize_feedback: bool = False,
    experiment_data: str = "",
    llm_client: Union[LLMClient, None] = None,
    include_header: bool = True,
):
    if include_header:
        feedback_header = (
            "## Human feedback messages:\n"
            "We have also recieved the following messages from the DM:\n"
        )
    else:
        feedback_header = ""
    human_feedback = _get_hf_from_context(context_df, env, include_goals)
    human_feedback = feedback_header + "\n" + human_feedback
    if summarize_feedback and llm_client is not None:
        summary = _summarize_feedback(
            y_names=env.y_names,
            experiment_data=experiment_data,
            human_feedback=human_feedback,
            llm_client=llm_client,
        )
        human_feedback = (
            human_feedback
            + "\n"
            + "## Summary of optimization goals and feedback:\n"
            + str(summary)
        )
    return human_feedback


def _summarize_feedback(
    y_names: list,
    experiment_data: str,
    human_feedback: str,
    llm_client: LLMClient,
) -> str:
    prompt = QA_SUMMARIZER.format(
        y_names=y_names,
        experiment_data=experiment_data,
        human_feedback=human_feedback,
    )
    summary = None
    it = 0
    while summary is None and it < 3:
        response = asyncio.run(
            llm_client.get_llm_response(prompt, kwargs={"max_tokens": 10000})
        )[0]
        summary = extract_json_from_text(response)
        if summary is not None:
            summary = summary["summary"]
        it += 1

    if summary is None:
        print("Failed to summarize feedback, using raw feedback instead")
        return human_feedback
    else:
        return summary


def get_questions(
    exp_df: pd.DataFrame,
    context_df: pd.DataFrame,
    env: SimulEnvironment,
    selected_arm_index_ls: list,
    n_questions: int,
    llm_client: LLMClient,
    include_goals: bool = True,
    pre_select_data: bool = True,
    prompt_type: str = "pairwise",
) -> list:
    human_feedback = _get_human_feedback(
        context_df, env, include_goals, summarize_feedback=False
    )
    if len(exp_df) > 0:
        experiment_data = exp_df[["arm_index"] + env.y_names]
        if env.cfg.outcome_func == "osy":
            experiment_data = (
                format_outcome_values(experiment_data, env.y_names)
                .to_markdown(index=False)
                .replace(r"\+", " +")
            )
        else:
            experiment_data = experiment_data.to_markdown(index=False)
        if pre_select_data:
            selected_data_str = f"Here are some points it may be useful to ask the decision maker about {selected_arm_index_ls}."
        else:
            selected_data_str = ""

        if prompt_type == "scalar":
            prompt_template = UAPPROX_QUESTIONS_SCALAR
        elif prompt_type == "pairwise":
            prompt_template = UAPPROX_QUESTIONS_PAIRWISE
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        prompt = prompt_template.format(
            y_names=env.y_names,
            experiment_data=experiment_data,
            human_feedback=human_feedback,
            selected_data_str=selected_data_str,
            n_questions=n_questions,
        )
    else:
        prompt = UAPPROX_QUESTIONS_INIT.format(
            y_names=env.y_names, human_feedback=human_feedback, n_questions=n_questions
        )
    response = None
    it = 0
    questions = []

    print(f"[get_questions DEBUG] Generating {n_questions} questions (attempt 1/{3})")
    print(f"[get_questions DEBUG] Prompt length: {len(prompt)} chars")
    print(f"[get_questions DEBUG] Has exp_df data: {len(exp_df) > 0}")

    last_exception = None  # Track the last exception for error reporting
    while response is None and it < 3:
        print(f"[get_questions DEBUG] === Attempt {it+1}/3 ===")
        try:
            llm_response_list = asyncio.run(
                llm_client.get_llm_response(prompt, kwargs={"max_tokens": 10000})
            )
            response = llm_response_list[0] if llm_response_list else None

            print(f"[get_questions DEBUG] LLM response received: {bool(response)}")
            if response:
                print(f"[get_questions DEBUG] Response length: {len(response)} chars")
                print(f"[get_questions DEBUG] Response preview: {response[:200]}...")
        except Exception as e:
            print(f"[get_questions ERROR] LLM call failed: {e}")
            import traceback
            traceback.print_exc()
            last_exception = e  # Store for later
            response = None
            it += 1
            continue

        q_dict = extract_json_from_text(response)
        if q_dict is None:
            print("[get_questions WARNING] JSON parsing failed!")
            print("Raw response:")
            print(response)
        else:
            print(f"[get_questions DEBUG] JSON parsed successfully: {len(q_dict)} questions found")
            print(f"[get_questions DEBUG] Required: {n_questions}, Found: {len(q_dict)}")

        if q_dict is not None and len(q_dict) >= n_questions:
            questions = []
            for q in q_dict.values():
                questions.append(q)
            questions = questions[:n_questions]
            print(f"[get_questions DEBUG] ✓ Successfully generated {len(questions)} questions")
        else:
            if q_dict is not None:
                print(f"[get_questions WARNING] Not enough questions ({len(q_dict)} < {n_questions}), retrying...")
            response = None

        it += 1

    if not questions:
        print(f"[get_questions ERROR] Failed to generate questions after {it} attempts")
        # Re-raise the last exception so caller can see what went wrong
        if last_exception:
            raise last_exception
        else:
            # LLM calls succeeded but returned invalid/insufficient responses
            raise RuntimeError(
                f"Failed to generate {n_questions} questions after {it} attempts. "
                "LLM may be returning invalid JSON or insufficient number of questions. "
                "Check [get_questions DEBUG] logs above for LLM response details."
            )

    return questions


def get_proxy_utilities_qa(
    to_label_df: pd.DataFrame,
    context_df: pd.DataFrame,
    env: SimulEnvironment,
    llm_client: LLMClient,
    num_responses: int,
    include_goals: bool = True,
    summarize_feedback: bool = False,
):
    prompt_template = UAPPROX_QA_LABEL
    experiment_data = to_label_df[["arm_index"] + env.y_names].to_markdown(index=False)
    human_feedback = _get_human_feedback(
        context_df=context_df,
        env=env,
        include_goals=include_goals,
        summarize_feedback=summarize_feedback,
        experiment_data=experiment_data,
        llm_client=llm_client,
    )
    prompt = prompt_template.format(
        y_names=env.y_names,
        experiment_data=experiment_data,
        human_feedback=human_feedback,
        idx0=to_label_df["arm_index"].iloc[0],
        idx1=to_label_df["arm_index"].iloc[1],
        idxn=to_label_df["arm_index"].iloc[len(to_label_df) - 1],
    )
    print(prompt)
    to_label_df = _elicit_utilities(
        prompt=prompt,
        to_label_df=to_label_df,
        env=env,
        llm_client=llm_client,
        num_responses=num_responses,
    )

    return to_label_df


def get_prior_proxy_utilities(
    to_label_df: pd.DataFrame,
    context_df: pd.DataFrame,
    labeled_exp_df: pd.DataFrame,
    env: SimulEnvironment,
    llm_client: LLMClient,
    num_responses: int,
    include_goals: bool = True,
    top_q: float = 0.1,
    prior_type: str = "promising_point",
):
    if len(labeled_exp_df) == 0:
        experiment_data = ""
    else:
        header = "\n## Experimental Outcomes:"
        header += "\nSo far, you have also observed the following inputs x and their estimated utilities:\n"
        experiment_data = labeled_exp_df[env.x_names + ["p_accept_mean"]]
        experiment_data.rename({"p_accept_mean": "utility"}, axis=1, inplace=True)
        experiment_data = experiment_data.to_markdown(index=False)
        experiment_data = header + experiment_data + "\n"

    prompt_template = UAPPROX_PRIOR_LABEL
    data_to_label = to_label_df[["arm_index"] + env.x_names].to_markdown(index=False)
    prior_knowledge = env.get_prior_message(top_q=top_q, prior_type=prior_type)
    human_feedback = _get_human_feedback(
        context_df=context_df[context_df.trial_index == -1],
        env=env,
        include_goals=include_goals,
        summarize_feedback=False,
        experiment_data="",
        llm_client=None,
        include_header=True,
    )
    prompt = prompt_template.format(
        x_names=env.x_names,
        y_names=env.y_names,
        human_feedback=human_feedback,
        prior_knowledge=prior_knowledge,
        experiment_data=experiment_data,
        data_to_label=data_to_label,
        idx0=to_label_df["arm_index"].iloc[0],
        idx1=to_label_df["arm_index"].iloc[1],
        idxn=to_label_df["arm_index"].iloc[len(to_label_df) - 1],
    )
    to_label_df = _elicit_utilities(
        prompt=prompt,
        to_label_df=to_label_df,
        env=env,
        llm_client=llm_client,
        num_responses=num_responses,
    )

    return to_label_df


def _get_pairs(
    to_label_df: pd.DataFrame,
    env: SimulEnvironment,
    max_n_pairs: int = 64,
    pref_model: Union[None, PairwiseGP] = None,
    pref_model_input_type: str = "y",
):
    arm_index_ls = to_label_df["arm_index"].tolist()
    pairs = []
    for i in range(len(arm_index_ls)):
        for j in range(i + 1, len(arm_index_ls)):
            pairs.append((arm_index_ls[i], arm_index_ls[j]))
    if len(pairs) > max_n_pairs and pref_model is not None:
        print("Selecting 64 pairs using EUBO ...")
        if pref_model_input_type == "y":
            data_cols = env.y_names
        elif pref_model_input_type == "x":
            data_cols = env.x_names
        else:
            raise ValueError(f"Unknown input type: {pref_model_input_type}")
        # Use the previous pref_model to select 64 pairs according to EUBO
        pair_data = []
        for idx_a, idx_b in pairs:
            a = to_label_df[to_label_df["arm_index"] == idx_a].iloc[0][data_cols]
            b = to_label_df[to_label_df["arm_index"] == idx_b].iloc[0][data_cols]
            a = torch.tensor(a)
            b = torch.tensor(b)
            pair_data.append(torch.stack([a, b]).unsqueeze(0))  # (1, 2, n_obj)
        pair_data = torch.cat(pair_data, dim=0)  # (n_pairs, 2, n_obj)
        acqf = AnalyticExpectedUtilityOfBestOption(pref_model=pref_model)
        acqf_vals = acqf(pair_data).detach().numpy()
        selected_idx = np.argsort(acqf_vals)[::-1][:max_n_pairs]
        pairs = [pairs[i] for i in selected_idx]
    if len(pairs) > max_n_pairs and pref_model is None:
        print("Selecting 64 random pairs ...")
        # Randomly select 64 pairs
        pairs = random.sample(pairs, max_n_pairs)
    return pairs


def format_outcome_values(df, y_names):
    df = df.copy()
    for y in y_names:
        df[y] = df[y].apply(lambda x: f"{x:.2g}" if x < 0 else f"\\+{x:.2g}")
    return df


def _get_llms_comaprisons(
    pairwise_comp_dict: dict,
    prompts: list,
    pairs: list,
    llm_client: LLMClient,
    num_responses: int = 1,
):
    it = 0
    new_pairwise_comp_dict = {}
    while len(new_pairwise_comp_dict.keys()) < 0.9 * len(pairs) and it < 3:
        response = asyncio.run(
            llm_client.get_batch_llm_responses(
                prompts, num_responses=num_responses, kwargs={"max_tokens": 3000}
            )
        )
        print(response[0][0])
        for pair, raw_response_ls in zip(pairs, response):
            predictions = [extract_json_from_text(r) for r in raw_response_ls]
            predictions = [int(r["answer"]) for r in predictions if r is not None]
            if len(predictions) == 0:
                continue  # stop processing the answers and retry
            if pair in new_pairwise_comp_dict.keys():
                new_pairwise_comp_dict[pair].extend(predictions)
            else:
                new_pairwise_comp_dict[pair] = predictions
        it += 1
    pairwise_comp_dict.update(new_pairwise_comp_dict)
    return pairwise_comp_dict


def _get_pairwise_llm_comparisons(
    to_label_df: pd.DataFrame,
    context_df: pd.DataFrame,
    env: SimulEnvironment,
    llm_client: LLMClient,
    include_goals: bool = True,
    num_responses: int = 1,
    pref_model: Union[None, PairwiseGP] = None,
    pref_model_input_type="y",
    summarize_feedback: bool = False,
) -> list:
    # Format experiment data to markdown
    if env.cfg.outcome_func == "osy":
        experiment_data = (
            format_outcome_values(
                to_label_df[["arm_index"] + env.y_names],
                env.y_names,
            )
            .to_markdown(index=False)
            .replace(r"\+", " +")
        )
    else:
        experiment_data = to_label_df[["arm_index"] + env.y_names].to_markdown(
            index=False
        )

    # Flatten the feedbacks:
    human_feedback = _get_human_feedback(
        context_df=context_df,
        env=env,
        include_goals=include_goals,
        summarize_feedback=summarize_feedback,
        experiment_data=experiment_data,
        llm_client=llm_client,
    )

    print("Getting pairwise comparisons for utility estimation")

    pairs = _get_pairs(
        to_label_df=to_label_df,
        env=env,
        pref_model=pref_model,
        pref_model_input_type=pref_model_input_type,
        max_n_pairs=64,
    )

    # Create a batch of prompts
    prompts = []
    for pair in pairs:
        pair_str = to_label_df.set_index("arm_index").loc[pair, :].reset_index().copy()
        pair_str.index = ["option_0", "option_1"]
        pair_str = pair_str[["arm_index"] + env.y_names]
        if env.cfg.outcome_func == "osy":
            pair_str = (
                format_outcome_values(pair_str, env.y_names)
                .to_markdown()
                .replace(r"\+", " +")
            )
        else:
            pair_str = pair_str.to_markdown()
        prompt = PAIRWISE_UAPROX.format(
            y_names=env.y_names,
            human_feedback=human_feedback,
            experiment_data=experiment_data,
            pair_str=pair_str,
        )
        if "qwen3" in llm_client.model:
            prompt += "Make your thinking process succinct and remember about the ```json header!"
        prompts.append(prompt)
    print(prompts[0])

    # Split into chunnks of 16
    chunk_size = 8  # 16
    n_pairs = len(pairs)
    chunked_prompts = [
        prompts[i : i + chunk_size] for i in range(0, n_pairs, chunk_size)
    ]
    chunked_pairs = [pairs[i : i + chunk_size] for i in range(0, n_pairs, chunk_size)]

    # Obtain LLM's responses
    pairwise_comp_dict = {}
    for i, (prompts, pairs) in enumerate(zip(chunked_prompts, chunked_pairs)):
        print(f"Generating comparisons for chunk {i + 1}/{len(chunked_prompts)}")
        pairwise_comp_dict = _get_llms_comaprisons(
            pairwise_comp_dict=pairwise_comp_dict,
            prompts=prompts,
            pairs=pairs,
            llm_client=llm_client,
            num_responses=num_responses,
        )

    print(f"Retrieved {len(pairwise_comp_dict.keys())}/{n_pairs} comparisons.")
    if len(pairwise_comp_dict.keys()) < 0.9 * len(pairs):
        raise ValueError("LLM error. Not enough pairwise comparisons retrieved.")

    pairwise_comp = []
    for pair, predictions in pairwise_comp_dict.items():
        for p in predictions:
            if p == 1:
                pairwise_comp.append((pair[0], pair[1], 1))
            else:
                pairwise_comp.append((pair[0], pair[1], 0))

    # Compute true pairwise preferences
    to_label_df = to_label_df.copy()
    to_label_df["true_utility"] = env.get_utility_from_y(
        to_label_df[env.y_names].values
    )
    true_pairwise_comp = []
    for i in range(len(pairwise_comp)):
        a, b, _ = pairwise_comp[i]
        a_utility = to_label_df[to_label_df["arm_index"] == a]["true_utility"].iloc[0]
        b_utility = to_label_df[to_label_df["arm_index"] == b]["true_utility"].iloc[0]
        if a_utility > b_utility:
            true_pairwise_comp.append(0)
        else:
            true_pairwise_comp.append(1)

    predictions = np.array([p[2] for p in pairwise_comp])
    true = np.array(true_pairwise_comp)
    accuracy = np.mean(predictions == true)
    print(f"LLM's Pairwise Accuracy: {accuracy:.2f}")

    return pairwise_comp


def get_pairwise_fedback_df(
    to_label_df: pd.DataFrame,
    context_df: pd.DataFrame,
    env: SimulEnvironment,
    llm_client: LLMClient,
    include_goals: bool = True,
    num_responses: int = 1,
    pref_model: Union[None, PairwiseGP] = None,
    pref_model_input_type: str = "y",
    summarize_feedback: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_comp = _get_pairwise_llm_comparisons(
        to_label_df=to_label_df,
        context_df=context_df,
        env=env,
        llm_client=llm_client,
        include_goals=include_goals,
        num_responses=num_responses,
        pref_model=pref_model,
        pref_model_input_type=pref_model_input_type,
        summarize_feedback=summarize_feedback,
    )
    feedback_df = pd.DataFrame(pairwise_comp, columns=["arm_a", "arm_b", "feedback"])
    unique_arms = list(set(feedback_df["arm_a"]).union(set(feedback_df["arm_b"])))
    train_df = (
        to_label_df[to_label_df.arm_index.isin(unique_arms)]
        .sort_values(by="arm_index")
        .copy()
    )
    return feedback_df, train_df


def get_pairwise_llm_proxy_utilities(
    to_label_df: pd.DataFrame,
    context_df: pd.DataFrame,
    env: SimulEnvironment,
    llm_client: LLMClient,
    include_goals: bool = True,
    num_responses: int = 1,
    pref_model: Union[None, PairwiseGP] = None,
    pref_model_input_type: str = "y",
    summarize_feedback: bool = False,
) -> Tuple[pd.DataFrame, PairwiseGP]:
    """
    Generates proxy utility values for experimental data using a pairwise LLM comparisons.
    Returns the dataframes with the predicted probabilities and the fitted pairwise GP model.
    """
    feedback_df, train_df = get_pairwise_fedback_df(
        to_label_df=to_label_df,
        context_df=context_df,
        env=env,
        llm_client=llm_client,
        include_goals=include_goals,
        num_responses=num_responses,
        pref_model=pref_model,
        pref_model_input_type=pref_model_input_type,
        summarize_feedback=summarize_feedback,
    )
    model = _fit_pairwise_gp_model(feedback_df, train_df, env)
    to_label_df["p_accept"] = np.nan
    to_label_df["true_utility"] = env.get_utility(to_label_df[env.x_names].values)
    posterior = model.posterior(torch.tensor(to_label_df[env.y_names].values))
    posterior = assert_is_instance(posterior, GPyTorchPosterior)
    mean = posterior.mean.detach().squeeze().numpy()
    var = posterior.variance.detach().squeeze().numpy()
    to_label_df["p_accept_mean"] = mean
    to_label_df["p_accept_var"] = var
    return to_label_df, model


def get_pairwise_proxy_and_pref_models(
    to_label_df: pd.DataFrame,
    context_df: pd.DataFrame,
    env: SimulEnvironment,
    llm_client: LLMClient,
    include_goals: bool = True,
    num_responses: int = 1,
    pref_model: Union[None, PairwiseGP] = None,
    pref_model_input_type: str = "y",
    summarize_feedback: bool = False,
) -> Tuple[pd.DataFrame, PairwiseGP, PairwiseGP]:
    """
    Fits PariwiseGP models (X -> U and Y -> U) using LLM pairwise comparisons between outcomes.
    Returns the dataframes with the predicted probabilities and the fitted pairwise GP model.
    """
    feedback_df, train_df = get_pairwise_fedback_df(
        to_label_df=to_label_df,
        context_df=context_df,
        env=env,
        llm_client=llm_client,
        include_goals=include_goals,
        num_responses=num_responses,
        pref_model=pref_model,
        pref_model_input_type=pref_model_input_type,
        summarize_feedback=summarize_feedback,
    )
    # Fit the proxy model X -> U
    proxy_model = _fit_pairwise_gp_model(feedback_df, train_df, env, input_type="x")
    # Predict on to_label_df
    to_label_df["p_accept"] = np.nan
    to_label_df["true_utility"] = env.get_utility(to_label_df[env.x_names].values)
    posterior = proxy_model.posterior(torch.tensor(to_label_df[env.x_names].values))
    posterior = assert_is_instance(posterior, GPyTorchPosterior)
    mean = posterior.mean.detach().squeeze().numpy()
    var = posterior.variance.detach().squeeze().numpy()
    to_label_df["p_accept_mean"] = mean
    to_label_df["p_accept_var"] = var
    # Fit the pref model Y -> U
    pref_model = _fit_pairwise_gp_model(feedback_df, train_df, env, input_type="y")
    return to_label_df, proxy_model, pref_model
