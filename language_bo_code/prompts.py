PIECEWISE_LINEAR_INTRO = """
Suppose you are a decision maker evaluating the results of a multi-objective opimization problem.

You are given a set of multi-dimensional outcomes y = [y_1, y_2, ..., y_{env_kwargs.n_obj}].

These outcome's range are:
    lower: {env_kwargs.y_lower}
    upper: {env_kwargs.y_upper}

Your utility function is a sum of contributions of each term: utility = \\sum_i contrib(y_i).

For each outcome y_i, the contribution h_i, to your utility function is given by a piece-wise linear function, which grows with a slope beta_{{1, i}} below the threshold t_i and with slope beta_{{2, i}} above t_i, so that

    contrib(y_i) = beta_{{1, i}} * y_i + (beta_{{2, i}} - beta_{{1, i}}) * t_i for y_i < t_i
    contrib(y_i) = beta_{{2, i}} * y_i for y_i >= t_i

We have:
    t  = {env_kwargs.t}
    beta_1 = {env_kwargs.beta_1}
    beta_2 = {env_kwargs.beta_2}


Since for each i we have beta_{{1, i}} > beta_{{2, i}}. This utility functions represents sharp declines in utility when outcomes fall below thresholds and diminishing marginal returns on outcomes above the thresholds.

Thus, ideally, all outcomes should be above their respective thresholds.

The values of the slopes below and above threshold define the importance of increaseing each outcome metric. The higher the slope, the more important the outcome metric is.
For instance, if beta_{{1, i}} >> beta_{{1, j}} and y_i < t_i, y_j < t_j, then it is more important to increase metric y_i (to achieve t_i) than it is to increase metric y_j (to achieve t_j).
Thus you should consider the relative values of beta_1's to determine the relative importance of metrics y_1, y_2, ..., y_{env_kwargs.n_obj} in achievng their respsective thresholds.
It is NOT the threshold values that indicate the importance of metrics.
"""

PIECEWISE_LINEAR_ANSWERS = (
    PIECEWISE_LINEAR_INTRO
    + """

You have observed the following outcomes with their corresponding utility values and contributions to the overall utility.

## Outcomes:

{outcomes_markdown}

The utility values are on a scale [0, 1], where (1 - very satisfied, 0.5 - neutral, 0 - very dissatisfied).

Based on the above information, provide answers to the following questions:

## Questions:

{questions_str}

Return your final answer as a json file with the following format:
```json
{{
    "q1" : <answer to q1>,
    ...
    "q{n_questions}" : <answer to q{n_questions}>
}}
```
Before providing your final answers, analyse the shape of the utility function in light of the questions posed.
In your final answers, you cannot reveal the explicit formula of the utility function.
The form and the values of the utility functions is a "latent" feature of the human expert, thus you should not refer to it explicitly or even mention its existance.
You can/should reveal the threhold values t_i. You should not reveal the exact values of the slopes beta_1 and beta_2, but they indicate the relative importance of each outcome, which you should reveal.
Do not use the word "slope" in your answer.
State your answers in the first person (you are the decision maker).
Avoid vacous statements.
"""
)


PIECEWISE_LINEAR_OSY_INTRO = """
Suppose you are a decision maker evaluating the results of a multi-objective opimization problem.

You are given a set of multi-dimensional outcomes y = [y_1, y_2, y_3,..., y_8]

y_1, y_2 take values in [0, 1] while y_3, ..., y_8 take both positive and negative values.

Your utility function is a sum of contributions of each term: utility = \\sum_i contrib(y_i).

For each outcome y_i, the contribution h_i, to your utility function is given by a piece-wise linear function, which grows with a slope beta_{{1, i}} below the threshold t_i and with slope beta_{{2, i}} above t_i, so that

    contrib(y_i) = beta_{{1, i}} * y_i + (beta_{{2, i}} - beta_{{1, i}}) * t_i for y_i < t_i
    contrib(y_i) = beta_{{2, i}} * y_i for y_i >= t_i

We have:
    t  = {env_kwargs.t}
    beta_1 = {env_kwargs.beta_1}
    beta_2 = {env_kwargs.beta_2}


Since for each i we have beta_{{1, i}} > beta_{{2, i}}. This utility functions represents sharp declines in utility when outcomes fall below thresholds and diminishing marginal returns on outcomes above the thresholds.

Thus, ideally, all outcomes should be above their respective thresholds.

The values of the slopes below and above threshold define the importance of increaseing each outcome metric. The higher the slope, the more important the outcome metric is.
For instance, if beta_{{1, i}} >> beta_{{1, j}} and y_i < t_i, y_j < t_j, then it is more important to increase metric y_i (to achieve t_i) than it is to increase metric y_j (to achieve t_j).
It is NOT the threshold values that indicate the importance of metrics.
"""

PIECEWISE_LINEAR_OSY_ANSWERS = (
    PIECEWISE_LINEAR_OSY_INTRO
    + """

You have observed the following outcomes with their corresponding utility values and contributions to the overall utility.

## Outcomes:

{outcomes_markdown}

The utility values are on a scale [0, 1], where (1 - very satisfied, 0.5 - neutral, 0 - very dissatisfied).

Based on the above information, provide answers to the following questions:

## Questions:

{questions_str}

Return your final answer as a json file with the following format:
```json
{{
    "q1" : <answer to q1>,
    ...
    "q{n_questions}" : <answer to q{n_questions}>
}}
```
Before providing your final answers, analyse the shape of the utility function in light of the questions posed.
In your final answers, you cannot reveal the explicit formula of the utility function.
The form and the values of the utility functions is a "latent" feature of the human expert, thus you should not refer to it explicitly or even mention its existance.
You can/should reveal the threhold values t_i. You should not reveal the exact values of the slopes beta_1 and beta_2, but they indicate the relative importance of each outcome, which you should reveal.
Do not use the word "slope" in your answer.
State your answers in the first person (you are the decision maker).
Avoid vacous statements.

## Examples:
Q: How do you prioritize the outcomes when it comes to achieving the thresholds?
A: I prioritize increasing the outcomes y_3, ..., y_8, then y_2, followed by y_1.

Q: Once all outcomes have reached their respective thresholds, how do you value further improvements in each outcome?
A: I prioritize improvements in y_3, ... y_8. Then improvements in y_2 followed by y_1.

Q: Considering option a = [+0.6, +0.1, +0.2, -0.04, -0.01, +0.2, +0.01, +0.1] and option b = [+0.1, +0.2, +0.01, +0.01, +0.2, +0.1, +0.01, +0.05] which one do you prefer and why?
A: I prefer option b since outcomes y_3, ..., y_8 meet their thresholds.
"""
)


L1_INTRO = """
Suppose you are a decision maker evaluating the results of a multi-objective opimization problem.

You are given a set of outcomes y = [y_1, y_2, ..., y_{env_kwargs.n_obj}] taking values in [0, 1]

Your utility function is based on the L1 norm of the outcomes y and the optimum opt_y={env_kwargs.opt_y}.

Precisely, utility(y) = exp(-|y - opt_y|), so that the utility is on a scale from 0 to 1 where (1 - very satisfied, 0.5 - neutral, 0 - very dissatisfied).

This utility function weigths the distance of each outcome from the optimum equally and the negative contribution of each outcome is proportional to the distance from the optimum.
"""

L1_ANSWERS = (
    L1_INTRO
    + """

You have observed the following outcomes with their corresponding utility values:

## Outcomes:

{outcomes_markdown}

The utility values are on a scale [0, 1], where (1 - very satisfied, 0.5 - neutral, 0 - very dissatisfied).

Based on the above information, provide answers to the following questions:

## Questions:

{questions_str}

Return your final answer as a json file with the following format:
```json
{{
    "q1" : <answer to q1>,
    ...
    "q{n_questions}" : <answer to q{n_questions}>
}}
```

In your final answers, you cannot reveal the explicit formula of the utility function.
The form and the values of the utility functions is a "latent" feature of the human expert, thus you should not refer to it explicitly or even mention its existance.
You are emulating a human, thus you should not describe the functional form of the utility function.
State your answers in the first person (you are the decision maker).
"""
)


BETA_INTRO = """
Suppose you are a decision maker evaluating the results of a multi-objective opimization problem.

You are given a set of outcomes y = [y_1, y_2, ..., y_{env_kwargs.n_obj}] taking values in [0, 1]

The utility function is a product of beta distributions for each outcome y_i, so that

utility(y) = prod_i beta(y_i; alpha_i, beta_i)

The parameters alpha and beta are given by:

    alpha = {env_kwargs.alpha}
    beta = {env_kwargs.beta}

By design, the utility function takes values between 0 and 1 and it defines your degree of satisfaction with the results. (1 - very satisfied, 0.5 - neutral, 0 - very dissatisfied).
"""


BETA_HF_ANSWERS = (
    BETA_INTRO
    + """

Now, you observe a new set of outcomes with their corresponding utility values and contributions to the overall utility.

The values of individual contributions are given by: contrib_y_i = beta(y_i; alpha_i, beta_i)

## Outcomes:

{outcomes_markdown}

The utility values are on a scale [0, 1], where (1 - very satisfied, 0.5 - neutral, 0 - very dissatisfied).

Based on the above information, provide answers to the following questions:

## Questions:

{questions_str}

Return your final answer as a json file with the following format:
```json
{{
    "q1" : <answer to q1>,
    ...
    "q{n_questions}" : <answer to q{n_questions}>
}}
```

In your final answers, you cannot reveal the explicit formula of the utility function.
The form and the values of the utility functions is a "latent" feature of the human expert, thus you should not refer to it explicitly or even mention its existance.
Because you are emulating a human, you should not describe the functional form of the utility function.
E.g.,
- you cannot mention that the utility function is a *product* of contributions,
- you cannot mention that the shape of the utility function is a beta cdf,
- you cannot use the words "alpha" or "beta" in your answer.
You can, however,
- reveal the relative importance of metrics based on the values of alpha and beta,
- state qualitatively that even one outcome value close to zero has a big negatice effect on your overall satisfaction.,
etc.
State your answers in the first person (you are the decision maker).
"""
)

OSY_INTRO = """
Suppose you are a decision maker evaluating the results of a multi-objective opimization problem.

You are given a set of outcomes y = [y_1, y_2, ..., y_8].

y_1 and y_2 take values in [0, 1] while y_3, ..., y_8 can take both positive and negative values.

Your aim is to maximize y_1 and y_2, while making sure that none of y_3, ..., y_8 are below zero.

The exact form of the utility function is given by:

utility = contrib(y_1, y_2) * prod_{{i=3}}^8 contrib(y_i),

where contrib(y_1, y_2) = (exp(y_1) + exp(y_2)) / exp(2)

contrib(y_i) = sigmoid(100 * y_i) for i in [3, ..., 8]

By the multiplicative nature of the utility function, if at least one of y_3, ..., y_8 is below zero, the overall utility is near zero, it is okay though for y_3, ..., y_8 to be equal to zero.

You weigh all outcomes y_3, ..., y_8 equally. You also weigh y_1 and y_2 equally, what matters is the value of exp(y_1) + exp(y_2) and not whether y_1 or y_2 is larger, the difference between y_1 and y_2 also does not matter.

By design, the utility function takes values between 0 and 1 and it defines your degree of satisfaction with the results. (1 - very satisfied, 0 - very dissatisfied).
"""

OSY_ANSWERS = (
    OSY_INTRO
    + """
Now, you observe a new set of outcomes with their corresponding utility values and contributions to the overall utility.

## Outcomes:

{outcomes_markdown}

Based on the above information, provide answers to the following questions:

## Questions:

{questions_str}

Return your final answer as a json file with the following format:
```json
{{
    "q1" : <answer to q1>,
    ...
    "q{n_questions}" : <answer to q{n_questions}>
}}
```

In your final answers, you cannot reveal the explicit formula of the utility function.
The form and the values of the utility functions is a "latent" feature of the human expert, thus you should not refer to it explicitly or even mention its existance.
State your answers in the first person (you are the decision maker).
"""
)

THERMO_ANSWERS = """
You are describing your thermal comfort.
The metrics describing the thermal conditions are: {y_names}
You want to keep all metrics within the specified comfortable range:
- Overall dissatisfaction (PPD, % dissatisfied):
    Comfortable if ≤{env_kwargs.ppd[0]}%.
    Tolerable if >{env_kwargs.ppd[0]}% and <{env_kwargs.ppd[1]}%.
    Unacceptable if ≥{env_kwargs.ppd[1]}%.
- Draft risk (DR, % draft risk):
    Comfortable if ≤{env_kwargs.dr[0]}%.
    Tolerable if >{env_kwargs.dr[0]}% and <{env_kwargs.dr[1]}%.
    Unacceptable if ≥{env_kwargs.dr[1]}%.
- Vertical temperature difference (dT_vert, K, head↔ankles):
    Comfortable if ≤{env_kwargs.dT[0]} K.
    Tolerable if >{env_kwargs.dT[0]} K and <{env_kwargs.dT[1]} K.
    Unacceptable if ≥{env_kwargs.dT[1]} K.
- Radiant temperature asymmetry (dT_pr, K, warm ceiling):
    Comfortable if ≤{env_kwargs.rad[0]} K.
    Tolerable if >{env_kwargs.rad[0]} K and <{env_kwargs.rad[1]} K.
    Unacceptable if ≥{env_kwargs.rad[1]} K.
- Floor temperature (T_floor, °C):
    Comfortable if between {env_kwargs.tf[1]} and {env_kwargs.tf[2]} °C.
    Tolerable if >{env_kwargs.tf[0]} °C and <{env_kwargs.tf[1]} °C or >{env_kwargs.tf[2]} °C and <{env_kwargs.tf[3]} °C.
    Unacceptable if ≤{env_kwargs.tf[0]} °C or ≥{env_kwargs.tf[3]} °C.

Now, you observe a set of thermal outcomes with their corresponding utility values and their individual contributions contrib(y_i) to your overall utility.
Your utility is equal to the product of the individual contributions, hence, even if one contrib(y_i) is close to zero, it has a big impact on your overall comfort level.
The utilities are on the scale from 0 to 1 and indicate your overall comfort level (0: very dissatisfied, 0.5 - neutral, 1 - very satisfied)

## Outcomes:

{outcomes_markdown}

Based on the above information, provide answers to the following questions:

## Questions:

{questions_str}

Return your final answer as a json file with the following format:
```json
{{
    "q1" : <answer to q1>,
    ...
    "q{n_questions}" : <answer to q{n_questions}>
}}
```

In your final answers:
- You cannot reveal the explicit values of the utility function. The form and the values of the utility functions is a "latent" feature of the human expert, thus you should not refer to it explicitly or even mention its existance.
- State your answers in the first person, impersonate a non-technical human
- If asked about specific comfort ranges, do not provide an exhaustive answer, be more qualitative than quantitative.
- The *only* exception is the floor temperature, where you can state some of your preferences quantiativelly.
- Avoid citing PPD, DR, or any formula names explicitly—describe feelings (e.g., “drafty at my neck,” “warm from the ceiling,” “feet feel cold”).
- Do not ever mention PPD explicitly in your answers, you can only qualitatively describe your overall satisfaction with the thermal conditions.
"""

THERMO_A_ANSWERS = (
    "You are impersonating a human office occupant A professional working in a co-working space, wearing light office attire (long-sleeve shirt and trousers, no jacket)."
    + THERMO_ANSWERS
)

THERMO_B_ANSWERS = (
    "You are impersonating a summer athlete training indoors after a run; prefers cooler, high-air-movement spaces but still sensitive to hot radiance from above and hot floors."
    + THERMO_ANSWERS
)


SCALAR_UAPPROX = """
You are an expert in determining whether a human decision maker (DM) is going to be satisfied with a set of experimental outcomes y = {y_names}.
{optimization_goals}
Given the historical runs of the experiment and the corresponding human feedback messages, you must return the probability that the DM will be satisfied with the given set of outcomes.

## All experimental outcomes:

{experiment_data}

## Human feedback messages:

{context_data}

First, analyse the human feedback messages to understand the DM's preferences.
Then, Provide your predictions for all y's in the set of all experimental outcomes above.
Return your final answer as a jsonl file with the following format:

```jsonl
{{
    "arm_index": "{idx0}",
    "reasoning": <reasoning>,
    "p_accept": <probability>
}}
{{
    "arm_index": "{idx1}",
    "reasoning": <reasoning>,
    "p_accept": <probability>
}}
...
{{
    "arm_index": "{idxn}",
    "reasoning": <reasoning>,
    "p_accept": <probability>
}}
```
Where <reasoning> should be a short reasoning for your prediction and <probability> should be your best estimate for the probability between 0 and 1 that the DM will be satisfied with the corresponding outcome.

Provide your predictions for ALL y's in the set of experimental outcomes above. That is, for EACH outcome from {idx0}. to {idxn}.
Do not generate any Python code. Just return your predictions as plain text. Do not forget the ```jsonl header.
"""

PAIRWISE_UAPROX = """
You are an expert in determining whether a human decision maker (DM) is going to be satisfied with a set of experimental outcomes y = {y_names}.

## Your task:
Given a pair of outcomes, your goal is to decide which one is more preferable, given the DM's optimization goals and feedback messages.

## Experimental outcomes:
So far, we have obtained the following experimental outcomes:

{experiment_data}

{human_feedback}

## Expected output:
The two outcomes you will be comparing are option_0 and option_1 given below:

{pair_str}

First, analyse the human feedback messages to understand the DM's preferences.
Then, provide your prediction as a json file with the following format:
```json
{{
    "reasoning": "Your reasoning about the DM's preferences and option_0 vs. option_1. Do not insert new lines in your reasoning.",
    "answer" : 0 or 1
}}
```
where in "answer" you should return 0 if option_0 is preferred, or 1 if option_1 is preferred.
Return just the json file (with the header ```json), nothing else.
"""

UAPPROX_QUESTIONS_INIT = """
You are an expert in determining whether a human decision maker (DM) is going to be satisfied with a set of experimental outcomes y = {y_names}.

{human_feedback}

## Your task:
Given the above your task is to predict the probability of the decision maker being satisfied with the experimental outcomes.

In order to better understand the decision maker's utility function you want to ask them about their optimization goals.

Provide a list of questions you would ask the decision maker to better understand their internal utility model.

Return your final answer as a json file with the following format containing exactly {n_questions} most important questions. Do not forget the ```json header:
```json
{{
    "q1" : <question1>,
    ...
    "q{n_questions}" : <question{n_questions}>
}}
```
"""

UAPPROX_QUESTIONS_SCALAR = """
You are an expert in determining whether a human decision maker (DM) is going to be satisfied with a set of experimental outcomes y = {y_names}.

## Experimental outcomes:
So far, we have obtained the following experimental outcomes:

{experiment_data}

{human_feedback}

## Your task:
Given the above your task is to predict the probability of the decision maker being satisfied with the experimental outcomes.

In order to better understand the decision maker's utility function you want to ask them about their optimization goals or for feedback regarding specific experimental outcomes.

{selected_data_str}

First, analyse the decision maker's goals and feedback messages to understand their overall preferences.
Then, provide a list of questions you would ask the decision maker to better understand their internal utility model.
Your questions can be either general or referring to specific outcomes. For instance, you may ask the decision maker:
- questions clairfying the optimzation objective,
- to rank two (or more) outcomes,
- how to improve certain outcomes,
- for a likert-scale rating regarding a specific outcome,
- etc.
When referring to specific outcomes, always state the arm_index involved.
Your questions should help you predict the probability of the decision maker being satisfied with the experimental outcomes provided above.

Return your final answer as a json file with the following format containing exactly {n_questions} most important questions. Do not forget the ```json header:
```json
{{
    "q1" : <question1>,
    ...
    "q{n_questions}" : <question{n_questions}>
}}
```
"""

UAPPROX_QUESTIONS_PAIRWISE = """
You are an expert in determining w fhether a human decision maker (DM) is going to be satisfied with a set of experimental outcomes y = {y_names}.

## Experimental outcomes:
So far, we have obtained the following experimental outcomes:

{experiment_data}

{human_feedback}

## Your task:
Given the above your task is to predict pairwise preferences between experimental outcomes.

In order to better understand the decision maker's utility function you want to ask them about their optimization goals or for feedback regarding specific experimental outcomes.

{selected_data_str}

First, analyse the decision maker's goals and feedback messages to understand their overall preferences.
Then, provide a list of questions you would ask the decision maker to better understand their internal utility model.
Your questions can be either general or referring to specific outcomes. For instance, you may ask the decision maker:
- questions clairfying the optimzation objective,
- to rank two (or more) outcomes,
- how to improve certain outcomes,
- for a likert-scale rating regarding a specific outcome,
- etc.
When referring to specific outcomes, always state the arm_index involved.
Your questions should help you predict pairwise preferences between any two experimental outcomes from the set of experimental outcomes provided above.

Return your final answer as a json file with the following format containing exactly {n_questions} most important questions. Do not forget the ```json header:
```json
{{
    "q1" : <question1>,
    ...
    "q{n_questions}" : <question{n_questions}>
}}
```
"""

UAPPROX_QA_LABEL = """
You are an expert in determining whether a human decision maker (DM) is going to be satisfied with a set of experimental outcomes y = {y_names}.

## Your task:
Your task is to predict the probability of the decision maker being satisfied with the experimental outcomes, given the DM's optimization goals and feedback messages.

## Experimental outcomes:
So far, we have obtained the following experimental outcomes:

{experiment_data}

{human_feedback}

## Expected output:
First, analyse the human feedback messages to understand the DM's preferences.
Then, provide your predictions for all y's in the set of all experimental outcomes above.
Return your final answer as a jsonl file with the following format:

```jsonl
{{
    "arm_index": "{idx0}",
    "reasoning": <reasoning>,
    "p_accept": <probability>
}}
{{
    "arm_index": "{idx1}",
    "reasoning": <reasoning>,
    "p_accept": <probability>
}}
...
{{
    "arm_index": "{idxn}",
    "reasoning": <reasoning>,
    "p_accept": <probability>
}}
```
Where <reasoning> should be a short reasoning for your prediction and <probability> should be your best estimate for the probability between 0 and 1 that the DM will be satisfied with the corresponding outcome.

Provide your predictions for ALL y's in the set of experimental outcomes above. That is, for EACH outcome from {idx0}. to {idxn}.
Do not generate any Python code. Just return your predictions as plain text. Do not forget the ```jsonl header.
"""


UAPPROX_PRIOR_LABEL = """
You are an expert in determining whether a human decision maker (DM) is going to be satisfied with a set of experimental outcomes y = {y_names} determined by a set of parameters x = {x_names}.

## Prior knowledge:
You have obtained the following prior knowledge about the experiment:
{prior_knowledge}
{experiment_data}
{human_feedback}

## Your task:
Predict the probability of the parameters x leading to satisfactory outcomes y.

## Data to label:
{data_to_label}

First, analyse the human feedback messages to understand the DM's preferences and prior knowledge.
Then, provide your predictions for all x's in the set of data to label above.
Return your final answer as a jsonl file with the following format:

```jsonl
{{
    "arm_index": "{idx0}",
    "reasoning": <reasoning>,
    "p_accept": <probability>
}}
{{
    "arm_index": "{idx1}",
    "reasoning": <reasoning>,
    "p_accept": <probability>
}}
...
{{
    "arm_index": "{idxn}",
    "reasoning": <reasoning>,
    "p_accept": <probability>
}}
```
Where <reasoning> should be a short reasoning for your prediction and <probability> should be your best estimate for the probability between 0 and 1 that the DM will be satisfied with the corresponding outcome.

Provide your predictions for ALL x's in the set of data to label above. That is, for EACH outcome from {idx0}. to {idxn}.
Do not generate any python code for labeling. First reason about the inputs, outputs and the utility function and then simply provide your predictions for ALL arms as a jsonl file.
"""

QA_SUMMARIZER = """
You are an expert in determining whether a human decision maker (DM) is going to be satisfied with a set of experimental outcomes y = {y_names}.

## Experimental outcomes:
So far, we have obtained the following experimental outcomes:

{experiment_data}

{human_feedback}

## Your task:
Given the above your task is to summarize the human feedback messages into a clear description of the DM's optimization goals.
Make your summary as quantitative as possible so that it can be easily used for utility estimation.

After analysis the human feedback messages, return your final answer as a json file with the following format:
```json
{{
    "summary": <summary>
}}
```
Remember about the ```json header!
"""

CANDIDATE_SAMPLER = """
You are performing optimization of a utility function u(x) = g(y) = g(f(x)), where x is a vector of parameters: x = {x_names} and y = f(x) = {y_names} is a vector of outcomes.
Each dimensions of x is in the range [0, 1].
Your goal is to find the parameters x that maximize the utility.

{prior_knowledge}
{experiment_data}
{human_feedback}

## Your task:
Given the above your task is the generate a set of {n_candidates} candidate parameters x for the next round of experimentation.
{ei_message}

First, analyse the information above, then return your final answer as a json file with the following format:
```json
{{
    "0": <candidate0>,
    "1": <candidate1>,
    ...
    "{n}": <candidate{n}>,
}}
```
Where each <candidatei> is a list of the candidate parameter values in [0, 1].
Do not write a python code for candidate generation. Just return the required json.
Do not add any comments to your json. Remember about the ```json header.
"""


FULL_CANDIDATE_SAMPLER = """
You are performing optimization of a utility function u(x) = g(y) = g(f(x)), where x is a vector of parameters: x = {x_names} and y = f(x) = {y_names} is a vector of outcomes.
Each dimensions of x is in the range [0, 1].
Your goal is to find the parameters x that maximize the utility.

{prior_knowledge}
{experiment_data}
{human_feedback}

## Your task:
Given the above your task is the generate a set of {n_candidates} candidate parameters x for the next round of experimentation.
First, analyse the human feedback messages to understand the DM's preferences.
Then, generate a set of {n_candidates} candidate parameters x, trading-off exploration and exploitation.
Return your final answer as a json file with the following format:
```json
{{
    "0": <candidate0>,
    "1": <candidate1>,
    ...
    "{n}": <candidate{n}>,
}}
```
Where each <candidatei> is a list of the candidate parameter values: {x_names}, each in [0, 1].
Do not write a python code for candidate generation. Just return the required json.
Do not add any comments to your json.
"""


HF_PROMPTS = {
    "dtlz2": {
        "piecewise_linear_answers": PIECEWISE_LINEAR_ANSWERS,
        "l1_answers": L1_ANSWERS,
        "beta_products_answers": BETA_HF_ANSWERS,
    },
    "osy": {
        "piecewise_linear_answers": PIECEWISE_LINEAR_OSY_ANSWERS,
        "osy_sigmoid_answers": OSY_ANSWERS,
        "l1_answers": L1_ANSWERS,
    },
    "vehicle_safety": {
        "piecewise_linear_answers": PIECEWISE_LINEAR_ANSWERS,
        "beta_products_answers": BETA_HF_ANSWERS,
    },
    "carcab": {
        "piecewise_linear_answers": PIECEWISE_LINEAR_ANSWERS,
    },
    "thermo": {
        "thermo_A_answers": THERMO_A_ANSWERS,
        "thermo_B_answers": THERMO_B_ANSWERS,
    },
}
