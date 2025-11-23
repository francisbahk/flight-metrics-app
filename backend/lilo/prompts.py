"""
LILO Prompts - Exact prompts from the LILO paper (arxiv.org/pdf/2510.17671)
"""

def get_question_generation_prompt(experiment_data: str, human_feedback: str,
                                   selected_outcome_indices: str, n_questions: int = 3) -> str:
    """
    Prompt 2: Question generation after user ranks flights.
    Asks LLM to generate questions to better understand user's preferences.

    Args:
        experiment_data: String describing all flights shown (formatted)
        human_feedback: Previous feedback from user (empty in round 1)
        selected_outcome_indices: Indices of flights user ranked
        n_questions: Number of questions to generate (default 3)

    Returns:
        Prompt string for Gemini
    """
    y_names = "flight options"  # Generic name for our domain

    return f"""You are an expert in determining whether a human decision maker (DM)
is going to be satisfied with a set of experimental outcomes y = {{{y_names}}}.

## Experimental outcomes:
So far, we have obtained the following experimental outcomes:
{experiment_data}

## Human feedback messages:
We have also received the following messages from the DM:
{human_feedback}

## Your task:
Given the above your task is to predict pairwise preferences between
experimental outcomes.

In order to better understand the decision maker's utility function you want to ask them about their optimization goals or for feedback regarding specific experimental outcomes.

Here are some points it may be useful to ask the decision maker about {selected_outcome_indices}.

First, analyse the decision maker's goals and feedback messages to understand their overall preferences.

Then, provide a list of questions you would ask the decision maker to better understand their internal utility model.

Your questions can be either general or referring to specific outcomes. For instance, you may ask the decision maker:
- questions clarifying the optimization objective,
- to rank two (or more) outcomes,
- how to improve certain outcomes,
- for a likert-scale rating regarding a specific outcome,
- etc.

When referring to specific outcomes, always state the arm_index involved.

Your questions should help you predict pairwise preferences between any two experimental outcomes from the set of experimental outcomes provided above.

Return your final answer as a json file with the following format containing exactly {n_questions} most important questions:
'''json
{{
"q1" : <question1>,
...
"q{n_questions}" : <question{n_questions}>
}}
'''"""


def get_utility_estimation_prompt(experiment_data: str, human_feedback: str,
                                  human_feedback_summary: str = "") -> str:
    """
    Prompt 4: Scalar utility estimation.
    Predicts probability that DM will be satisfied with each flight.

    Args:
        experiment_data: String describing all flights
        human_feedback: All feedback messages from user
        human_feedback_summary: Optional summary from Prompt 5

    Returns:
        Prompt string for Gemini
    """
    y_names = "flight options"

    return f"""You are an expert in determining whether a human decision maker (DM)
is going to be satisfied with a set of experimental outcomes y = {{{y_names}}}.

## Experimental outcomes:
So far, we have obtained the following experimental outcomes:
{experiment_data}

## Human feedback messages:
We have also received the following messages from the DM:
{human_feedback}
{human_feedback_summary}

## Your task:
Given the above your task is to predict the probability of the
decision maker being satisfied with the experimental outcomes.

First, analyse the human feedback messages to understand the DM's preferences.

Then, provide your predictions for all y's in the set of all
experimental outcomes above.

Return your final answer as a jsonl file with the following format:
'''jsonl
{{
"arm_index": "{{idx0}}",
"reasoning": <reasoning>,
"p_accept": <probability>
}}
{{
"arm_index": "{{idx1}}",
"reasoning": <reasoning>,
"p_accept": <probability>
}}
...
{{
"arm_index": "{{idxn}}",
"reasoning": <reasoning>,
"p_accept": <probability>
}}
'''

Where <reasoning> should be a short reasoning for your prediction and
<probability> should be your best estimate for the probability between
0 and 1 that the DM will be satisfied with the corresponding outcome.

Provide your predictions for ALL y's in the set of experimental outcomes above. That is, for EACH outcome from {{idx0}} to {{idxn}}.

Do not generate any Python code. Just return your predictions as plain text."""


def get_feedback_summarization_prompt(experiment_data: str, human_feedback: str) -> str:
    """
    Prompt 5: Human feedback summarization.
    Summarizes all user feedback into clear optimization goals.

    Args:
        experiment_data: String describing all flights
        human_feedback: All feedback messages from user

    Returns:
        Prompt string for Gemini
    """
    y_names = "flight options"

    return f"""You are an expert in determining whether a human decision maker (DM)
is going to be satisfied with a set of experimental outcomes y = {{{y_names}}}.

## Experimental outcomes:
So far, we have obtained the following experimental outcomes:
{experiment_data}

## Human feedback messages:
We have also received the following messages from the DM:
{human_feedback}

## Your task:
Given the above your task is to summarize the human feedback messages
into a clear description of the DM's optimization goals.

Make your summary as quantitative as possible so that it can be easily
used for utility estimation.

After analysis the human feedback messages, return your final answer
as a json file with the following format:
'''json
{{
"summary": <summary>
}}
'''

Remember about the '''json header!"""