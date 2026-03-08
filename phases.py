"""
Phase-based study configuration.

Replaces the old per-token pilot system with shared phase URLs:
  ?id=PHASEONE  — participants submit their own prompt + rankings, no re-ranking
  ?id=PHASETWO  — participants submit their own prompt + rankings + 1 cross-validation
                  (re-ranks one dynamically selected Phase 1 submission)

How tokens work in the phase system:
  - URL param (?id=PHASEONE) identifies the phase.
  - Participant enters their Prolific ID on a gate screen before seeing the app.
  - Effective DB token = "{PHASE}_{prolific_id}" (e.g. "PHASEONE_abc123xyz").
  - Token marking-as-used is skipped for phase participants (Prolific handles identity).
"""

from typing import Optional

PHASES = {
    'PHASEONE': {
        'label': 'Phase 1',
        'group': 'A',       # No re-ranking (same as old Group A)
        'num_reranks': 0,
    },
    'PHASETWO': {
        'label': 'Phase 2',
        'group': 'B',       # One dynamic cross-validation re-rank
        'num_reranks': 1,
    },
}


def is_phase_url(token: str) -> bool:
    """Return True if the URL ?id= value is a known phase identifier."""
    if not token:
        return False
    return token.upper() in PHASES


def get_phase_group(token: str) -> Optional[str]:
    """Return the group letter ('A', 'B', …) for the given phase URL token."""
    config = PHASES.get(token.upper())
    return config['group'] if config else None


def get_phase_num_reranks(token: str) -> int:
    """Return how many cross-validation re-rankings this phase requires."""
    config = PHASES.get(token.upper())
    return config['num_reranks'] if config else 0


def get_phase_label(token: str) -> str:
    """Return a human-readable label for the phase."""
    config = PHASES.get(token.upper())
    return config['label'] if config else token


def make_effective_token(phase_url: str, prolific_id: str) -> str:
    """
    Build the token stored in the DB for this participant.
    E.g. make_effective_token('PHASEONE', 'abc123xyz') -> 'PHASEONE_abc123xyz'
    """
    return f"{phase_url.upper()}_{prolific_id.strip()}"


def is_phase_token(token: str) -> bool:
    """
    Return True if an effective DB token was created by the phase system
    (i.e. starts with a known phase prefix followed by '_').
    """
    if not token:
        return False
    for phase in PHASES:
        if token.upper().startswith(f"{phase}_"):
            return True
    return False
