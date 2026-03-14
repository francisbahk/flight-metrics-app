"""
Pilot token configuration and generation for group-based cross-validation.

Groups:
- Group A (GA01-GA05): Complete search, NO re-ranking required
- Group B (GB01-GB05): Complete search, re-rank 4 Group A prompts each
- Group C (GC01-GC05): Complete search, re-rank 4 Group B prompts each
- Group D (GD01-GD05): Complete search, re-rank 4 Group C prompts each

Assignment algorithm (balanced coverage):
- 5 people x 4 reviews = 20 reviews per group
- 5 prompts need 4 reviews each = 20 reviews needed
- Each person skips exactly 1 prompt from previous group
"""

from typing import List, Optional, Dict

# Token definitions with group and re-rank assignments
PILOT_TOKENS: Dict[str, Dict] = {
    # Group A - No re-ranking (first group in study)
    'GA01': {'group': 'A', 'rerank_targets': []},
    'GA02': {'group': 'A', 'rerank_targets': []},
    'GA03': {'group': 'A', 'rerank_targets': []},
    'GA04': {'group': 'A', 'rerank_targets': []},
    'GA05': {'group': 'A', 'rerank_targets': []},

    # Group B - Re-ranks Group A prompts (4 each, balanced coverage)
    # Each Group A prompt gets reviewed by exactly 4 Group B participants
    'GB01': {'group': 'B', 'rerank_targets': ['GA02', 'GA03', 'GA04', 'GA05']},  # skips GA01
    'GB02': {'group': 'B', 'rerank_targets': ['GA01', 'GA03', 'GA04', 'GA05']},  # skips GA02
    'GB03': {'group': 'B', 'rerank_targets': ['GA01', 'GA02', 'GA04', 'GA05']},  # skips GA03
    'GB04': {'group': 'B', 'rerank_targets': ['GA01', 'GA02', 'GA03', 'GA05']},  # skips GA04
    'GB05': {'group': 'B', 'rerank_targets': ['GA01', 'GA02', 'GA03', 'GA04']},  # skips GA05

    # Group C - Re-ranks Group B prompts (4 each, balanced coverage)
    # Each Group B prompt gets reviewed by exactly 4 Group C participants
    'GC01': {'group': 'C', 'rerank_targets': ['GB02', 'GB03', 'GB04', 'GB05']},  # skips GB01
    'GC02': {'group': 'C', 'rerank_targets': ['GB01', 'GB03', 'GB04', 'GB05']},  # skips GB02
    'GC03': {'group': 'C', 'rerank_targets': ['GB01', 'GB02', 'GB04', 'GB05']},  # skips GB03
    'GC04': {'group': 'C', 'rerank_targets': ['GB01', 'GB02', 'GB03', 'GB05']},  # skips GB04
    'GC05': {'group': 'C', 'rerank_targets': ['GB01', 'GB02', 'GB03', 'GB04']},  # skips GB05

    # Group D - Re-ranks Group C prompts (4 each, balanced coverage)
    # Each Group C prompt gets reviewed by exactly 4 Group D participants
    'GD01': {'group': 'D', 'rerank_targets': ['GC02', 'GC03', 'GC04', 'GC05']},  # skips GC01
    'GD02': {'group': 'D', 'rerank_targets': ['GC01', 'GC03', 'GC04', 'GC05']},  # skips GC02
    'GD03': {'group': 'D', 'rerank_targets': ['GC01', 'GC02', 'GC04', 'GC05']},  # skips GC03
    'GD04': {'group': 'D', 'rerank_targets': ['GC01', 'GC02', 'GC03', 'GC05']},  # skips GC04
    'GD05': {'group': 'D', 'rerank_targets': ['GC01', 'GC02', 'GC03', 'GC04']},  # skips GC05
}


def get_token_group(token: str) -> Optional[str]:
    """
    Get the group letter for a token.

    Args:
        token: Pilot token (e.g., 'GA01', 'GB03')

    Returns:
        Group letter ('A', 'B', 'C', or 'D') or None if not a pilot token
    """
    config = PILOT_TOKENS.get(token.upper())
    return config['group'] if config else None


def get_rerank_targets(token: str) -> List[str]:
    """
    Get the list of tokens this user should re-rank.

    Args:
        token: Pilot token (e.g., 'GB01')

    Returns:
        List of target tokens to re-rank (e.g., ['GA02', 'GA03', 'GA04', 'GA05'])
        Empty list for Group A tokens
    """
    config = PILOT_TOKENS.get(token.upper())
    return config['rerank_targets'] if config else []


def is_pilot_token(token: str) -> bool:
    """
    Check if a token is a pilot study token.

    Args:
        token: Token to check

    Returns:
        True if it's a valid pilot token, False otherwise
    """
    if not token:
        return False
    return token.upper() in PILOT_TOKENS


def get_num_reranks(token: str) -> int:
    """
    Get the number of re-rankings required for a token.

    Args:
        token: Pilot token

    Returns:
        Number of re-rankings (0 for Group A, 4 for Groups B and C)
    """
    return len(get_rerank_targets(token))


def generate_pilot_tokens_md(output_file: str = 'pilot_tokens.md') -> str:
    """
    Generate a markdown file with all pilot tokens and their assignments.

    Args:
        output_file: Output filename (default: pilot_tokens.md)

    Returns:
        Path to generated file
    """
    lines = [
        "# Pilot Study Tokens",
        "",
        "## Overview",
        "- **4 groups** of 5 participants each (20 total tokens)",
        "- **Group A**: Complete flight search and ranking (no re-ranking)",
        "- **Group B**: Complete flight search + re-rank 4 prompts from Group A",
        "- **Group C**: Complete flight search + re-rank 4 prompts from Group B",
        "- **Group D**: Complete flight search + re-rank 4 prompts from Group C",
        "",
        "## Balanced Coverage",
        "Each prompt from the previous group is reviewed by exactly 4 participants.",
        "",
        "---",
        "",
        "## Group A Tokens (No Re-ranking)",
        "",
        "| Token | Re-rank Targets |",
        "|-------|-----------------|",
    ]

    # Group A
    for token in ['GA01', 'GA02', 'GA03', 'GA04', 'GA05']:
        lines.append(f"| `{token}` | None (first group) |")

    lines.extend([
        "",
        "---",
        "",
        "## Group B Tokens (Re-rank Group A)",
        "",
        "| Token | Re-rank Targets | Skips |",
        "|-------|-----------------|-------|",
    ])

    # Group B
    for token in ['GB01', 'GB02', 'GB03', 'GB04', 'GB05']:
        targets = get_rerank_targets(token)
        all_ga = ['GA01', 'GA02', 'GA03', 'GA04', 'GA05']
        skipped = [t for t in all_ga if t not in targets][0]
        lines.append(f"| `{token}` | {', '.join(targets)} | {skipped} |")

    lines.extend([
        "",
        "---",
        "",
        "## Group C Tokens (Re-rank Group B)",
        "",
        "| Token | Re-rank Targets | Skips |",
        "|-------|-----------------|-------|",
    ])

    # Group C
    for token in ['GC01', 'GC02', 'GC03', 'GC04', 'GC05']:
        targets = get_rerank_targets(token)
        all_gb = ['GB01', 'GB02', 'GB03', 'GB04', 'GB05']
        skipped = [t for t in all_gb if t not in targets][0]
        lines.append(f"| `{token}` | {', '.join(targets)} | {skipped} |")

    lines.extend([
        "",
        "---",
        "",
        "## Group D Tokens (Re-rank Group C)",
        "",
        "| Token | Re-rank Targets | Skips |",
        "|-------|-----------------|-------|",
    ])

    # Group D
    for token in ['GD01', 'GD02', 'GD03', 'GD04', 'GD05']:
        targets = get_rerank_targets(token)
        all_gc = ['GC01', 'GC02', 'GC03', 'GC04', 'GC05']
        skipped = [t for t in all_gc if t not in targets][0]
        lines.append(f"| `{token}` | {', '.join(targets)} | {skipped} |")

    lines.extend([
        "",
        "---",
        "",
        "## Coverage Matrix",
        "",
        "### Group A prompts reviewed by Group B:",
        "",
        "| Prompt | Reviewed By |",
        "|--------|-------------|",
    ])

    # Coverage matrix for A -> B
    for ga_token in ['GA01', 'GA02', 'GA03', 'GA04', 'GA05']:
        reviewers = [gb for gb in ['GB01', 'GB02', 'GB03', 'GB04', 'GB05']
                     if ga_token in get_rerank_targets(gb)]
        lines.append(f"| {ga_token} | {', '.join(reviewers)} |")

    lines.extend([
        "",
        "### Group B prompts reviewed by Group C:",
        "",
        "| Prompt | Reviewed By |",
        "|--------|-------------|",
    ])

    # Coverage matrix for B -> C
    for gb_token in ['GB01', 'GB02', 'GB03', 'GB04', 'GB05']:
        reviewers = [gc for gc in ['GC01', 'GC02', 'GC03', 'GC04', 'GC05']
                     if gb_token in get_rerank_targets(gc)]
        lines.append(f"| {gb_token} | {', '.join(reviewers)} |")

    lines.extend([
        "",
        "### Group C prompts reviewed by Group D:",
        "",
        "| Prompt | Reviewed By |",
        "|--------|-------------|",
    ])

    # Coverage matrix for C -> D
    for gc_token in ['GC01', 'GC02', 'GC03', 'GC04', 'GC05']:
        reviewers = [gd for gd in ['GD01', 'GD02', 'GD03', 'GD04', 'GD05']
                     if gc_token in get_rerank_targets(gd)]
        lines.append(f"| {gc_token} | {', '.join(reviewers)} |")

    lines.extend([
        "",
        "---",
        "",
        "## Usage Instructions",
        "",
        "1. **Group A goes first**: Distribute GA01-GA05 tokens to first 5 participants",
        "2. **Wait for Group A to complete**: All 5 must finish before Group B starts",
        "3. **Group B goes second**: Distribute GB01-GB05 tokens to next 5 participants",
        "4. **Wait for Group B to complete**: All 5 must finish before Group C starts",
        "5. **Group C goes third**: Distribute GC01-GC05 tokens to next 5 participants",
        "6. **Wait for Group C to complete**: All 5 must finish before Group D starts",
        "7. **Group D goes last**: Distribute GD01-GD05 tokens to final 5 participants",
        "",
    ])

    content = '\n'.join(lines)

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"Generated {output_file}")
    return output_file


if __name__ == "__main__":
    # Generate the markdown file
    generate_pilot_tokens_md()

    # Print summary
    print("\nPilot Token Summary:")
    print("=" * 50)

    for group in ['A', 'B', 'C', 'D']:
        print(f"\nGroup {group}:")
        for token, config in PILOT_TOKENS.items():
            if config['group'] == group:
                targets = config['rerank_targets']
                if targets:
                    print(f"  {token}: re-ranks {', '.join(targets)}")
                else:
                    print(f"  {token}: no re-ranking")
