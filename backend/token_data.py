"""
Script to generate a master CSV report of all tokens with their status.
Includes color-coded terminal output and CSV export.
"""
from db import SessionLocal, Token, Search
import csv
from datetime import datetime


# ANSI color codes for terminal output
class Colors:
    # Mild colors for readability
    AMBER = '\033[38;5;214m'      # Mild amber for test tokens
    BLUE = '\033[38;5;117m'       # Mild blue for real tokens
    GREEN = '\033[38;5;120m'      # Mild green for unused
    RED = '\033[38;5;210m'        # Mild red for used
    RESET = '\033[0m'
    BOLD = '\033[1m'


def get_token_data():
    """
    Fetch all token data from database with usage information.

    Returns:
        List of dicts containing token information
    """
    db = SessionLocal()
    token_data = []

    try:
        # Get all tokens
        all_tokens = db.query(Token).order_by(Token.created_at.desc()).all()

        for token_record in all_tokens:
            # Determine if test token or real token
            is_test = token_record.token.startswith('TEST')
            token_type = 'TEST' if is_test else 'REAL'

            # Get search_id if token was used
            search_id = None
            if token_record.is_used:
                # Find the search that used this token
                search = db.query(Search).filter(Search.token == token_record.token).first()
                if search:
                    search_id = search.search_id

            token_info = {
                'token': token_record.token,
                'type': token_type,
                'is_test': is_test,
                'created_at': token_record.created_at,
                'is_used': bool(token_record.is_used),
                'used_at': token_record.used_at,
                'search_id': search_id
            }
            token_data.append(token_info)

        return token_data

    finally:
        db.close()


def print_colored_table(token_data):
    """
    Print a colored table of token data to the terminal.

    Args:
        token_data: List of token information dicts
    """
    print("\n" + "=" * 120)
    print(f"{Colors.BOLD}TOKEN STATUS REPORT{Colors.RESET}")
    print("=" * 120)

    # Header
    print(f"\n{Colors.BOLD}{'Token':<15} {'Type':<8} {'Created':<20} {'Status':<12} {'Used At':<20} {'Search ID':<10}{Colors.RESET}")
    print("-" * 120)

    # Data rows
    for token_info in token_data:
        # Color for token type (test=amber, real=blue)
        type_color = Colors.AMBER if token_info['is_test'] else Colors.BLUE
        token_display = f"{type_color}{token_info['token']:<15}{Colors.RESET}"
        type_display = f"{type_color}{token_info['type']:<8}{Colors.RESET}"

        # Color for status (used=red, unused=green)
        status_color = Colors.RED if token_info['is_used'] else Colors.GREEN
        status_text = 'USED' if token_info['is_used'] else 'AVAILABLE'
        status_display = f"{status_color}{status_text:<12}{Colors.RESET}"

        # Format dates
        created_str = token_info['created_at'].strftime('%Y-%m-%d %H:%M:%S') if token_info['created_at'] else 'N/A'
        used_str = token_info['used_at'].strftime('%Y-%m-%d %H:%M:%S') if token_info['used_at'] else '-'
        search_str = str(token_info['search_id']) if token_info['search_id'] else '-'

        print(f"{token_display} {type_display} {created_str:<20} {status_display} {used_str:<20} {search_str:<10}")

    print("-" * 120)
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    total = len(token_data)
    test_tokens = sum(1 for t in token_data if t['is_test'])
    real_tokens = total - test_tokens
    used = sum(1 for t in token_data if t['is_used'])
    available = total - used

    print(f"  Total Tokens: {total}")
    print(f"  {Colors.BLUE}Real Tokens: {real_tokens}{Colors.RESET} | {Colors.AMBER}Test Tokens: {test_tokens}{Colors.RESET}")
    print(f"  {Colors.RED}Used: {used}{Colors.RESET} | {Colors.GREEN}Available: {available}{Colors.RESET}")
    print()


def export_to_csv(token_data, filename=None):
    """
    Export token data to a CSV file.

    Args:
        token_data: List of token information dicts
        filename: Output filename (defaults to token_report_YYYYMMDD_HHMMSS.csv)

    Returns:
        Path to the created CSV file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"token_report_{timestamp}.csv"

    # Define CSV columns
    fieldnames = [
        'Token',
        'Type',
        'Created At',
        'Status',
        'Used At',
        'Search ID',
        'Color_Token_Type',    # For Excel conditional formatting
        'Color_Status'         # For Excel conditional formatting
    ]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for token_info in token_data:
            # Prepare row data
            row = {
                'Token': token_info['token'],
                'Type': token_info['type'],
                'Created At': token_info['created_at'].strftime('%Y-%m-%d %H:%M:%S') if token_info['created_at'] else '',
                'Status': 'USED' if token_info['is_used'] else 'AVAILABLE',
                'Used At': token_info['used_at'].strftime('%Y-%m-%d %H:%M:%S') if token_info['used_at'] else '',
                'Search ID': token_info['search_id'] if token_info['search_id'] else '',
                'Color_Token_Type': 'AMBER' if token_info['is_test'] else 'BLUE',
                'Color_Status': 'RED' if token_info['is_used'] else 'GREEN'
            }
            writer.writerow(row)

    return filename


if __name__ == "__main__":
    print("=" * 120)
    print("FETCHING TOKEN DATA FROM DATABASE...")
    print("=" * 120)

    # Get token data
    token_data = get_token_data()

    if not token_data:
        print("\n⚠️  No tokens found in database.")
        print("   Generate tokens using:")
        print("   - python3 backend/generate_tokens.py <count>        (for real tokens)")
        print("   - python3 backend/generate_test_tokens.py <count>   (for test tokens)")
    else:
        # Print colored table to terminal
        print_colored_table(token_data)

        # Export to CSV
        csv_filename = export_to_csv(token_data)
        print(f"✓ Token data exported to: {csv_filename}")
        print(f"\n{Colors.BOLD}Color Coding Reference:{Colors.RESET}")
        print(f"  Token Type:  {Colors.AMBER}■{Colors.RESET} Test tokens (TEST*)  |  {Colors.BLUE}■{Colors.RESET} Real tokens")
        print(f"  Status:      {Colors.GREEN}■{Colors.RESET} Available          |  {Colors.RED}■{Colors.RESET} Used")
        print()
