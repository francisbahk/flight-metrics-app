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


def export_to_html(token_data, filename=None):
    """
    Export token data to a color-coded HTML file.

    Args:
        token_data: List of token information dicts
        filename: Output filename (defaults to token_report_YYYYMMDD_HHMMSS.html)

    Returns:
        Path to the created HTML file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"token_report_{timestamp}.html"

    # Define color scheme (mild colors)
    html_colors = {
        'amber': '#FFB84D',      # Mild amber for test tokens
        'blue': '#87CEEB',       # Mild blue for real tokens
        'green': '#90EE90',      # Mild green for available
        'red': '#FFB6B9'         # Mild red for used
    }

    # Calculate summary stats
    total = len(token_data)
    test_tokens = sum(1 for t in token_data if t['is_test'])
    real_tokens = total - test_tokens
    used = sum(1 for t in token_data if t['is_used'])
    available = total - used

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Token Status Report</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            border-bottom: 2px solid #4a4a4a;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: #2d2d2d;
        }}
        th {{
            background-color: #3a3a3a;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #4a4a4a;
            font-weight: bold;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #3a3a3a;
        }}
        tr:hover {{
            background-color: #353535;
        }}
        .summary {{
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }}
        .summary h3 {{
            margin-top: 0;
        }}
        .token-test {{ color: {html_colors['amber']}; font-weight: bold; }}
        .token-real {{ color: {html_colors['blue']}; font-weight: bold; }}
        .status-used {{ color: {html_colors['red']}; font-weight: bold; }}
        .status-available {{ color: {html_colors['green']}; font-weight: bold; }}
        .legend {{
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        .color-box {{
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 5px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>TOKEN STATUS REPORT</h1>

        <div class="legend">
            <h3>Color Legend:</h3>
            <div class="legend-item">
                <span class="color-box" style="background-color: {html_colors['amber']}"></span>
                <span>Test Tokens (TEST*)</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: {html_colors['blue']}"></span>
                <span>Real Tokens</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: {html_colors['green']}"></span>
                <span>Available</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: {html_colors['red']}"></span>
                <span>Used</span>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Token</th>
                    <th>Type</th>
                    <th>Created At</th>
                    <th>Status</th>
                    <th>Used At</th>
                    <th>Search ID</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add data rows
    for token_info in token_data:
        token_class = 'token-test' if token_info['is_test'] else 'token-real'
        status_class = 'status-used' if token_info['is_used'] else 'status-available'

        created_str = token_info['created_at'].strftime('%Y-%m-%d %H:%M:%S') if token_info['created_at'] else 'N/A'
        used_str = token_info['used_at'].strftime('%Y-%m-%d %H:%M:%S') if token_info['used_at'] else '-'
        search_str = str(token_info['search_id']) if token_info['search_id'] else '-'
        status_text = 'USED' if token_info['is_used'] else 'AVAILABLE'

        html += f"""                <tr>
                    <td class="{token_class}">{token_info['token']}</td>
                    <td class="{token_class}">{token_info['type']}</td>
                    <td>{created_str}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{used_str}</td>
                    <td>{search_str}</td>
                </tr>
"""

    # Add summary
    html += f"""            </tbody>
        </table>

        <div class="summary">
            <h3>Summary</h3>
            <p><strong>Total Tokens:</strong> {total}</p>
            <p>
                <span class="token-real"><strong>Real Tokens:</strong> {real_tokens}</span> |
                <span class="token-test"><strong>Test Tokens:</strong> {test_tokens}</span>
            </p>
            <p>
                <span class="status-used"><strong>Used:</strong> {used}</span> |
                <span class="status-available"><strong>Available:</strong> {available}</span>
            </p>
        </div>

        <p style="text-align: center; color: #888; margin-top: 30px;">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
</body>
</html>
"""

    with open(filename, 'w') as f:
        f.write(html)

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

        # Export to HTML only
        html_filename = export_to_html(token_data)

        print(f"✓ Token data exported to: {html_filename} (color-coded HTML report)")
        print(f"\n{Colors.BOLD}Color Coding Reference:{Colors.RESET}")
        print(f"  Token Type:  {Colors.AMBER}■{Colors.RESET} Test tokens (TEST*)  |  {Colors.BLUE}■{Colors.RESET} Real tokens")
        print(f"  Status:      {Colors.GREEN}■{Colors.RESET} Available          |  {Colors.RED}■{Colors.RESET} Used")
        print()
