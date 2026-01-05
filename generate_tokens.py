#!/usr/bin/env python3
"""
Generate access tokens for the flight app study.
Usage: python generate_tokens.py <number_of_tokens>
"""

import sys
import secrets
from backend.db import SessionLocal, AccessToken
from datetime import datetime


def generate_access_tokens(count: int = 100) -> list:
    """
    Generate N unique access tokens and save them to the database.

    Args:
        count: Number of tokens to generate (default 100)

    Returns:
        List of generated tokens
    """
    db = SessionLocal()
    tokens = []

    try:
        print(f"Generating {count} access tokens...")

        for i in range(count):
            # Generate a unique 8-character token
            while True:
                token = secrets.token_urlsafe(6).upper()  # ~8 chars

                # Check if token already exists
                existing = db.query(AccessToken).filter(AccessToken.token == token).first()
                if not existing:
                    break

            # Create token record
            access_token = AccessToken(
                token=token,
                created_at=datetime.utcnow(),
                is_used=0,
                used_at=None,
                completion_token=None
            )

            db.add(access_token)
            tokens.append(token)

            # Commit every 20 tokens to avoid large transactions
            if (i + 1) % 20 == 0:
                db.commit()
                print(f"  Generated {i + 1}/{count} tokens...")

        # Commit any remaining tokens
        db.commit()
        print(f"\n✅ Successfully generated {len(tokens)} tokens!")

        return tokens

    except Exception as e:
        db.rollback()
        print(f"\n❌ Error generating tokens: {e}")
        import traceback
        traceback.print_exc()
        return []

    finally:
        db.close()


def print_tokens_as_urls(tokens: list, base_url: str = "https://flight-metrics-app.streamlit.app"):
    """
    Print tokens as full URLs that participants can use.

    Args:
        tokens: List of tokens
        base_url: Base URL of the app
    """
    print("\n" + "="*80)
    print("PARTICIPANT ACCESS URLS")
    print("="*80 + "\n")

    for i, token in enumerate(tokens, 1):
        url = f"{base_url}/?token={token}"
        print(f"{i:3d}. {url}")

    print("\n" + "="*80)
    print(f"Total URLs: {len(tokens)}")
    print("="*80)


def save_tokens_to_file(tokens: list, filename: str = "access_tokens.txt"):
    """
    Save tokens to a text file.

    Args:
        tokens: List of tokens
        filename: Output filename
    """
    base_url = "https://flight-metrics-app.streamlit.app"

    with open(filename, 'w') as f:
        f.write("Flight App - Participant Access URLs\n")
        f.write("="*80 + "\n\n")

        for i, token in enumerate(tokens, 1):
            url = f"{base_url}/?token={token}"
            f.write(f"{i:3d}. {url}\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"Total URLs: {len(tokens)}\n")
        f.write(f"Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

    print(f"\n💾 Tokens saved to: {filename}")


if __name__ == "__main__":
    # Get number of tokens from command line or default to 100
    count = 100
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}")
            print("Usage: python generate_tokens.py <number_of_tokens>")
            sys.exit(1)

    # Generate tokens
    tokens = generate_access_tokens(count)

    if tokens:
        # Print as URLs
        print_tokens_as_urls(tokens)

        # Save to file
        save_tokens_to_file(tokens)

        print(f"\n✅ Done! Generated {len(tokens)} participant access URLs.")
    else:
        print("\n❌ Failed to generate tokens.")
        sys.exit(1)
