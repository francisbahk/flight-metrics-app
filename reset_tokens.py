#!/usr/bin/env python3
"""
Delete all existing access tokens and generate new ones.
Usage: python reset_tokens.py <number_of_tokens>
"""

import sys
from backend.db import SessionLocal, AccessToken
from generate_tokens import generate_access_tokens, print_tokens_as_urls, save_tokens_to_file


def delete_all_tokens():
    """Delete all access tokens from the database."""
    db = SessionLocal()

    try:
        # Count existing tokens
        count = db.query(AccessToken).count()

        if count == 0:
            print("No existing tokens to delete.")
            return True

        print(f"Found {count} existing tokens in database.")
        print("Deleting all tokens...")

        # Delete all tokens
        db.query(AccessToken).delete()
        db.commit()

        print(f"✅ Successfully deleted {count} tokens!")
        return True

    except Exception as e:
        db.rollback()
        print(f"❌ Error deleting tokens: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        db.close()


if __name__ == "__main__":
    # Get number of tokens from command line or default to 100
    count = 100
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}")
            print("Usage: python reset_tokens.py <number_of_tokens>")
            sys.exit(1)

    print("="*80)
    print("TOKEN RESET SCRIPT")
    print("="*80 + "\n")

    # Step 1: Delete all existing tokens
    if not delete_all_tokens():
        print("\n❌ Failed to delete existing tokens. Aborting.")
        sys.exit(1)

    print()

    # Step 2: Generate new tokens
    tokens = generate_access_tokens(count)

    if tokens:
        # Print as URLs
        print_tokens_as_urls(tokens)

        # Save to file
        save_tokens_to_file(tokens)

        print(f"\n✅ Done! Deleted all old tokens and generated {len(tokens)} new access URLs.")
    else:
        print("\n❌ Failed to generate new tokens.")
        sys.exit(1)
