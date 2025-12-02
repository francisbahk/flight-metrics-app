"""
Script to generate TEST tokens for testing purposes.
Test tokens start with 'TEST' followed by 5 random characters (9 chars total).
"""
import secrets
import string
from db import SessionLocal, Token


def generate_test_token():
    """
    Generate a test token starting with 'TEST' + 5 random characters.

    Returns:
        A test token string (9 characters total)
    """
    # Use uppercase letters and digits for easy typing
    alphabet = string.ascii_uppercase + string.digits
    random_suffix = ''.join(secrets.choice(alphabet) for _ in range(5))
    return f"TEST{random_suffix}"


def generate_test_tokens(count):
    """
    Generate and save multiple unique test tokens to the database.

    Args:
        count: Number of test tokens to generate

    Returns:
        List of generated test tokens
    """
    db = SessionLocal()
    tokens = []

    try:
        print(f"Generating {count} test tokens (9 characters: TEST + 5 random)...")

        # Get existing tokens to avoid duplicates
        existing_tokens = {t.token for t in db.query(Token).all()}

        while len(tokens) < count:
            token = generate_test_token()

            # Ensure uniqueness
            if token not in existing_tokens and token not in tokens:
                tokens.append(token)

        # Save to database
        for token in tokens:
            token_record = Token(token=token)
            db.add(token_record)

        db.commit()
        print(f"\n✓ Generated and saved {count} test tokens!")
        print("\nTest Tokens:")
        print("-" * 60)
        for i, token in enumerate(tokens, 1):
            url = f"https://listen-cornell3.streamlit.app/?id={token}"
            print(f"{i:3d}. {token:12s} -> {url}")
        print("-" * 60)

        # Also save to a file
        filename = f"test_tokens_{count}.txt"
        with open(filename, 'w') as f:
            f.write("TEST TOKENS FOR FLIGHT RANKING STUDY\n")
            f.write("=" * 80 + "\n\n")
            f.write("⚠️  These are TEST tokens for internal testing only.\n")
            f.write("   All tokens starting with 'TEST' are for development/testing.\n\n")
            for i, token in enumerate(tokens, 1):
                url = f"https://listen-cornell3.streamlit.app/?id={token}"
                f.write(f"{i}. Token: {token}\n")
                f.write(f"   URL: {url}\n\n")

        print(f"\n✓ Test tokens also saved to: {filename}")

        return tokens

    except Exception as e:
        db.rollback()
        print(f"✗ Error generating test tokens: {str(e)}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    import sys

    # Default: generate 5 test tokens
    count = 5

    if len(sys.argv) > 1:
        count = int(sys.argv[1])

    print("=" * 80)
    print("TEST TOKEN GENERATOR FOR FLIGHT RANKING STUDY")
    print("=" * 80)
    print(f"\nGenerating {count} test tokens (9 characters: TEST + 5 random)")
    print(f"⚠️  Test tokens are for internal testing only\n")

    tokens = generate_test_tokens(count)

    print(f"\n✓ Done! Use these tokens for testing.")
    print(f"✓ Test tokens work the same as regular tokens but are easily identifiable.")
