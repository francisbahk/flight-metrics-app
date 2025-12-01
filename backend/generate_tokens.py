"""
Script to generate unique participant tokens for the study.
"""
import secrets
import string
from db import SessionLocal, Token

def generate_token(length=8):
    """
    Generate a random token using letters and numbers.

    Args:
        length: Length of the token (default: 8 characters)

    Returns:
        A random token string
    """
    # Use uppercase letters and digits for easy typing
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_tokens(count, length=8):
    """
    Generate and save multiple unique tokens to the database.

    Args:
        count: Number of tokens to generate
        length: Length of each token (default: 8)

    Returns:
        List of generated tokens
    """
    db = SessionLocal()
    tokens = []

    try:
        print(f"Generating {count} tokens of length {length}...")

        # Get existing tokens to avoid duplicates
        existing_tokens = {t.token for t in db.query(Token).all()}

        while len(tokens) < count:
            token = generate_token(length)

            # Ensure uniqueness
            if token not in existing_tokens and token not in tokens:
                tokens.append(token)

        # Save to database
        for token in tokens:
            token_record = Token(token=token)
            db.add(token_record)

        db.commit()
        print(f"\n✓ Generated and saved {count} tokens!")
        print("\nTokens:")
        print("-" * 50)
        for i, token in enumerate(tokens, 1):
            url = f"https://listen-cornell3.streamlit.app/?id={token}"
            print(f"{i:3d}. {token:12s} -> {url}")
        print("-" * 50)

        # Also save to a file
        filename = f"tokens_{count}.txt"
        with open(filename, 'w') as f:
            f.write("PARTICIPANT TOKENS\n")
            f.write("=" * 80 + "\n\n")
            for i, token in enumerate(tokens, 1):
                url = f"https://listen-cornell3.streamlit.app/?id={token}"
                f.write(f"{i}. Token: {token}\n")
                f.write(f"   URL: {url}\n\n")

        print(f"\n✓ Tokens also saved to: {filename}")

        return tokens

    except Exception as e:
        db.rollback()
        print(f"✗ Error generating tokens: {str(e)}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    import sys

    # Default: generate 10 tokens
    count = 10
    length = 8

    if len(sys.argv) > 1:
        count = int(sys.argv[1])

    if len(sys.argv) > 2:
        length = int(sys.argv[2])

    print("=" * 80)
    print("TOKEN GENERATOR FOR FLIGHT RANKING STUDY")
    print("=" * 80)
    print(f"\nGenerating {count} tokens (length: {length} characters)")
    print(f"Each token will create a unique study link\n")

    tokens = generate_tokens(count, length)

    print(f"\n✓ Done! Share one unique URL with each participant.")
    print(f"✓ Each token can only be used once.")
