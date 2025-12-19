"""
Generate access tokens for study participants.
These tokens control study entry but are NOT linked to research data.
"""
import sys
import secrets
import string
from datetime import datetime
from db import SessionLocal, AccessToken


def generate_random_token(length=8):
    """Generate a random alphanumeric token."""
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_access_tokens(count=10):
    """
    Generate access tokens for study participants.

    Args:
        count: Number of tokens to generate

    Returns:
        List of generated token strings
    """
    db = SessionLocal()
    tokens = []

    try:
        print(f"\n🔑 Generating {count} access tokens...")

        for i in range(count):
            # Generate unique token
            while True:
                token_str = generate_random_token(8)
                # Check if token already exists
                existing = db.query(AccessToken).filter(AccessToken.token == token_str).first()
                if not existing:
                    break

            # Create token record
            token_record = AccessToken(
                token=token_str,
                created_at=datetime.now(),
                is_used=0
            )
            db.add(token_record)
            tokens.append(token_str)

            # Print progress
            if (i + 1) % 10 == 0 or (i + 1) == count:
                print(f"  Generated {i + 1}/{count} tokens...")

        # Commit all tokens
        db.commit()
        print(f"\n✅ Successfully generated {count} access tokens!")

        # Print tokens
        print("\n" + "=" * 60)
        print("ACCESS TOKENS (for study entry)")
        print("=" * 60)
        for i, token in enumerate(tokens, 1):
            print(f"{i:3d}. {token}")
        print("=" * 60)

        # Save to file
        filename = f"access_tokens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write("ACCESS TOKENS FOR STUDY ENTRY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            for i, token in enumerate(tokens, 1):
                f.write(f"{i}. {token}\n")

        print(f"\n💾 Tokens saved to: {filename}")

        return tokens

    except Exception as e:
        db.rollback()
        print(f"\n❌ Error generating tokens: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # Get count from command line or use default
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if count < 1 or count > 1000:
        print("❌ Error: Please specify a count between 1 and 1000")
        sys.exit(1)

    generate_access_tokens(count)
