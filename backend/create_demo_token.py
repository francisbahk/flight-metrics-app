"""
Script to create a special DEMO token that never expires.
This token can be used unlimited times for testing/demo purposes.
"""
from db import SessionLocal, Token


def create_demo_token():
    """
    Create or update the special DEMO token.
    This token will never be marked as used, allowing unlimited submissions.
    """
    db = SessionLocal()

    try:
        demo_token = "DEMO"

        # Check if DEMO token already exists
        existing = db.query(Token).filter(Token.token == demo_token).first()

        if existing:
            print(f"⚠️  DEMO token already exists")
            print(f"   Created: {existing.created_at}")
            print(f"   Status: {'USED' if existing.is_used else 'AVAILABLE'}")

            # Reset it to available if it was marked as used
            if existing.is_used:
                existing.is_used = 0
                existing.used_at = None
                db.commit()
                print(f"   ✓ Reset to AVAILABLE")
        else:
            # Create new DEMO token
            token_record = Token(token=demo_token)
            db.add(token_record)
            db.commit()
            print(f"✓ Created new DEMO token")

        print("\n" + "=" * 80)
        print("DEMO TOKEN CREATED")
        print("=" * 80)
        print(f"\nToken: {demo_token}")
        print(f"URL: https://listen-cornell3.streamlit.app/?id={demo_token}")
        print("\n⚠️  SPECIAL PROPERTIES:")
        print("   - Never expires")
        print("   - Can be used unlimited times")
        print("   - Perfect for demos and testing")
        print("   - All submissions will be saved to database")
        print("\n💡 Share this URL with anyone who wants to try the website!")
        print("=" * 80)

    except Exception as e:
        db.rollback()
        print(f"✗ Error creating demo token: {str(e)}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    create_demo_token()
