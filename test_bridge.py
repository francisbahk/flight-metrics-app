import os
import pandas as pd
os.environ['GEMINI_API_KEY'] = 'AIzaSyA8gHulJVvnzEJgJGmME1X0iBuNspDbLOs'

from lilo_integration import StreamlitLILOBridge

print("Testing StreamlitLILOBridge...")

# Create bridge
bridge = StreamlitLILOBridge()
print("✓ Bridge created")

# Create a session
session = bridge.create_session(
    session_id="test_123",
    flights_df=pd.DataFrame()
)
print(f"✓ Session created: {session.session_id}")

# Get initial questions
questions = bridge.get_initial_questions("test_123")
print(f"✓ Got {len(questions)} initial questions:")
for i, q in enumerate(questions, 1):
    print(f"  {i}. {q[:80]}...")

# Simulate user answers
answers = [
    "I prefer morning flights because I like to arrive early",
    "Price is very important, I'm willing to take connections to save money"
][:len(questions)]

result = bridge.submit_user_answers("test_123", answers)
print(f"✓ Submitted answers: {result}")

# Get status
status = bridge.get_session_status("test_123")
print(f"✓ Session status: {status}")

print("\n✅ Bridge test complete!")
