#!/usr/bin/env python3
"""
Quick test script to verify survey save works without going through the whole app.
"""

from backend.db import save_survey_response

# Test survey data
test_survey_data = {
    'satisfaction': 5,
    'ease_of_use': 4,
    'encountered_issues': 'No',
    'issues_description': None,
    'search_method': 'AI',
    'understood_ranking': 5,
    'helpful_features': ['AI Search', 'LILO'],
    'flights_matched': 4,
    'confusing_frustrating': None,
    'missing_features': None,
    'would_use_again': 'Yes',
    'would_use_again_reason': 'Very helpful',
    'compared_to_others': 5,
    'additional_comments': 'Great tool!'
}

print("Testing survey save...")
print("-" * 50)

try:
    success = save_survey_response(
        session_id="test_session_123",
        survey_data=test_survey_data,
        completion_token="TEST_TOKEN"
    )

    if success:
        print("✅ SUCCESS! Survey saved to database.")
        print("The survey save functionality is working correctly.")
    else:
        print("❌ FAILED! save_survey_response returned False.")
        print("Check the error message printed above.")

except Exception as e:
    print(f"❌ EXCEPTION: {str(e)}")
    import traceback
    print("\nFull traceback:")
    print(traceback.format_exc())

print("-" * 50)
