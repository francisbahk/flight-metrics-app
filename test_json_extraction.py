#!/usr/bin/env python3
"""
Test enhanced JSON extraction to handle various Gemini response formats.
"""
from language_bo_code.utils import extract_json_from_text


def test_json_with_code_fence():
    """Test extraction from standard ```json code block."""
    text = """Here's the response:
```json
{
    "q1": "What is your budget?",
    "q2": "How important is travel time?"
}
```
"""
    result = extract_json_from_text(text)
    assert result is not None, "Failed to extract JSON from code fence"
    assert "q1" in result, "Missing q1 key"
    assert "q2" in result, "Missing q2 key"
    print("✅ PASS: Standard ```json code block")


def test_json_without_language_tag():
    """Test extraction from generic ``` code block."""
    text = """Here's the response:
```
{
    "q1": "What is your budget?",
    "q2": "How important is travel time?"
}
```
"""
    result = extract_json_from_text(text)
    assert result is not None, "Failed to extract JSON from generic code fence"
    assert "q1" in result, "Missing q1 key"
    assert "q2" in result, "Missing q2 key"
    print("✅ PASS: Generic ``` code block")


def test_json_without_code_fence():
    """Test extraction from raw JSON in text."""
    text = """Here are the questions you requested:

{
    "q1": "What is your budget?",
    "q2": "How important is travel time?"
}

Let me know if you need more questions!"""
    result = extract_json_from_text(text)
    assert result is not None, "Failed to extract raw JSON from text"
    assert "q1" in result, "Missing q1 key"
    assert "q2" in result, "Missing q2 key"
    print("✅ PASS: Raw JSON without code fence")


def test_malformed_json():
    """Test that malformed JSON returns None."""
    text = """Here's a broken response:
```json
{
    "q1": "What is your budget?",
    "q2": "How important is travel time?"
    # Missing closing brace
```
"""
    result = extract_json_from_text(text)
    assert result is None, "Should return None for malformed JSON"
    print("✅ PASS: Malformed JSON returns None")


def test_no_json():
    """Test that text without JSON returns None."""
    text = """This is just regular text without any JSON content."""
    result = extract_json_from_text(text)
    assert result is None, "Should return None when no JSON found"
    print("✅ PASS: No JSON returns None")


def run_all_tests():
    """Run all JSON extraction tests."""
    print("="*80)
    print("JSON EXTRACTION TESTS")
    print("="*80)

    tests = [
        test_json_with_code_fence,
        test_json_without_language_tag,
        test_json_without_code_fence,
        test_malformed_json,
        test_no_json,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {test_func.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{len(tests)} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
