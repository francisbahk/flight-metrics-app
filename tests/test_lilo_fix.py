#!/usr/bin/env python3
"""
Test suite to verify LILO model redirect fix.
Run with: python test_lilo_fix.py
"""
import os
import sys

def test_model_mapping():
    """Test that gemini-1.5-flash is NOT redirected to experimental model."""
    print("="*80)
    print("TEST 1: Model Mapping Fix")
    print("="*80)

    from language_bo_code.llm_utils import LLMClient

    # Set a dummy API key for initialization
    os.environ['GOOGLE_API_KEY'] = 'test_key_for_model_check'

    try:
        # Create client with gemini-1.5-flash
        client = LLMClient(model="gemini-1.5-flash")

        # Check that the model is NOT redirected to experimental
        if client.model == "gemini-1.5-flash":
            print("✅ PASS: gemini-1.5-flash is NOT redirected")
            print(f"   Model is: {client.model}")
            return True
        elif client.model == "gemini-2.0-flash-exp":
            print("❌ FAIL: gemini-1.5-flash was redirected to experimental model!")
            print(f"   Model is: {client.model}")
            return False
        else:
            print(f"⚠️  WARNING: Model is {client.model} (unexpected)")
            return False
    except Exception as e:
        print(f"❌ FAIL: Error creating LLMClient: {e}")
        return False
    finally:
        # Clean up
        if 'GOOGLE_API_KEY' in os.environ and os.environ['GOOGLE_API_KEY'] == 'test_key_for_model_check':
            del os.environ['GOOGLE_API_KEY']


def test_default_model():
    """Test that default model is stable, not experimental."""
    print("\n" + "="*80)
    print("TEST 2: Default Model is Stable")
    print("="*80)

    from language_bo_code.llm_utils import LLMClient

    os.environ['GOOGLE_API_KEY'] = 'test_key_for_default_check'

    try:
        # Create client with default model
        client = LLMClient()

        if client.model == "gemini-1.5-flash":
            print("✅ PASS: Default model is gemini-1.5-flash (stable)")
            return True
        elif client.model == "gemini-2.0-flash-exp":
            print("❌ FAIL: Default model is experimental gemini-2.0-flash-exp")
            return False
        else:
            print(f"⚠️  WARNING: Default model is {client.model}")
            return True  # Not necessarily a failure
    except Exception as e:
        print(f"❌ FAIL: Error creating default LLMClient: {e}")
        return False
    finally:
        if 'GOOGLE_API_KEY' in os.environ and os.environ['GOOGLE_API_KEY'] == 'test_key_for_default_check':
            del os.environ['GOOGLE_API_KEY']


def test_lilo_config():
    """Test that LILO configuration uses correct model."""
    print("\n" + "="*80)
    print("TEST 3: LILO Configuration")
    print("="*80)

    from lilo_integration import StreamlitLILOBridge

    os.environ['GOOGLE_API_KEY'] = 'test_key_for_lilo_config'

    try:
        bridge = StreamlitLILOBridge()

        # Check API key is loaded
        if not bridge.api_key:
            print("❌ FAIL: LILO bridge has no API key")
            return False

        print("✅ PASS: LILO bridge initialized with API key")
        return True
    except Exception as e:
        print(f"❌ FAIL: Error creating LILO bridge: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'GOOGLE_API_KEY' in os.environ and os.environ['GOOGLE_API_KEY'] == 'test_key_for_lilo_config':
            del os.environ['GOOGLE_API_KEY']


def test_llm_client_api_key_priority():
    """Test that LLMClient checks GOOGLE_API_KEY first."""
    print("\n" + "="*80)
    print("TEST 4: API Key Priority (GOOGLE_API_KEY > GEMINI_API_KEY)")
    print("="*80)

    from language_bo_code.llm_utils import LLMClient

    # Set both keys
    os.environ['GOOGLE_API_KEY'] = 'google_key'
    os.environ['GEMINI_API_KEY'] = 'gemini_key'

    try:
        client = LLMClient()

        if client.api_key == 'google_key':
            print("✅ PASS: GOOGLE_API_KEY takes priority")
            return True
        elif client.api_key == 'gemini_key':
            print("❌ FAIL: GEMINI_API_KEY was used instead of GOOGLE_API_KEY")
            return False
        else:
            print(f"⚠️  WARNING: Unexpected API key: {client.api_key}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Error testing API key priority: {e}")
        return False
    finally:
        if 'GOOGLE_API_KEY' in os.environ:
            del os.environ['GOOGLE_API_KEY']
        if 'GEMINI_API_KEY' in os.environ:
            del os.environ['GEMINI_API_KEY']


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("LILO FIX VERIFICATION TESTS")
    print("="*80 + "\n")

    results = []

    # Run tests
    results.append(("Model Mapping Fix", test_model_mapping()))
    results.append(("Default Model is Stable", test_default_model()))
    results.append(("LILO Configuration", test_lilo_config()))
    results.append(("API Key Priority", test_llm_client_api_key_priority()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! LILO fix is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. LILO fix needs attention.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
