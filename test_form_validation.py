#!/usr/bin/env python3
"""
Test form validation functionality
"""
from app import FormValidator

def test_form_validation():
    """Test the FormValidator class with different field types and answers"""
    
    print("=== Testing Form Validation ===\n")
    
    # Test cases for different field types
    test_cases = [
        # Text field tests
        {"field": {"name": "firstName", "type": "text"}, "answer": "John", "expected_valid": True},
        {"field": {"name": "firstName", "type": "text"}, "answer": "  John  ", "expected_valid": True, "expected_value": "John"},
        {"field": {"name": "firstName", "type": "text"}, "answer": "", "expected_valid": False},
        {"field": {"name": "firstName", "type": "text"}, "answer": "   ", "expected_valid": False},
        
        # Checkbox field tests
        {"field": {"name": "agreeTerms", "type": "checkbox"}, "answer": "yes", "expected_valid": True, "expected_value": "Yes"},
        {"field": {"name": "agreeTerms", "type": "checkbox"}, "answer": "YES", "expected_valid": True, "expected_value": "Yes"},
        {"field": {"name": "agreeTerms", "type": "checkbox"}, "answer": "true", "expected_valid": True, "expected_value": "Yes"},
        {"field": {"name": "agreeTerms", "type": "checkbox"}, "answer": "1", "expected_valid": True, "expected_value": "Yes"},
        {"field": {"name": "agreeTerms", "type": "checkbox"}, "answer": "no", "expected_valid": True, "expected_value": ""},
        {"field": {"name": "agreeTerms", "type": "checkbox"}, "answer": "false", "expected_valid": True, "expected_value": ""},
        {"field": {"name": "agreeTerms", "type": "checkbox"}, "answer": "0", "expected_valid": True, "expected_value": ""},
        {"field": {"name": "agreeTerms", "type": "checkbox"}, "answer": "maybe", "expected_valid": False},
        {"field": {"name": "agreeTerms", "type": "checkbox"}, "answer": "", "expected_valid": False},
        
        # Choice field tests
        {"field": {"name": "country", "type": "choice"}, "answer": "United States", "expected_valid": True},
        {"field": {"name": "country", "type": "choice"}, "answer": "  Canada  ", "expected_valid": True, "expected_value": "Canada"},
        {"field": {"name": "country", "type": "choice"}, "answer": "", "expected_valid": False},
        
        # Unknown field tests
        {"field": {"name": "unknown_field", "type": "unknown"}, "answer": "anything", "expected_valid": True},
        {"field": {"name": "unknown_field", "type": "unknown"}, "answer": "", "expected_valid": True},
    ]
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        field = test_case["field"]
        answer = test_case["answer"]
        expected_valid = test_case["expected_valid"]
        expected_value = test_case.get("expected_value")
        
        print(f"Test {i}: {field['type']} field '{field['name']}' with answer '{answer}'")
        
        result = FormValidator.validate_and_format(field, answer)
        
        # Check validity
        if result["is_valid"] == expected_valid:
            print(f"   ✅ Validity check passed: {result['is_valid']}")
        else:
            print(f"   ❌ Validity check failed: expected {expected_valid}, got {result['is_valid']}")
            failed += 1
            continue
        
        # Check value if expected
        if expected_value is not None:
            if result["value"] == expected_value:
                print(f"   ✅ Value check passed: '{result['value']}'")
            else:
                print(f"   ❌ Value check failed: expected '{expected_value}', got '{result['value']}'")
                failed += 1
                continue
        
        # Show error message if invalid
        if not result["is_valid"]:
            print(f"   💬 Error message: {result['error']}")
        else:
            print(f"   💬 Formatted value: '{result['value']}'")
        
        passed += 1
        print()
    
    print("="*50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All form validation tests passed!")
    else:
        print(f"❌ {failed} tests failed")

if __name__ == "__main__":
    test_form_validation()