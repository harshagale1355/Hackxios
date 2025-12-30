#!/usr/bin/env python3
"""
Debug script for 422 Unprocessable Entity errors on /next-question
"""
import requests
import json

def debug_422_error():
    """Debug the 422 error on /next-question endpoint"""
    
    base_url = "http://localhost:8004"
    
    print("=== Debugging 422 Unprocessable Entity Error ===\n")
    
    # Test different request formats to identify the issue
    test_cases = [
        {
            "name": "Valid JSON with both fields",
            "data": {"session_id": "test-session", "answer": "test answer"},
            "headers": {"Content-Type": "application/json"}
        },
        {
            "name": "Valid JSON with only session_id",
            "data": {"session_id": "test-session"},
            "headers": {"Content-Type": "application/json"}
        },
        {
            "name": "Missing session_id",
            "data": {"answer": "test answer"},
            "headers": {"Content-Type": "application/json"}
        },
        {
            "name": "Empty JSON object",
            "data": {},
            "headers": {"Content-Type": "application/json"}
        },
        {
            "name": "Invalid JSON format",
            "data": '{"session_id": "test", "answer":}',  # Invalid JSON
            "headers": {"Content-Type": "application/json"},
            "raw": True
        },
        {
            "name": "Wrong Content-Type",
            "data": {"session_id": "test-session", "answer": "test"},
            "headers": {"Content-Type": "text/plain"}
        }
    ]
    
    print("Testing different request formats:\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
        
        try:
            if test_case.get('raw'):
                # Send raw string for invalid JSON test
                response = requests.post(
                    f"{base_url}/next-question",
                    data=test_case['data'],
                    headers=test_case['headers']
                )
            else:
                response = requests.post(
                    f"{base_url}/next-question",
                    json=test_case['data'],
                    headers=test_case['headers']
                )
            
            print(f"   Status: {response.status_code}")
            
            try:
                response_json = response.json()
                print(f"   Response: {response_json}")
            except:
                print(f"   Response (raw): {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("   ❌ Cannot connect to server")
            break
        except Exception as e:
            print(f"   ❌ Request error: {e}")
        
        print()
    
    print("=== Debug Request Inspection ===\n")
    
    # Test the debug endpoint
    print("Testing debug endpoint with problematic request:")
    try:
        # Simulate a potentially problematic request
        debug_data = {"session_id": "test-session", "answer": "test answer"}
        response = requests.post(
            f"{base_url}/debug/next-question",
            json=debug_data
        )
        
        print(f"Debug response: {response.json()}")
        
    except Exception as e:
        print(f"Debug endpoint error: {e}")
    
    print("\n=== Common 422 Causes ===")
    print()
    print("1. MISSING REQUIRED FIELDS:")
    print("   - session_id is required in NextQuestionRequest")
    print("   - Make sure your request includes: {\"session_id\": \"...\"}")
    print()
    print("2. WRONG CONTENT-TYPE:")
    print("   - Use 'Content-Type: application/json'")
    print("   - Don't use 'text/plain' or 'application/x-www-form-urlencoded'")
    print()
    print("3. INVALID JSON FORMAT:")
    print("   - Check for trailing commas, missing quotes, etc.")
    print("   - Validate JSON before sending")
    print()
    print("4. EXTRA/UNEXPECTED FIELDS:")
    print("   - Only send 'session_id' and optionally 'answer'")
    print("   - Remove any extra fields not in the model")
    print()
    print("=== Correct Request Format ===")
    print()
    print("curl -X POST http://localhost:8004/next-question \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"session_id\": \"your-session-id\", \"answer\": \"your-answer\"}'")
    print()
    print("Or without answer:")
    print("curl -X POST http://localhost:8004/next-question \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"session_id\": \"your-session-id\"}'")

if __name__ == "__main__":
    debug_422_error()