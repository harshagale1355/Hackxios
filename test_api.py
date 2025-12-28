#!/usr/bin/env python3
"""
Test the API endpoints using FastAPI's test client
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from fastapi.testclient import TestClient
    from main import app
    
    client = TestClient(app)
    
    def test_endpoints():
        print("Testing FastAPI endpoints...")
        print("=" * 40)
        
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        response = client.get("/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        print("✓ Health endpoint works")
        
        # Test start-session endpoint
        print("\n2. Testing start-session endpoint...")
        response = client.post("/start-session")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")
        assert response.status_code == 200
        assert "session_id" in data
        assert "question" in data
        assert data["question"]["text"] == "Placeholder question"
        assert data["question"]["explanation"] == "This will be replaced later"
        session_id = data["session_id"]
        print("✓ Start-session endpoint works")
        
        # Test next-question endpoint
        print("\n3. Testing next-question endpoint...")
        response = client.post("/next-question", json={"session_id": session_id})
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")
        assert response.status_code == 200
        assert "question" in data
        assert data["question"]["text"] == "Placeholder next question"
        assert data["question"]["explanation"] == "This will be replaced later"
        print("✓ Next-question endpoint works")
        
        # Test next-question with invalid session
        print("\n4. Testing next-question with invalid session...")
        response = client.post("/next-question", json={"session_id": "invalid-id"})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 404
        print("✓ Error handling works correctly")
        
        print("\n" + "=" * 40)
        print("✅ All API tests passed!")
        return True
    
    if __name__ == "__main__":
        success = test_endpoints()
        print(f"\nTest result: {'SUCCESS' if success else 'FAILURE'}")

except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install httpx")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)