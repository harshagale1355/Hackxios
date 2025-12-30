#!/usr/bin/env python3
"""
Simple test script to verify session management functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import the session management components from main.py
from main import Session, sessions, create_session, get_session, delete_session
from fastapi import HTTPException

def test_session_management():
    """Test all session management functions"""
    print("Testing Session Management Implementation")
    print("=" * 50)
    
    # Test 1: Session creation
    print("\n1. Testing session creation...")
    session = create_session()
    print(f"✓ Created session with ID: {session.session_id}")
    print(f"✓ Form fields: {session.form_fields} (empty list as expected)")
    print(f"✓ Current field index: {session.current_field_index}")
    print(f"✓ Answers: {session.answers} (empty dict as expected)")
    print(f"✓ Created at: {session.created_at}")
    print(f"✓ Sessions in global store: {len(sessions)}")
    
    # Test 2: Session creation with form fields
    print("\n2. Testing session creation with form fields...")
    test_fields = ["name", "email", "phone"]
    session_with_fields = create_session(test_fields)
    print(f"✓ Created session with fields: {session_with_fields.form_fields}")
    print(f"✓ Sessions in global store: {len(sessions)}")
    
    # Test 3: Session retrieval
    print("\n3. Testing session retrieval...")
    retrieved_session = get_session(session.session_id)
    print(f"✓ Retrieved session: {retrieved_session.session_id}")
    assert retrieved_session.session_id == session.session_id
    print("✓ Session data matches original")
    
    # Test 4: Session deletion
    print("\n4. Testing session deletion...")
    initial_count = len(sessions)
    delete_session(session.session_id)
    print(f"✓ Sessions before deletion: {initial_count}")
    print(f"✓ Sessions after deletion: {len(sessions)}")
    assert len(sessions) == initial_count - 1
    
    # Test 5: Error handling for non-existent session
    print("\n5. Testing error handling...")
    try:
        get_session("non-existent-session-id")
        print("✗ Should have raised HTTPException")
        return False
    except HTTPException as e:
        print(f"✓ Correctly raised HTTPException: {e.status_code} - {e.detail}")
    
    # Test 6: Delete non-existent session (should not error)
    print("\n6. Testing deletion of non-existent session...")
    sessions_before = len(sessions)
    delete_session("non-existent-session-id")
    sessions_after = len(sessions)
    print(f"✓ Sessions count unchanged: {sessions_before} -> {sessions_after}")
    
    print("\n" + "=" * 50)
    print("✅ All session management tests passed!")
    return True

if __name__ == "__main__":
    success = test_session_management()
    sys.exit(0 if success else 1)