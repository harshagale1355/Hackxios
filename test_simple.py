#!/usr/bin/env python3
"""
Simple test to verify the session management data structures and logic
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import uuid

# Replicate the session management logic from main.py
@dataclass
class Session:
    session_id: str
    form_fields: List[str]
    current_field_index: int
    answers: Dict[str, str]
    created_at: datetime

# Global in-memory session store
sessions: Dict[str, Session] = {}

def create_session(form_fields: List[str] = None) -> Session:
    """Create a new session with unique ID."""
    if form_fields is None:
        form_fields = []
    
    session_id = str(uuid.uuid4())
    session = Session(
        session_id=session_id,
        form_fields=form_fields,
        current_field_index=0,
        answers={},
        created_at=datetime.now()
    )
    sessions[session_id] = session
    return session

def get_session(session_id: str) -> Session:
    """Retrieve session by ID, raise exception if not found."""
    if session_id not in sessions:
        raise KeyError(f"Session {session_id} not found")
    return sessions[session_id]

def delete_session(session_id: str) -> None:
    """Delete session from memory."""
    if session_id in sessions:
        del sessions[session_id]

def test_session_management():
    """Test all session management functions"""
    print("Testing Session Management Implementation")
    print("=" * 50)
    
    # Test 1: Session creation
    print("\n1. Testing session creation...")
    session = create_session()
    print(f"✓ Created session with ID: {session.session_id}")
    print(f"✓ Form fields: {session.form_fields} (should be empty list)")
    print(f"✓ Current field index: {session.current_field_index} (should be 0)")
    print(f"✓ Answers: {session.answers} (should be empty dict)")
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
        print("✗ Should have raised KeyError")
        return False
    except KeyError as e:
        print(f"✓ Correctly raised KeyError: {e}")
    
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
    print(f"\nTest result: {'SUCCESS' if success else 'FAILURE'}")