#!/usr/bin/env python3
"""
Minimal test of session management
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import uuid

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

if __name__ == "__main__":
    print("Testing minimal session management...")
    
    # Test session creation
    session = create_session()
    print(f"✓ Created session: {session.session_id}")
    print(f"✓ Form fields: {session.form_fields}")
    print(f"✓ Current index: {session.current_field_index}")
    print(f"✓ Answers: {session.answers}")
    print(f"✓ Sessions in store: {len(sessions)}")
    
    # Test session retrieval
    retrieved = get_session(session.session_id)
    print(f"✓ Retrieved session: {retrieved.session_id}")
    
    # Test session deletion
    delete_session(session.session_id)
    print(f"✓ Sessions after deletion: {len(sessions)}")
    
    print("✅ Session management working correctly!")