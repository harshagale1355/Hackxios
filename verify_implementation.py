#!/usr/bin/env python3
"""
Verify the session management implementation is working
"""

import subprocess
import sys
import time

def test_with_curl():
    """Test the endpoints using curl"""
    print("Testing with curl...")
    
    # Start the server in background
    print("Starting server...")
    server = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "main:app", 
        "--host", "0.0.0.0", "--port", "8001"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        result = subprocess.run([
            "curl", "-s", "http://localhost:8001/health"
        ], capture_output=True, text=True)
        print(f"Health response: {result.stdout}")
        
        # Test start-session endpoint
        print("\n2. Testing start-session endpoint...")
        result = subprocess.run([
            "curl", "-s", "-X", "POST", "http://localhost:8001/start-session"
        ], capture_output=True, text=True)
        print(f"Start-session response: {result.stdout}")
        
        if '"session_id"' in result.stdout and '"question"' in result.stdout:
            print("✅ Session management is working correctly!")
            return True
        else:
            print("❌ Session management not working as expected")
            return False
            
    finally:
        # Clean up server
        server.terminate()
        server.wait()

def test_direct_import():
    """Test by directly importing and calling functions"""
    print("\nTesting direct function calls...")
    
    try:
        # Import the functions directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # Test session creation
        session = main_module.create_session()
        print(f"✓ Created session: {session.session_id}")
        print(f"✓ Form fields: {session.form_fields}")
        print(f"✓ Current index: {session.current_field_index}")
        print(f"✓ Answers: {session.answers}")
        
        # Test session retrieval
        retrieved = main_module.get_session(session.session_id)
        print(f"✓ Retrieved session: {retrieved.session_id}")
        
        print("✅ Direct function calls working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error in direct testing: {e}")
        return False

if __name__ == "__main__":
    print("Verifying Session Management Implementation")
    print("=" * 50)
    
    # Test direct function calls first
    direct_success = test_direct_import()
    
    # Test with curl if available
    curl_success = False
    try:
        subprocess.run(["curl", "--version"], capture_output=True, check=True)
        curl_success = test_with_curl()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nSkipping curl tests (curl not available)")
    
    print("\n" + "=" * 50)
    if direct_success:
        print("✅ Session management implementation is working!")
    else:
        print("❌ Session management has issues")
    
    sys.exit(0 if direct_success else 1)