#!/usr/bin/env python3
"""
Launcher script for Streamlit Lifecycle Retirement Simulation App

This script:
1. Checks if Streamlit is installed
2. Opens the app in your default browser
3. Provides helpful error messages if something goes wrong
"""

import sys
import os
import subprocess
import webbrowser
import time

def check_streamlit():
    """Check if streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("Lifecycle Retirement Simulation - Streamlit App Launcher")
    print("=" * 60)
    print()
    
    # Check if streamlit is installed
    print("Checking dependencies...")
    if not check_streamlit():
        print("❌ ERROR: Streamlit is not installed!")
        print()
        print("Please install it using:")
        print("  pip install streamlit")
        print()
        input("Press Enter to exit...")
        return 1
    
    print("✅ Streamlit is installed")
    print()
    
    # Get the app file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(script_dir, 'app.py')
    
    if not os.path.exists(app_file):
        print(f"❌ ERROR: app.py not found at {app_file}")
        input("Press Enter to exit...")
        return 1
    
    print(f"✅ Found app.py at {app_file}")
    print()
    
    # Open browser after a short delay (Streamlit takes a moment to start)
    print("🚀 Starting Streamlit app...")
    print("   The app will open in your default browser automatically.")
    print("   If it doesn't open, go to: http://localhost:8501")
    print()
    print("   Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Open browser after 2 seconds
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:8501')
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_file,
            '--server.headless', 'false',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: Failed to start Streamlit: {e}")
        input("Press Enter to exit...")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

