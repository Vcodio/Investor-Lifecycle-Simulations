

import sys
import os
import subprocess

def check_streamlit():
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(script_dir, 'app.py')
    if not os.path.exists(app_file):
        print(f"❌ ERROR: app.py not found at {app_file}")
        input("Press Enter to exit...")
        return 1
    print(f"✅ Found app.py at {app_file}")
    print()
    print("🚀 Starting Streamlit app...")
    print("   If the browser doesn't open, go to: http://localhost:8501")
    print()
    print("   Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
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
