
import sys
import os
import subprocess
import shutil


def find_streamlit():
    """Return (cmd, mode) where cmd is a list and mode is 'bin' or 'module'."""
    # 1. Prefer a streamlit binary on PATH (covers pipx installs)
    st_bin = shutil.which('streamlit')
    if st_bin:
        return [st_bin], 'bin'

    # 2. Fall back to `python -m streamlit` if the current interpreter has it
    try:
        subprocess.run(
            [sys.executable, '-m', 'streamlit', '--version'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return [sys.executable, '-m', 'streamlit'], 'module'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None, None


def main():
    print("=" * 60)
    print("Lifecycle Retirement Simulation - Streamlit App Launcher")
    print("=" * 60)
    print()
    print("Checking dependencies...")

    st_cmd, mode = find_streamlit()
    if st_cmd is None:
        print("❌ ERROR: Streamlit is not installed!")
        print()
        print("Please install it using:")
        print("  pip install streamlit")
        print("  # or, for an isolated install:")
        print("  pipx install streamlit")
        print()
        input("Press Enter to exit...")
        return 1

    label = st_cmd[0] if mode == 'bin' else f"{sys.executable} -m streamlit"
    print(f"✅ Streamlit found ({label})")
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
        subprocess.run(
            st_cmd + ['run', app_file,
                      '--server.headless', 'false',
                      '--browser.gatherUsageStats', 'false']
        )
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
