import subprocess
import time
import sys
import os
import signal

def setup_environment():
    """
    Checks for a virtual environment, creates it if it doesn't exist,
    and installs dependencies from requirements.txt.
    """
    venv_dir = "venv"
    is_windows = (os.name == 'nt')
    
    # Define platform-specific paths for the Python executable and pip
    if is_windows:
        python_executable = os.path.join(venv_dir, 'Scripts', 'python.exe')
        pip_executable = os.path.join(venv_dir, 'Scripts', 'pip.exe')
    else:
        python_executable = os.path.join(venv_dir, 'bin', 'python')
        pip_executable = os.path.join(venv_dir, 'bin', 'pip')

    # 1. Check if the virtual environment directory exists
    if not os.path.isdir(venv_dir):
        print(f"--- Virtual environment not found. Creating one at './{venv_dir}'... ---")
        try:
            # Use the current system's Python to create the venv
            subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)
            print("--- Virtual environment created successfully. ---")
        except subprocess.CalledProcessError as e:
            print(f"!!! ERROR: Failed to create virtual environment. {e} !!!")
            sys.exit(1)

    # 2. Install/update dependencies from requirements.txt using the venv's pip
    print(f"--- Installing or updating dependencies from requirements.txt... ---")
    try:
        subprocess.run([pip_executable, 'install', '-r', 'requirements.txt'], check=True)
        print("--- Dependencies installed successfully. ---")
    except subprocess.CalledProcessError as e:
        print(f"!!! ERROR: Failed to install dependencies. {e} !!!")
        sys.exit(1)
        
    # Return the path to the python executable for the server to use
    return python_executable

def start_server():
    """
    Sets up the environment and then starts the web server and Huey worker.
    """
    # --- NEW: Setup environment first ---
    python_executable = setup_environment()
    print("-" * 35)
    
    is_windows = (os.name == 'nt')
    venv_dir = "venv"

    # --- MODIFIED: Define server command using the venv's executables ---
    if is_windows:
        print("--- Windows OS detected. Using Flask development server. ---")
        server_cmd = [python_executable, 'app.py']
    else:
        print("--- Linux/macOS detected. Using Gunicorn server. ---")
        gunicorn_executable = os.path.join(venv_dir, 'bin', 'gunicorn')
        server_cmd = [
            gunicorn_executable,
            '--workers', '1',
            '--bind', '0.0.0.0:5000',
            'app:app'
        ]

    # --- MODIFIED: Command to start Huey worker now uses the venv's Python ---
    huey_cmd = [
        python_executable,
        '-m', 'huey.bin.huey_consumer',
        'tasks.huey',
        '-w', '4',
        '--health-check-interval', '60'
    ]

    processes = []

    def cleanup(signum, frame):
        """Signal handler to terminate all child processes gracefully."""
        print("\n--- Shutting down all processes ---")
        for p in processes:
            if p.poll() is None:
                try:
                    if not is_windows:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    else:
                        p.terminate()
                except ProcessLookupError:
                    pass
        sys.exit(0)

    # Register the cleanup function for Ctrl+C and termination signals
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        print("--- Starting Web Server ---")
        server_process = subprocess.Popen(server_cmd, preexec_fn=os.setsid if not is_windows else None)
        processes.append(server_process)
        print(f"Web Server started with PID: {server_process.pid}")
        print("-" * 35)
        
        time.sleep(2)

        print("--- Starting Huey Background Worker ---")
        huey_process = subprocess.Popen(huey_cmd, preexec_fn=os.setsid if not is_windows else None)
        processes.append(huey_process)
        print(f"Huey worker started with PID: {huey_process.pid}")
        print("-" * 35)

        print("\nServer and worker are running.")
        print("Press Ctrl+C to stop all processes.")

        server_process.wait()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cleanup(None, None)

if __name__ == '__main__':
    start_server()
