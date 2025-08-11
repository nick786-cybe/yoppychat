import subprocess
import time
import sys
import os
import signal

def start_server():
    """
    Starts the appropriate web server and the Huey worker based on the OS.
    - On Linux/macOS, it uses Gunicorn for the web server.
    - On Windows, it uses the standard Flask development server.
    """
    is_windows = (os.name == 'nt')

    # Define server command based on OS
    if is_windows:
        print("--- Windows OS detected. Using Flask development server. ---")
        server_cmd = [sys.executable, 'app.py']
    else:
        print("--- Linux/macOS detected. Using Gunicorn server. ---")
        server_cmd = [
            'gunicorn',
            '--workers', '1',
            '--bind', '0.0.0.0:5000',
            'app:app'
        ]

    # Command to start the Huey worker (this works on both OSes)
    huey_cmd = [
        sys.executable,
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

        # --- FIX: Replace os.wait() with a cross-platform alternative ---
        # We wait for the server process to exit. If it crashes or is closed,
        # the script will proceed to the finally block to clean up the worker.
        # This works on both Windows and Linux.
        server_process.wait()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Final cleanup call to ensure everything is terminated on exit
        cleanup(None, None)

if __name__ == '__main__':
    start_server()
