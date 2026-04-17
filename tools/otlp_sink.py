"""
Minimal OTLP gRPC sink — accepts and discards telemetry from Ella Core.

This script starts a bare TCP listener on port 4317 that accepts
connections but discards all data. Its sole purpose is to satisfy
Ella Core's requirement for a valid `telemetry.otlp-endpoint` so
that the internal Prometheus counters start incrementing.

Usage:
    python3 tools/otlp_sink.py &
"""

import socket
import threading
import sys


def handle_client(conn, addr):
    """Accept and discard all incoming data."""
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
    except Exception:
        pass
    finally:
        conn.close()


def main(host="127.0.0.1", port=4317):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    print(f"[otlp-sink] Listening on {host}:{port} (discarding all data)")
    sys.stdout.flush()

    try:
        while True:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[otlp-sink] Stopped")
    finally:
        server.close()


if __name__ == "__main__":
    main()
