"""
Long-lived executor server for Pod Pool.

This server runs inside executor pods and accepts cloudpickle payloads via HTTP,
executes functions, and returns results. Pods remain alive for reuse.
"""

import base64
import io
import os
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

try:
    import cloudpickle
except ImportError:
    print("[EXECUTOR] Installing cloudpickle...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore", "cloudpickle"]
    )
    import cloudpickle

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/execute", methods=["POST"])
def execute():
    """
    Execute a serialized function payload.

    Request body: cloudpickle-serialized dict with:
    - fn: function to execute
    - args: positional arguments
    - kwargs: keyword arguments
    - config: execution config (packages, env, etc.)

    Returns: JSON with result or error
    """
    try:
        # 1. Load serialized payload
        payload_bytes = request.data
        if not payload_bytes:
            return jsonify({"error": "Empty payload"}), 400

        payload = cloudpickle.loads(payload_bytes)

        fn = payload.get("fn")
        args = payload.get("args", ())
        kwargs = payload.get("kwargs", {})
        config = payload.get("config", {})

        if not fn:
            return jsonify({"error": "Missing 'fn' in payload"}), 400

        logs = io.StringIO()

        # 2. Install user-specified packages
        packages = config.get("packages", [])
        if packages:
            logs.write(f"[EXECUTOR] Installing packages: {', '.join(packages)}\n")
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--root-user-action=ignore"]
                + list(packages),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.stdout:
                logs.write(proc.stdout)
            if proc.returncode != 0:
                raise RuntimeError(f"pip install failed:\n{proc.stdout}")
            logs.write("[EXECUTOR] Packages installed successfully\n")

        logs.write(f"[EXECUTOR] Executing {fn.__name__}(...)\n")

        # 3. Execute function, capturing its print() output
        fn_stdout = io.StringIO()
        fn_stderr = io.StringIO()
        with redirect_stdout(fn_stdout), redirect_stderr(fn_stderr):
            result = fn(*args, **kwargs)

        captured = fn_stdout.getvalue()
        if fn_stderr.getvalue():
            captured += fn_stderr.getvalue()
        if captured:
            logs.write(captured)

        logs.write("[EXECUTOR] Execution completed successfully\n")

        # 4. Serialize result
        result_dict = {"return_value": result, "error": None}
        result_bytes = cloudpickle.dumps(result_dict)
        result_b64 = base64.b64encode(result_bytes).decode()

        return jsonify({"success": True, "result": result_b64, "stdout": logs.getvalue()})

    except Exception as e:
        # Capture exception
        tb_str = traceback.format_exc()
        print(f"[EXECUTOR] ❌ Execution failed: {e}")
        print(tb_str)

        error_dict = {
            "return_value": None,
            "error": str(e),
            "traceback": tb_str,
        }
        error_bytes = cloudpickle.dumps(error_dict)
        error_b64 = base64.b64encode(error_bytes).decode()

        return jsonify({
            "success": False,
            "result": error_b64,
            "stdout": f"[EXECUTOR] \u274c Execution failed: {e}\n{tb_str}",
        })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"[EXECUTOR] Starting executor server on port {port}")
    # Single-threaded mode to align with one-request-per-pod semantics
    app.run(host="0.0.0.0", port=port)
