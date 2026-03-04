"""
Long-lived executor server for Pod Pool.

This server runs inside executor pods and accepts cloudpickle payloads via HTTP,
executes functions, and returns results. Pods remain alive for reuse.
"""

import base64
import os
import subprocess
import sys
import traceback

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

        # 2. Install user-specified packages
        packages = config.get("packages", [])
        if packages:
            print(f"[EXECUTOR] Installing packages: {', '.join(packages)}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore"]
                + list(packages)
            )
            print("[EXECUTOR] Packages installed successfully")

        print(f"[EXECUTOR] Executing {fn.__name__}(...)")
        print(f"[EXECUTOR] Args: {len(args)} positional, {len(kwargs)} keyword")

        # 3. Execute function
        result = fn(*args, **kwargs)

        # 4. Serialize result
        print("[EXECUTOR] Execution completed successfully")
        result_dict = {"return_value": result, "error": None}
        result_bytes = cloudpickle.dumps(result_dict)
        result_b64 = base64.b64encode(result_bytes).decode()

        return jsonify({"success": True, "result": result_b64})

    except Exception as e:
        # Capture exception
        print(f"[EXECUTOR] ❌ Execution failed: {e}")
        traceback.print_exc()

        error_dict = {
            "return_value": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        error_bytes = cloudpickle.dumps(error_dict)
        error_b64 = base64.b64encode(error_bytes).decode()

        return jsonify({"success": False, "result": error_b64})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"[EXECUTOR] Starting executor server on port {port}")
    # Single-threaded mode to align with one-request-per-pod semantics
    app.run(host="0.0.0.0", port=port)
