"""
KitchenHero — app.py
Flask AI Bridge: Recipe Generation (HuggingFace) + Image Generation (SenseNova)
"""

import os
import base64
import subprocess
import shlex
import logging
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow requests from the HTML frontend (any origin)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config  (override via environment variables)
# ─────────────────────────────────────────────
HF_API_KEY      = os.getenv("HF_API_KEY", "")          # HuggingFace token
HF_MODEL        = "google/gemma-2-9b-it"
HF_API_URL      = f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions"

SENSENOVA_MODEL  = os.getenv("SENSENOVA_MODEL_PATH", "sensenova/SenseNova-U1-8B-MoT")
SENSENOVA_GGUF   = os.getenv("SENSENOVA_GGUF_PATH", "")   # optional GGUF checkpoint path
OUTPUT_IMAGE     = Path("output.png")
INFERENCE_SCRIPT = Path("examples/t2i/inference.py")


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "KitchenHero backend is running 🚀"})


# ─────────────────────────────────────────────
# ROUTE: /recipe
# Accepts: { "ingredients": ["spinach", "eggs", ...] }
# Returns: { "recipe": "..." }
# ─────────────────────────────────────────────
@app.route("/recipe", methods=["POST"])
def generate_recipe():
    data = request.get_json(force=True)
    ingredients: list = data.get("ingredients", [])

    if not ingredients:
        return jsonify({"error": "No ingredients provided"}), 400

    ingredient_str = ", ".join(ingredients)
    prompt = (
        f"You are a creative chef. Using ONLY these expiring ingredients: {ingredient_str}. "
        "Suggest ONE delicious recipe. Give it a creative name, list the ingredients used, "
        "and provide brief cooking steps (under 200 words). Be enthusiastic and practical!"
    )

    logger.info("Requesting recipe for: %s", ingredient_str)

    try:
        recipe_text = _call_huggingface(prompt)
        return jsonify({"recipe": recipe_text})
    except Exception as exc:
        logger.error("Recipe generation failed: %s", exc)
        return jsonify({"error": str(exc)}), 502


# ─────────────────────────────────────────────
# ROUTE: /generate-image
# Accepts: { "prompt": "A beautiful plate of ..." }
# Returns: { "base64": "<png data>", "image_path": "output.png" }
# ─────────────────────────────────────────────
@app.route("/generate-image", methods=["POST"])
def generate_image():
    data   = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    logger.info("Generating image for prompt: %s", prompt[:80])

    try:
        _run_sensenova(prompt)

        if not OUTPUT_IMAGE.exists():
            return jsonify({"error": "Image generation completed but output.png not found"}), 500

        with open(OUTPUT_IMAGE, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({
            "base64":     encoded,
            "image_path": str(OUTPUT_IMAGE),
            "message":    "Image generated successfully"
        })

    except subprocess.CalledProcessError as exc:
        logger.error("SenseNova script error (exit %d): %s", exc.returncode, exc.stderr)
        return jsonify({
            "error":   "Image generation failed",
            "detail":  exc.stderr or "No stderr captured",
            "returncode": exc.returncode
        }), 500
    except FileNotFoundError as exc:
        logger.error("inference.py not found: %s", exc)
        return jsonify({
            "error":  "SenseNova inference script not found",
            "detail": (
                f"Expected at: {INFERENCE_SCRIPT.resolve()}. "
                "Install SenseNova and ensure the repo structure is intact."
            )
        }), 500
    except Exception as exc:
        logger.error("Unexpected image generation error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────
# ROUTE: /output-image  (serve the last PNG)
# ─────────────────────────────────────────────
@app.route("/output-image", methods=["GET"])
def serve_output_image():
    if not OUTPUT_IMAGE.exists():
        return jsonify({"error": "No image has been generated yet"}), 404
    return send_file(OUTPUT_IMAGE, mimetype="image/png")


# ─────────────────────────────────────────────
# Internal: Call HuggingFace Chat Completions
# ─────────────────────────────────────────────
def _call_huggingface(prompt: str) -> str:
    """Send a prompt to the HuggingFace Inference API and return the text."""
    api_key = HF_API_KEY
    if not api_key:
        raise ValueError(
            "HF_API_KEY environment variable is not set. "
            "Export it before starting the server: export HF_API_KEY=hf_..."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       HF_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  512,
        "temperature": 0.8,
    }

    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()

    result = resp.json()
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected HuggingFace response format: {result}") from exc


# ─────────────────────────────────────────────
# Internal: Run SenseNova inference script
# ─────────────────────────────────────────────
def _run_sensenova(prompt: str) -> None:
    """
    Execute the SenseNova text-to-image inference script safely.

    Installs the required extras if not already present, then runs:
      python examples/t2i/inference.py
        --model_path <SENSENOVA_MODEL>
        [--gguf_checkpoint <SENSENOVA_GGUF>]
        --prompt <prompt>
        --output output.png

    The prompt is passed as a list argument to subprocess to prevent
    shell injection — no shell=True is used.
    """

    # ── Step 1: ensure gguf extra is installed ──────────────────────────
    logger.info("Ensuring SenseNova gguf extras are installed...")
    subprocess.run(
        ["pip", "install", "--quiet", "gguf>=0.10.0", "diffusers>=0.30.0"],
        check=True,
        capture_output=True,
        text=True,
    )

    # ── Step 2: build the inference command ────────────────────────────
    cmd = [
        "python", str(INFERENCE_SCRIPT),
        "--model_path", SENSENOVA_MODEL,
        "--prompt",     prompt,          # safe: passed as list, no shell
        "--output",     str(OUTPUT_IMAGE),
    ]

    # Optionally add GGUF checkpoint if configured
    if SENSENOVA_GGUF:
        cmd += ["--gguf_checkpoint", SENSENOVA_GGUF]

    logger.info("Running: %s", " ".join(shlex.quote(c) for c in cmd))

    # ── Step 3: execute ────────────────────────────────────────────────
    result = subprocess.run(
        cmd,
        check=True,         # raises CalledProcessError on non-zero exit
        capture_output=True,
        text=True,
        timeout=600,        # 10-minute timeout for large model inference
    )

    logger.info("SenseNova stdout: %s", result.stdout[-500:] if result.stdout else "(none)")
    if result.stderr:
        logger.warning("SenseNova stderr: %s", result.stderr[-500:])


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("""
  ██╗  ██╗██╗████████╗ ██████╗██╗  ██╗███████╗███╗  ██╗██╗  ██╗███████╗██████╗  ██████╗
  ██║ ██╔╝██║╚══██╔══╝██╔════╝██║  ██║██╔════╝████╗ ██║██║  ██║██╔════╝██╔══██╗██╔═══██╗
  █████╔╝ ██║   ██║   ██║     ███████║█████╗  ██╔██╗██║███████║█████╗  ██████╔╝██║   ██║
  ██╔═██╗ ██║   ██║   ██║     ██╔══██║██╔══╝  ██║╚████║██╔══██║██╔══╝  ██╔══██╗██║   ██║
  ██║  ██╗██║   ██║   ╚██████╗██║  ██║███████╗██║ ╚███║██║  ██║███████╗██║  ██║╚██████╔╝
  ╚═╝  ╚═╝╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝
    """)
    print("  🚀 KitchenHero backend starting on http://localhost:5000")
    print(f"  🤖 HuggingFace model : {HF_MODEL}")
    print(f"  🖼️  SenseNova model  : {SENSENOVA_MODEL}")
    print(f"  📁 GGUF checkpoint  : {SENSENOVA_GGUF or '(not set — using full precision)'}")
    print("  ─────────────────────────────────────────────────────────")
    print("  Set env vars before starting:")
    print("    export HF_API_KEY=hf_your_key_here")
    print("    export SENSENOVA_MODEL_PATH=sensenova/SenseNova-U1-8B-MoT")
    print("    export SENSENOVA_GGUF_PATH=/path/to/SenseNova-U1-8B-MoT-Merger-Q4_K_M.gguf")
    print("  ─────────────────────────────────────────────────────────\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
