#!/usr/bin/env python3
"""TOOLBOXLAP Kaggle launcher and OpenAI-compatible Ollama proxy.

Run with no arguments in a Kaggle GPU notebook. Use --serve internally to
start only the proxy after Ollama has been configured.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

DEFAULT_MODEL = "hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M"
PUBLIC_MODEL_ID = "toolboxlap"
OLLAMA_URL = "http://127.0.0.1:11434"
PROXY_PORT = 8000
HIGH_CONTEXT = 131072
FALLBACK_CONTEXT = 65536

OLLAMA_TUNING = {
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_NUM_PARALLEL": "1",
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_KEEP_ALIVE": "30m",
}


def run(command: list[str], *, check: bool = True, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command))
    return subprocess.run(command, check=check, text=True, **kwargs)


def require_requests() -> Any:
    try:
        import requests
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "requests"])
        import requests
    return requests


def ensure_zstd() -> None:
    if shutil.which("zstd"):
        return
    if not shutil.which("apt-get"):
        raise RuntimeError("zstd is missing and apt-get is unavailable.")
    run(["apt-get", "update"])
    run(["apt-get", "install", "-y", "zstd"])


def ensure_ollama() -> None:
    if shutil.which("ollama"):
        return
    run(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"])
    if not shutil.which("ollama"):
        raise RuntimeError("Ollama installation did not put 'ollama' on PATH.")


def normalize_model(raw: str) -> str:
    """Accept an Ollama ID, hf.co reference, or Hugging Face model/file URL."""
    value = raw.strip()
    if not value:
        return DEFAULT_MODEL
    if value.startswith("hf.co/"):
        return value
    parsed = urlparse(value)
    if parsed.netloc.lower() in {"huggingface.co", "www.huggingface.co"}:
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("A Hugging Face URL must include owner and repository.")
        owner, repository = parts[0], parts[1]
        tag = ""
        # /owner/repo/{blob,resolve}/branch/file.gguf → :file
        if len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
            filename = parts[-1]
            if filename.lower().endswith(".gguf"):
                tag = ":" + filename[:-5]
        # /owner/repo?revision=Q4_K_M and URL fragments are convenient tags.
        query_tag = re.search(r"(?:revision|tag)=([^&]+)", parsed.query)
        if query_tag:
            tag = ":" + unquote(query_tag.group(1))
        return f"hf.co/{owner}/{repository}{tag}"
    return value


def ollama_env(context: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(OLLAMA_TUNING)
    env["OLLAMA_CONTEXT_LENGTH"] = str(context)
    return env


def wait_for_ollama(timeout: int = 90) -> None:
    requests = require_requests()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("Ollama did not become ready.")


def start_ollama(context: int) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        ["ollama", "serve"], env=ollama_env(context), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    wait_for_ollama()
    return proc


def stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def pull_and_load(model: str, context: int) -> None:
    run(["ollama", "pull", model])
    requests = require_requests()
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": "", "stream": False, "keep_alive": "30m",
              "options": {"num_ctx": context}},
        timeout=900,
    )
    response.raise_for_status()


def placement_uses_cpu() -> tuple[bool, str]:
    result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=True)
    output = result.stdout.strip()
    # Ollama reports mixed placement as e.g. "79%/21% CPU/GPU" or a CPU processor.
    uses_cpu = bool(re.search(r"\bCPU\b", output, flags=re.IGNORECASE))
    return uses_cpu, output


def install_ngrok() -> None:
    if shutil.which("ngrok"):
        return
    if platform.system() != "Linux":
        raise RuntimeError("This launcher installs the official Linux ngrok agent for Kaggle.")
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"aarch64", "arm64"} else "amd64"
    archive = f"ngrok-v3-stable-linux-{arch}.tgz"
    url = f"https://bin.equinox.io/c/bNyj1mQVY4c/{archive}"
    run(["bash", "-lc", f"curl -fsSL {url} | tar xz -C /usr/local/bin ngrok"])
    if not shutil.which("ngrok"):
        raise RuntimeError("Official ngrok agent installation failed.")


def wait_for_proxy(port: int, timeout: int = 45) -> None:
    requests = require_requests()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("TOOLBOXLAP proxy did not become ready.")


def tunnel_url(timeout: int = 45) -> str:
    requests = require_requests()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            tunnels = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=2).json()["tunnels"]
            for tunnel in tunnels:
                if tunnel["public_url"].startswith("https://"):
                    return tunnel["public_url"]
        except (requests.RequestException, KeyError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError("ngrok did not publish an HTTPS tunnel.")


def serve_proxy(backend: str, port: int) -> None:
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, Response, StreamingResponse
        import requests
        import uvicorn
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "requests"])
        return serve_proxy(backend, port)

    app = FastAPI(title="TOOLBOXLAP Proxy", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise HTTPException(status_code=503, detail=f"Ollama unavailable: {exc}") from exc
        return {"status": "ok", "model": PUBLIC_MODEL_ID, "backend": backend}

    @app.get("/models")
    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": PUBLIC_MODEL_ID, "object": "model", "owned_by": "TOOLBOXLAP"}]}

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be JSON.") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

        # Normalize common OpenAI-compatible client differences at the proxy boundary.
        # Keep the client-facing model ID stable while routing to the selected backend.
        payload["model"] = backend
        payload["reasoning_effort"] = "none"

        # Ollama's documented OpenAI-compatible endpoint uses max_tokens.
        if "max_tokens" not in payload and "max_completion_tokens" in payload:
            payload["max_tokens"] = payload["max_completion_tokens"]
        payload.pop("max_completion_tokens", None)

        if "max_tokens" not in payload:
            payload["max_tokens"] = 32768

        # Remove common OpenAI-only fields that Ollama does not document for
        # /v1/chat/completions and that can cause compatibility failures.
        for unsupported in (
            "parallel_tool_calls",
            "store",
            "metadata",
            "service_tier",
            "logprobs",
            "top_logprobs",
            "modalities",
            "audio",
        ):
            payload.pop(unsupported, None)

        # Normalize common agent message edge cases.
        messages = payload.get("messages")
        if isinstance(messages, list):
            normalized_messages = []
            for message in messages:
                if not isinstance(message, dict):
                    normalized_messages.append(message)
                    continue

                message = dict(message)

                if message.get("role") == "developer":
                    message["role"] = "system"

                if (
                    message.get("role") == "assistant"
                    and message.get("tool_calls")
                    and message.get("content") == ""
                ):
                    message["content"] = None

                normalized_messages.append(message)

            payload["messages"] = normalized_messages

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length"}
        }

        try:
            upstream = requests.post(
                f"{OLLAMA_URL}/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=bool(payload.get("stream")),
                timeout=900,
            )
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama request failed: {exc}",
            ) from exc

        content_type = upstream.headers.get(
            "content-type",
            "application/json",
        )

        if payload.get("stream"):
            return StreamingResponse(
                upstream.iter_content(chunk_size=None),
                status_code=upstream.status_code,
                media_type=content_type,
            )

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=content_type,
        )

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def configured_value(value: str | None, env_name: str) -> str | None:
    """Return an explicit CLI value or environment value without prompting."""
    if value is not None:
        return value.strip()
    env_value = os.getenv(env_name)
    return env_value.strip() if env_value is not None else None


def interactive(*, model: str | None = None, ngrok_authtoken: str | None = None) -> None:
    print("TOOLBOXLAP — Kaggle Hugging Face / Ollama / ngrok API")
    selected_input = configured_value(model, "TOOLBOXLAP_MODEL")
    if selected_input is None:
        selected_input = input(f"Model [ENTER = default]: ").strip()
    selected = normalize_model(selected_input)
    print(f"Selected backend model: {selected}")
    ensure_zstd()
    ensure_ollama()
    ollama = start_ollama(HIGH_CONTEXT)
    context = HIGH_CONTEXT
    try:
        pull_and_load(selected, context)
        uses_cpu, ps_output = placement_uses_cpu()
        print("ollama ps:\n" + ps_output)
        if uses_cpu:
            print(f"CPU placement detected; restarting with {FALLBACK_CONTEXT} context.")
            stop_process(ollama)
            ollama = start_ollama(FALLBACK_CONTEXT)
            context = FALLBACK_CONTEXT
            pull_and_load(selected, context)
            _, ps_output = placement_uses_cpu()
            print("ollama ps after fallback:\n" + ps_output)

        proxy = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--serve",
                                  "--backend", selected, "--port", str(PROXY_PORT)])
        wait_for_proxy(PROXY_PORT)
        install_ngrok()
        authtoken = configured_value(ngrok_authtoken, "NGROK_AUTHTOKEN")
        if authtoken is None:
            authtoken = input("ngrok authtoken (used only for this Kaggle session): ").strip()
        if not authtoken:
            raise ValueError("An ngrok authtoken is required to create a public tunnel.")
        print("+ ngrok config add-authtoken [redacted]")
        subprocess.run(["ngrok", "config", "add-authtoken", authtoken], check=True, text=True)
        ngrok = subprocess.Popen(["ngrok", "http", f"--host-header=rewrite", str(PROXY_PORT), "--log", "stdout"])
        public_url = tunnel_url()
        requests = require_requests()
        test = requests.get(f"{public_url}/health", headers={"ngrok-skip-browser-warning": "true"}, timeout=30)
        test.raise_for_status()
        print("\nPublic URL:", public_url)
        print("OpenAI Base URL:", f"{public_url}/v1")
        print("OpenAI Base URL (root also accepted):", public_url)
        print("Model ID = toolboxlap")
        print("Backend model:", selected)
        print("Active context:", context)
        print("\nEVERYTHING IS WORKING")
        print("Keep this cell running while you use the API. Interrupt it to close the tunnel.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("Stopping TOOLBOXLAP services.")
            stop_process(ngrok)
            stop_process(proxy)
    finally:
        stop_process(ollama)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Run only the local proxy.")
    parser.add_argument("--backend", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=PROXY_PORT)
    parser.add_argument("--model", help="Backend model; overrides TOOLBOXLAP_MODEL.")
    parser.add_argument("--ngrok-authtoken", help="ngrok token; overrides NGROK_AUTHTOKEN.")
    args = parser.parse_args()
    if args.serve:
        serve_proxy(args.backend, args.port)
    else:
        interactive(model=args.model, ngrok_authtoken=args.ngrok_authtoken)


if __name__ == "__main__":
    main()
