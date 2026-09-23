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



def sanitize_tool_schema(value: Any) -> Any:
    """Make common agent-generated JSON Schema safer for Ollama's Go parser."""
    if isinstance(value, list):
        return [sanitize_tool_schema(item) for item in value]

    if not isinstance(value, dict):
        return value

    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        # Ollama's OpenAI-compatible tool parser does not document these
        # OpenAI-specific schema/function extensions.
        if key in {"strict", "$schema"}:
            continue

        # Some generators can emit conditional required objects. Ollama's
        # parser expects required to be an array of strings.
        if key == "required" and not isinstance(item, list):
            continue

        cleaned[key] = sanitize_tool_schema(item)

    return cleaned


def build_safe_openai_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Build a minimal OpenAI-compatible request for an Ollama retry.

    The first request preserves the client's compatible fields. If Ollama
    rejects that request with a 4xx validation error, the proxy retries once
    with provider-specific extras removed.
    """
    safe = dict(payload)

    safe.pop("tool_choice", None)
    safe.pop("parallel_tool_calls", None)
    safe.pop("store", None)
    safe.pop("metadata", None)
    safe.pop("service_tier", None)
    safe.pop("logprobs", None)
    safe.pop("top_logprobs", None)
    safe.pop("modalities", None)
    safe.pop("audio", None)
    safe.pop("max_completion_tokens", None)
    safe.pop("stream_options", None)

    # If JSON-schema response formatting is the incompatible part, let the
    # second attempt proceed as a normal text response rather than failing
    # the whole agent request.
    response_format = safe.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        safe.pop("response_format", None)

    if "max_tokens" not in safe:
        safe["max_tokens"] = 32768

    if isinstance(safe.get("tools"), list):
        safe["tools"] = sanitize_tool_schema(safe["tools"])

    if isinstance(safe.get("messages"), list):
        safe["messages"] = sanitize_tool_schema(safe["messages"])

    return safe


def proxy_request_with_compat_retry(
    payload: dict[str, Any],
    headers: dict[str, str],
) -> Any:
    """POST to Ollama; on HTTP 400/422, retry once with a sanitized body."""
    requests = require_requests()

    upstream = requests.post(
        f"{OLLAMA_URL}/v1/chat/completions",
        json=payload,
        headers=headers,
        stream=bool(payload.get("stream")),
        timeout=900,
    )

    if upstream.status_code not in {400, 422}:
        return upstream

    first_error = upstream.text[:4000]
    safe_payload = build_safe_openai_payload(payload)

    # Streaming responses are not buffered, so the retry can only safely be
    # performed when the first response is a validation failure.
    retry = requests.post(
        f"{OLLAMA_URL}/v1/chat/completions",
        json=safe_payload,
        headers=headers,
        stream=bool(safe_payload.get("stream")),
        timeout=900,
    )

    if retry.status_code >= 400:
        print(
            f"⚠️ Ollama first response HTTP {upstream.status_code}: "
            f"{first_error}",
            flush=True,
        )
        print(
            f"⚠️ Ollama compatibility retry HTTP {retry.status_code}: "
            f"{retry.text[:4000]}",
            flush=True,
        )
    else:
        print(
            f"ℹ️ TOOLBOXLAP compatibility retry succeeded after "
            f"HTTP {upstream.status_code}.",
            flush=True,
        )

    upstream.close()
    return retry


def serve_proxy(backend: str, port: int) -> None:
    """Run the TOOLBOXLAP proxy in the current process.

    This is intentionally Flask/thread based because it matches the known-good
    Kaggle V6 execution path and keeps streaming requests simple.
    """
    try:
        from flask import Flask, Response, jsonify, request, stream_with_context
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "-U", "flask"])
        from flask import Flask, Response, jsonify, request, stream_with_context

    import requests

    app = Flask("toolboxlap_proxy")

    def forwarded_headers():
        blocked = {
            "host",
            "content-length",
            "connection",
            "transfer-encoding",
        }
        return {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in blocked
        }

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "service": "TOOLBOXLAP",
            "model": backend,
            "context": globals().get("ACTIVE_CONTEXT"),
        })

    @app.get("/v1/models")
    def models():
        return jsonify({
            "object": "list",
            "data": [{
                "id": PUBLIC_MODEL_ID,
                "object": "model",
                "owned_by": "toolboxlap",
            }],
        })

    @app.post("/v1/chat/completions")
    def chat_completions():
        body = request.get_json(silent=True)

        if not isinstance(body, dict):
            return jsonify({
                "error": {"message": "JSON body required."}
            }), 400

        # Keep the client-facing ID stable and route internally to the selected
        # backend. Preserve the rest of the client's OpenAI-compatible payload.
        body["model"] = backend
        body["reasoning_effort"] = "none"

        if (
            "max_tokens" not in body
            and "max_completion_tokens" not in body
        ):
            body["max_tokens"] = 32768

        streaming = bool(body.get("stream", False))

        try:
            upstream = requests.post(
                OLLAMA_URL + "/v1/chat/completions",
                headers=forwarded_headers(),
                json=body,
                stream=True,
                timeout=900,
            )
        except requests.RequestException as exc:
            return jsonify({
                "error": {"message": f"Ollama connection failed: {exc}"}
            }), 502

        if streaming:
            content_type = upstream.headers.get(
                "content-type",
                "text/event-stream",
            )

            @stream_with_context
            def generate():
                try:
                    for chunk in upstream.iter_content(
                        chunk_size=8192
                    ):
                        if chunk:
                            yield chunk
                finally:
                    upstream.close()

            return Response(
                generate(),
                status=upstream.status_code,
                content_type=content_type,
            )

        return Response(
            upstream.content,
            status=upstream.status_code,
            content_type=upstream.headers.get(
                "content-type",
                "application/json",
            ),
        )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
    )


def configured_value(value: str | None, env_name: str) -> str | None:
    if value is not None and value.strip():
        return value.strip()
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.strip()
    return None


def choose_model(explicit: str | None = None) -> str:
    value = configured_value(explicit, "TOOLBOXLAP_MODEL")
    if value is None:
        value = input(
            f"Model [ENTER = default: {DEFAULT_MODEL}]: "
        ).strip()

    if not value:
        return DEFAULT_MODEL

    return normalize_model(value)


def choose_ngrok_token(explicit: str | None = None) -> str:
    value = configured_value(explicit, "NGROK_AUTHTOKEN")
    if value is None:
        value = getpass.getpass(
            "ngrok authtoken (used only for this Kaggle session): "
        ).strip()

    if not value:
        raise ValueError(
            "An ngrok authtoken is required to create a public tunnel."
        )

    return value


def start_proxy_thread(backend: str, port: int):
    thread = threading.Thread(
        target=serve_proxy,
        args=(backend, port),
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 45
    requests = require_requests()

    while time.monotonic() < deadline:
        try:
            if requests.get(
                f"http://127.0.0.1:{port}/health",
                timeout=2,
            ).ok:
                return thread
        except requests.RequestException:
            pass
        time.sleep(1)

    raise TimeoutError("TOOLBOXLAP proxy did not become ready.")


def interactive(
    model: str | None = None,
    ngrok_authtoken: str | None = None,
) -> None:
    print(
        "TOOLBOXLAP — Kaggle Hugging Face / Ollama / ngrok API",
        flush=True,
    )

    selected = choose_model(model)

    print(
        f"Selected backend model: {selected}",
        flush=True,
    )

    ensure_zstd()
    ensure_ollama()

    ollama_process = start_ollama(HIGH_CONTEXT)
    context = HIGH_CONTEXT

    try:
        pull_and_load(selected, context)

        uses_cpu, ps_output = placement_uses_cpu()
        print(
            "ollama ps:\n" + ps_output,
            flush=True,
        )

        if uses_cpu:
            print(
                f"CPU offload detected at {context:,} context.",
                flush=True,
            )
            print(
                f"Restarting Ollama with {FALLBACK_CONTEXT:,} context...",
                flush=True,
            )

            stop_process(ollama_process)
            ollama_process = start_ollama(FALLBACK_CONTEXT)

            context = FALLBACK_CONTEXT
            pull_and_load(selected, context)

            _, ps_output = placement_uses_cpu()

            print(
                "ollama ps after fallback:\n" + ps_output,
                flush=True,
            )

        global ACTIVE_CONTEXT
        ACTIVE_CONTEXT = context

        start_proxy_thread(selected, PROXY_PORT)

        ngrok_token = choose_ngrok_token(ngrok_authtoken)

        install_ngrok()

        print(
            "+ ngrok config add-authtoken [redacted]",
            flush=True,
        )

        subprocess.run(
            [
                "ngrok",
                "config",
                "add-authtoken",
                ngrok_token,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        subprocess.run(
            "pkill -f '/usr/local/bin/ngrok http' || true",
            shell=True,
        )
        time.sleep(2)

        ngrok_process = subprocess.Popen(
            [
                "ngrok",
                "http",
                str(PROXY_PORT),
                "--host-header=rewrite",
                "--log=stdout",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        public_url = tunnel_url()
        base_url = public_url.rstrip("/") + "/v1"

        requests = require_requests()

        # Verify exactly the public path Cline will call.
        public_test = requests.post(
            public_url + "/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true",
            },
            json={
                "model": PUBLIC_MODEL_ID,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Reply with exactly: "
                        "TOOLBOXLAP PUBLIC API WORKING"
                    ),
                }],
                "stream": False,
                "max_tokens": 32,
            },
            timeout=300,
        )

        print(
            "\n" + "=" * 72,
            flush=True,
        )
        print(
            "✅ TOOLBOXLAP PUBLIC API READY",
            flush=True,
        )
        print(
            "=" * 72,
            flush=True,
        )
        print(
            "\nCOPY THIS BASE URL INTO CLINE:",
            flush=True,
        )
        print(
            base_url,
            flush=True,
        )
        print(
            "\nCline Model ID:",
            PUBLIC_MODEL_ID,
            flush=True,
        )
        print(
            "Custom Header: ngrok-skip-browser-warning = true",
            flush=True,
        )
        print(
            "Backend model:",
            selected,
            flush=True,
        )
        print(
            "Active context:",
            context,
            flush=True,
        )

        print(
            "\nPublic API test HTTP:",
            public_test.status_code,
            flush=True,
        )

        public_test.raise_for_status()

        print(
            "Public API response:",
            public_test.json()["choices"][0]["message"]["content"],
            flush=True,
        )

        print(
            "\n✅ EVERYTHING IS WORKING",
            flush=True,
        )
        print(
            "Keep this Kaggle session running while the API is in use.",
            flush=True,
        )

        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print(
                "Stopping TOOLBOXLAP services.",
                flush=True,
            )
            stop_process(ngrok_process)

    finally:
        stop_process(ollama_process)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run only the local proxy.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PROXY_PORT,
    )
    parser.add_argument(
        "--model",
        help="Backend model; overrides TOOLBOXLAP_MODEL.",
    )
    parser.add_argument(
        "--ngrok-authtoken",
        help="ngrok token; overrides NGROK_AUTHTOKEN.",
    )

    args = parser.parse_args()

    if args.serve:
        serve_proxy(
            normalize_model(args.backend),
            args.port,
        )
    else:
        interactive(
            model=args.model,
            ngrok_authtoken=args.ngrok_authtoken,
        )


if __name__ == "__main__":
    main()
