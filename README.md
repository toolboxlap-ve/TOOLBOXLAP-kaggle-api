# TOOLBOXLAP — Kaggle Hugging Face / Ollama / ngrok API

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/toolboxlap-ve/TOOLBOXLAP-kaggle-api/blob/main/kaggle/TOOLBOXLAP-Kaggle-HF-Ollama-ngrok.ipynb)

**TOOLBOXLAP** turns a Kaggle GPU session into a temporary OpenAI-compatible API. It installs and runs Ollama without systemd, loads a Hugging Face GGUF or Ollama model, applies a GPU-oriented configuration, and publishes a local proxy through ngrok.

Website: [toolboxlap.com](https://toolboxlap.com)<br>
YouTube: [@TOOLBOXLAP-u1c](https://www.youtube.com/@TOOLBOXLAP-u1c)

## What it provides

The notebook exposes these routes through a temporary public ngrok URL. The proxy accepts both the standard `/v1` form and the same endpoints without `/v1`, so clients do not need provider-specific URL tricks.

- `GET /health`
- `GET /v1/models` — advertises the stable public model ID `toolboxlap`
- `POST /v1/chat/completions` — forwards requests to the selected Ollama backend

The proxy automatically normalizes common client differences: it forces `reasoning_effort: none`, converts `max_completion_tokens` to Ollama's `max_tokens`, supplies a 32768-token default when no output limit is provided, removes common unsupported OpenAI-only fields, and normalizes common tool-call message edge cases. Messages, tools, and streaming are otherwise preserved. The model name sent by a client is replaced with the selected backend model, so clients should use `toolboxlap`.

## Kaggle setup

1. Open the [TOOLBOXLAP Kaggle notebook](https://github.com/toolboxlap-ve/TOOLBOXLAP-kaggle-api/blob/main/kaggle/TOOLBOXLAP-Kaggle-HF-Ollama-ngrok.ipynb) in Kaggle with the button above, then select **Save Version** or **Copy & Edit**.
2. In notebook settings, enable a **GPU** accelerator and enable **Internet**.
3. Run the notebook. Its single Python cell downloads the current versioned launcher from this repository, installs required runtime dependencies (including `zstd` if missing), starts Ollama directly, and asks for a model.
4. At `Model [ENTER = default]:`, press Enter for the supplied TOOLBOXLAP V6 baseline model or enter another supported model reference.
5. Enter your own ngrok authtoken when asked. It is used only in that Kaggle session; it is never written to this repository.
6. Copy the generated **Base URL** (`https://…ngrok…/v1`) into your client. The Cline Model ID is exactly `toolboxlap`.

The launch process first attempts a 131072-token context. It uses `ollama ps` to inspect placement and restarts with 65536 if CPU offload is detected. Use the **Active context** printed at the end in your client configuration.

## Model selection

The default is:

```text
hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M
```

You can paste a Hugging Face GGUF/Ollama reference, for example:

```text
hf.co/owner/gguf-repository:Q4_K_M
```

You can also paste an Ollama model identifier such as:

```text
llama3.2:3b
```

Hugging Face model-page URLs are accepted too. For example, `https://huggingface.co/owner/gguf-repository` becomes `hf.co/owner/gguf-repository`. If the URL identifies a GGUF file, its filename (without `.gguf`) is used as the tag. For best results, use an Ollama-compatible `hf.co/owner/repository:quantization` reference.

## Cline configuration

After the notebook prints its connection details, copy the exact **OpenAI Base URL** printed by the notebook. Do not manually append `/v1`; the printed URL already includes it. The proxy also accepts the root URL without `/v1` for clients that prefer it.

Configure Cline as follows:

| Setting | Value |
| --- | --- |
| Provider | OpenAI Compatible |
| Base URL | copy the exact **OpenAI Base URL** printed by the notebook (already includes `/v1`) |
| Model ID | `toolboxlap` |
| Custom Header | `ngrok-skip-browser-warning = true` |
| Context Window | the **Active context** printed by the notebook |
| Max Output Tokens | `32768` |
| Reasoning Effort | None |

No API key is required by the local proxy unless your client insists on one; any placeholder value may be used in that case.

## ngrok and secrets

ngrok provides a public HTTPS tunnel to the Kaggle proxy. Its URL is **temporary**: it changes when the ngrok process or Kaggle session ends. Treat it as session-scoped access and stop the notebook when finished.

Never commit ngrok authtokens, Hugging Face tokens, API keys, `.env` files, Kaggle credentials, or generated tunnel configuration. This repository’s `.gitignore` intentionally excludes common secret and token files.

## Local validation

```bash
python -m py_compile scripts/toolboxlap_kaggle_hf_ollama_ngrok.py
python -m json.tool kaggle/TOOLBOXLAP-Kaggle-HF-Ollama-ngrok.ipynb > /dev/null
```

MIT licensed. © TOOLBOXLAP.
