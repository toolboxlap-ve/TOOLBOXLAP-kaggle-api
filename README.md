# TOOLBOXLAP — Kaggle Hugging Face / Ollama / ngrok API

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/toolboxlap-ve/TOOLBOXLAP-kaggle-api/blob/main/kaggle/TOOLBOXLAP-Kaggle-HF-Ollama-ngrok.ipynb)

**TOOLBOXLAP** turns a Kaggle GPU session into a temporary OpenAI-compatible API powered by Ollama.

It is designed to be simple to use and simple to explain:

**GitHub → Open in Kaggle → GPU + Internet → choose model → ngrok token → copy Base URL → use in Cline**

Website: [toolboxlap.com](https://toolboxlap.com)  |  YouTube: [@TOOLBOXLAP-u1c](https://www.youtube.com/@TOOLBOXLAP-u1c)  |  GitHub: [TOOLBOXLAP-kaggle-api](https://github.com/toolboxlap-ve/TOOLBOXLAP-kaggle-api)

## What this project does

The notebook starts a temporary model-serving stack inside a Kaggle GPU session:

```text
Kaggle GPU
   ↓
Ollama
   ↓
Hugging Face GGUF or Ollama model
   ↓
TOOLBOXLAP OpenAI-compatible proxy
   ↓
ngrok HTTPS tunnel
   ↓
Cline / other OpenAI-compatible clients
```

The public API exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Clients use the stable public model ID:

```text
toolboxlap
```

The proxy routes that public ID to the backend model selected when the Kaggle session starts.

## Quick start

### 1. Open the notebook

Use the **Open in Kaggle** button at the top of this README.

You can also open the notebook file directly:

[TOOLBOXLAP-Kaggle-HF-Ollama-ngrok.ipynb](https://github.com/toolboxlap-ve/TOOLBOXLAP-kaggle-api/blob/main/kaggle/TOOLBOXLAP-Kaggle-HF-Ollama-ngrok.ipynb)

### 2. Enable Kaggle GPU and Internet

In the Kaggle notebook settings:

```text
Accelerator: GPU
Internet: ON
```

The notebook is intended to run inside a Kaggle GPU session.

### 3. Run the single cell

The notebook has one runnable Python cell.

It downloads the current launcher from this repository instead of duplicating the full launcher code inside the notebook.

You will be asked:

```text
Model [ENTER = default: ...]:
```

Press **Enter** to use the TOOLBOXLAP default model.

Then enter your own ngrok authtoken. The token is supplied at runtime and is not stored in this repository.

### 4. Copy the generated Base URL

When startup finishes, the notebook prints a block like:

```text
======================================================================

✅ TOOLBOXLAP PUBLIC API READY

======================================================================

COPY THIS BASE URL INTO CLINE:
https://xxxxxxxx.ngrok-free.dev/v1

Cline Model ID:
toolboxlap

Custom Header: ngrok-skip-browser-warning = true

Backend model:
hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M

Active context:
131072

Public API test HTTP:
200

Public API response:
TOOLBOXLAP PUBLIC API WORKING

✅ EVERYTHING IS WORKING
```

Copy the **entire Base URL exactly as printed**.

Do not manually add another `/v1`.

### 5. Configure Cline

Use:

| Setting | Value |
| --- | --- |
| Provider | OpenAI Compatible |
| Base URL | the generated URL printed by the notebook |
| API Key | any placeholder value if Cline requires one |
| Model ID | `toolboxlap` |
| Custom Header | `ngrok-skip-browser-warning = true` |
| Context Window | the printed **Active context** |
| Max Output Tokens | `32768` |
| Reasoning Effort | None |

The local TOOLBOXLAP proxy does not require an upstream API key.

## Choosing another model

The default model is:

```text
hf.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M
```

At the model prompt, you can press Enter or paste another supported model reference.

### Hugging Face GGUF

Use an Ollama-compatible `hf.co` reference:

```text
hf.co/OWNER/REPOSITORY:Q4_K_M
```

Example:

```text
hf.co/your-org/your-gguf-model:Q4_K_M
```

A Hugging Face model-page URL is also accepted:

```text
https://huggingface.co/OWNER/REPOSITORY
```

For GGUF files, you can also use a file URL; the launcher derives the GGUF filename as the tag.

### Ollama model

You can enter a normal Ollama model ID, for example:

```text
llama3.2:3b
```

The public client-facing Model ID remains:

```text
toolboxlap
```

Only the backend model changes.

## Performance and context

The launcher is configured for the tested Kaggle GPU workflow.

It starts with:

```text
Context target: 131072
```

It checks placement with `ollama ps`.

If CPU offload is detected, it restarts with:

```text
Fallback context: 65536
```

The notebook prints the **Active context** actually being used. Use that number in the client.

The current tuned Ollama environment includes:

```text
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE=30m
```

## ngrok

ngrok provides the temporary HTTPS public endpoint.

The URL is session-scoped and can change when the Kaggle session or ngrok process ends.

Keep the Kaggle cell running while the API is being used. Stop the session when you are finished.

The ngrok command uses host-header rewriting so the public tunnel forwards correctly to the local proxy.

## Security

Never commit:

- ngrok authtokens
- Hugging Face tokens
- API keys
- Kaggle credentials
- `.env` files
- generated ngrok configuration

The notebook asks for the ngrok token at runtime.

Do not publish your personal token in a video, screenshot, README, Git commit, or GitHub issue.

## Repository structure

```text
TOOLBOXLAP-kaggle-api/
├── README.md
├── LICENSE
├── .gitignore
├── kaggle/
│   └── TOOLBOXLAP-Kaggle-HF-Ollama-ngrok.ipynb
├── scripts/
│   └── toolboxlap_kaggle_hf_ollama_ngrok.py
└── .github/
    └── ISSUE_TEMPLATE/
        └── bug_report.md
```

The notebook is intentionally small. The full launcher lives in `scripts/`, so updates can be published without duplicating the main Python code into the notebook.

## Simple video flow

For a walkthrough video, the intended sequence is:

```text
1. Open the GitHub repository
2. Click Open in Kaggle
3. Turn on GPU
4. Turn on Internet
5. Run the notebook
6. Press Enter for the default model
7. Enter the ngrok token
8. Wait for EVERYTHING IS WORKING
9. Copy the printed Base URL
10. Open Cline
11. Select OpenAI Compatible
12. Paste Base URL
13. Set Model ID = toolboxlap
14. Add ngrok-skip-browser-warning = true
15. Send "hi"
```

## Local validation

```bash
python -m py_compile scripts/toolboxlap_kaggle_hf_ollama_ngrok.py
python -m json.tool kaggle/TOOLBOXLAP-Kaggle-HF-Ollama-ngrok.ipynb > /dev/null
```

## License

MIT License — © TOOLBOXLAP.
