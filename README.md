# EchoHire Model Service

This repository contains EchoHire's custom model and inference-serving layer. It loads a Hugging Face causal language model and uses a job title plus skills to generate interview-question text. EchoHire is an AI-powered recruitment and interview platform developed as a Final Year Project.

## Project Context

EchoHire is composed of separate application components. This repository isolates model inference from the main application backend, while the frontend, backend, and database live in their own repositories. The HTTP service is designed to be called by another application component; this repository does not include a backend client or orchestration code that demonstrates that integration.

## Model Purpose

The implemented task is **interview question generation**. The API constructs a prompt from a job title and a comma-separated skills string, then asks the model for questions only. It does not implement candidate scoring, résumé ranking, answer assessment, or a general-purpose recruitment workflow.

## Model Architecture

- **Model artifact:** [`Zeeshan506/echohire-qgen-distilgpt2`](https://huggingface.co/Zeeshan506/echohire-qgen-distilgpt2)
- **Model type:** Hugging Face `AutoModelForCausalLM` and `AutoTokenizer`
- **Published artifact metadata:** `GPT2LMHeadModel` / GPT-2, with `openai-community/gpt2` recorded as its base model
- **Important naming note:** the artifact identifier contains `distilgpt2`, but the published artifact metadata identifies GPT-2 rather than DistilGPT-2. The runtime loads the artifact by identifier and does not hard-code an architecture class.

The model and tokenizer are loaded with `from_pretrained`. Cog loads them from the Hugging Face Hub; the FastAPI service loads them from a local directory. No model weights are committed to this repository.

### Generation configuration

Generation differs by entry point:

| Entry point | Generation settings |
| --- | --- |
| Cog predictor | `max_length=512`, sampling enabled, `top_k=50`, `top_p=0.95` |
| FastAPI service | `max_new_tokens=300`, `num_beams=3`, `no_repeat_ngram_size=2`, early stopping enabled |
| Local script | `max_new_tokens=250`, `num_beams=3`, `no_repeat_ngram_size=2` |

The Cog predictor selects CUDA when `torch.cuda.is_available()` is true and otherwise uses CPU. The FastAPI service explicitly loads the local model with `device_map="cpu"`.

## Inference Flow

```text
job title + skills
        ↓
prompt text
        ↓
tokenizer
        ↓
causal language-model generation
        ↓
decoded text response
```

For the FastAPI endpoint, the prompt is:

```text
Job Title: <job_title>
Skills: <skills>
Questions Only. No Answers:
```

The tokenizer produces PyTorch tensors, `model.generate(...)` produces token IDs, and the tokenizer decodes the entire generated sequence with special tokens skipped. As implemented, the returned text includes the input prompt as well as its continuation; neither the API nor Cog extracts or validates a structured list of questions.

## Serving Options

### Local inference

[`local_inference.py`](local_inference.py) is a standalone CPU script. It includes a sample Cybersecurity Engineer prompt, measures elapsed time, and prints the decoded text. Before running it, change its hard-coded `local_model_path` to a valid local model directory; the committed path is Windows-specific and is not present in this repository.

```bash
python local_inference.py
```

### Cog

[`predict.py`](predict.py) and [`cog.yaml`](cog.yaml) package a Cog predictor. Its `setup()` downloads the Hugging Face model artifact and its `predict()` method accepts one `prompt` string.

```bash
cog predict -i prompt="Job Title: Backend Developer
Skills: Python, FastAPI, Docker
Questions Only. No Answers:"
```

`cog.yaml` declares a Python 3.11 build, enables GPU support, and names the Replicate image `replicate/zeeshan506/echohire-qgen-distilgpt2`.

### FastAPI service

[`serve.py`](serve.py) defines a FastAPI application with `GET /generate-questions`. It expects `job_title` and `skills`, plus optional `max_new_tokens` and `num_beams` query parameters. It responds with JSON in this form:

```json
{"questions": "<decoded prompt and generated continuation>"}
```

The service requires a local model artifact at `./models/echohire-qgen-distilgpt2`; `snapshot.py` downloads the artifact into the Hugging Face cache by default, so it does **not** populate that directory automatically. With that directory already available, start the service with:

```bash
uvicorn serve:app --host 0.0.0.0 --port 9000
```

Then request it with:

```bash
curl --get 'http://127.0.0.1:9000/generate-questions' \
  --data-urlencode 'job_title=Backend Developer' \
  --data-urlencode 'skills=Python, FastAPI, Docker'
```

### Docker

The [`Dockerfile`](Dockerfile) builds a Python 3.13 slim image, installs CPU-only PyTorch, copies `serve.py`, exposes port 9000, and launches Uvicorn. It does not copy or download the local model artifact, so a model directory must be mounted at runtime for the service to start.

```bash
docker build -t echohire-model-service .
docker run --rm -p 9000:9000 \
  -v "$(pwd)/models/echohire-qgen-distilgpt2:/app/models/echohire-qgen-distilgpt2:ro" \
  echohire-model-service
```

## Model Artifact

[`snapshot.py`](snapshot.py) calls `huggingface_hub.snapshot_download` for `Zeeshan506/echohire-qgen-distilgpt2` and prints the resulting local cache path. The application therefore relies on a model registry artifact instead of checked-in weights.

## Training Data

[`data.jsonl`](data.jsonl) is line-delimited JSON intended for prompt/completion-style question-generation examples. Complete records contain:

- `prompt`: a `Job Title`, `Skills`, and `Questions:` instruction
- `completion`: a numbered set of interview questions

The file covers a variety of job titles and skills. It is not referenced by any executable training or fine-tuning script in this repository, so this repository does not establish how or whether it was used to train the published artifact. One record is malformed: it has a `prompt` but no `completion`; consumers should validate records before using the file as training data. Dataset provenance is not documented in the repository.

## Engineering Highlights

- An isolated inference component for EchoHire's question-generation task.
- Hugging Face Transformers and Hub integration for model and tokenizer loading.
- A Cog predictor for portable packaged inference.
- A FastAPI endpoint with configurable generation length and beam count.
- CPU/GPU selection where implemented: automatic CUDA selection in Cog and CPU-only FastAPI loading.
- Docker packaging for the FastAPI process, with model-artifact mounting required at runtime.

## Technology Stack

- **Language and runtime:** Python 3.11 for Cog; Python 3.13 in the Dockerfile and project metadata.
- **Machine learning:** PyTorch, Hugging Face Transformers, Hugging Face Hub, Accelerate, hf-xet.
- **Serving:** Cog, FastAPI, Uvicorn, Pydantic, python-dotenv.
- **Packaging:** Docker.

## Repository Role

This repository contains EchoHire's custom model/inference layer. Related components live separately:

- frontend
- backend
- database

## Related Repositories

- Frontend: <https://github.com/Zeeshan506/Echohire-Frontend>
- Backend: <https://github.com/Zeeshan506/Echohire-Backend>
- Database: <https://github.com/Zeeshan506/echohire-database>

## Repository Structure

```text
.
├── predict.py             # Cog predictor; loads the Hub artifact
├── serve.py               # FastAPI question-generation endpoint
├── local_inference.py     # Local, path-based inference example
├── snapshot.py            # Downloads the Hub artifact into the HF cache
├── data.jsonl             # Prompt/completion question-generation examples
├── cog.yaml               # Cog build and predictor configuration
├── Dockerfile             # FastAPI container image definition
├── requirements.txt       # Runtime dependencies
├── pyproject.toml         # Python project metadata and dependencies
└── main.py                # Minimal project entry point
```

## Environment Configuration

`serve.py` loads a `.env` file and reads `API_URL` to add an allowed CORS origin. `HF_API_TOKEN` appears only in a commented-out Cog example; it is not actively wired into the predictor. The Dockerfile also sets `PYTHONUNBUFFERED`.

## Current Status

Implemented artifacts include the Hugging Face-backed Cog predictor, a local inference example, a FastAPI question-generation route, a Hub snapshot helper, and a Dockerfile for the API process. The FastAPI/Docker path depends on a separately provisioned local model directory, and this repository does not include code that materializes the Hub download at the path required by `serve.py`.

## Limitations

- The service returns raw decoded generation rather than a parsed or validated question list.
- The `Questions Only. No Answers:` instruction guides generation but is not enforced programmatically.
- The FastAPI endpoint exposes generation controls without explicit bounds, authentication, or rate limiting in this repository.
- The local inference script contains a machine-specific Windows model path.
- The Docker image does not include the model artifact and will not start successfully without a compatible mounted model directory.
- No evaluation results are present in this repository; none are claimed here.

## Maintenance Note

`pyproject.toml` currently retains placeholder package metadata: `description = "Add your description here"`. It was intentionally left unchanged during this README-focused update and should be addressed in a later repository-cleanup pass.
