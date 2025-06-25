---
title: Feel
emoji: 🚀
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.10.0
app_file: app/app.py
pinned: false
---

# Feel

Platform developed at MIT in collaboration with the HuggingFace platform and with inputs from IBM, Cohere, KAIST, and other independent collaborators. Aimed at improving performance of existing open source Large Language Models through real-time human feedback loop.

This repository hosts the development of a novel automated RLHF platform where the community can provide feedback which is then immediately integrated and used to improve the models in real time. The feedback is automatically integrated into an RLHF pipeline to continuously fine-tune and improve the models.

## What is Feel?

A community-driven project to improve multilingual capabilities in Large Language Models (LLMs). Leverages feedback from users and automated RLHF pipelines to continuously improve model performance.

## Why FeeL?

FeeL = Feedback Loop. It is a platform that enables the community to provide real-time feedback on language models. The feedback is automatically integrated into an RLHF pipeline to continuously fine-tune and improve the models.

## Repository Structure

The repository is organized as follows:

```
ml/                # Directory for machine learning code
├── README.md      # Dataset schema and project structure
├── data/          # Directory for dataset files
├── models/        # Directory for model files
app/               # Directory for application code
├── app.py         # Main application file
```

## Installation

The repository uses `uv` for managing virtual environments. To install `uv`, go [here](https://docs.astral.sh/uv/getting-started/installation/). Create a virtual environment.

```bash
uv venv --python 3.11
```

Activate the virtual environment

```bash
source .venv/bin/activate
```

Sync the dependencies

```bash
uv sync --all-groups
```

To install the required dependencies, run the following commands:

### ML Dependencies

```bash
uv sync --group ml
```

### App Dependencies

```bash
uv sync --group app
```

## Sync with Hugging Face Space

First, add the remote
```
git remote add hf https://huggingface.co/spaces/feel-fl/open-human-feedback-chat
```

Push to the remote branch

```
git push hf main
```
