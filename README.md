# AI Traceability Lab

A small local tool for testing LLMs. Give a model documents and a question, repeat
it, and review its facts, quotations and answer against your expected results.

Accompanying Medium article: [**Making AI Answers Traceable and Testable**](https://calje.medium.com/making-ai-answers-traceable-and-testable-51b9800cb363?sk=1592d80e7c3deed458b132296910e279) .

Built with **OpenAI Codex**. Use Codex to adapt the app, or ChatGPT to design
experiments using the [JSON format](docs/import-export-types.md) and
[examples](experiments/). Check generated experiments before using them.

## Start

You need an existing LLM server with an **OpenAI-compatible API**.

### Docker

Install Docker with Compose. From the project folder:

```sh
docker compose up --build
```

Open **http://127.0.0.1:8000**. Stop with **Ctrl+C**.

`./data`, `./experiments` and `./prompts` are mounted from your computer.
Use your model server's LAN IP: `localhost` inside Docker refers to the container.

### Local development

For development, use [uv](https://docs.astral.sh/uv/) and Python 3.14:

```sh
uv sync --locked
uv run uvicorn traceability_lab.main:app --host 127.0.0.1 --port 8000
```

## Run an experiment

1. **Manage servers** opens on first use. Enter a name and API base URL, click
   **Load models**, then select and save a model.
2. Select an experiment. Use **Manage experiments** and **Manage prompts** to edit
   the supplied examples or create your own.
3. Choose the answer and review models, prompts, temperature and repetitions, then start.

Results appear as runs finish. Expand them to inspect the response and checks for
JSON format, facts, quotations and the final answer. Reviewers can make mistakes.
Copy or download the analysis for further investigation.

## Your data

`experiments/` and `prompts/` contain shareable JSON, loaded on first startup.
Local app data is stored in `data/lab.sqlite3` (ignored by Git). The latest 100
completed batches are retained; export results you want to keep.

**Sync files** loads JSON from disk into the app, or exports saved experiments and
prompts back to disk. Each action replaces the destination and asks for confirmation.
Server settings and runs are kept. Save edits before exporting.

For a complete reset, stop the app and empty `./data`. This also removes server settings
and run history.

[Experiment JSON format](docs/import-export-types.md) ·
[Developing the tool](CONTRIBUTING.md) · [GPL v3.0 license](LICENSE)

Copyright © 2026 AI Traceability Lab contributors. Licensed under GPL-3.0-only.
