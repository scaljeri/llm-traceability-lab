# Developing the tool

Codex can help change the app; ChatGPT can help design experiments. Describe the
behavior you want, include an example, and ask for a focused change with a test.

The Python API lives in `src/traceability_lab/main.py`, the interface in
`src/traceability_lab/web/`,
and evaluation in `fact_answers.py` and `semantic_evaluation.py`. Example
experiments live in `experiments/`; their JSON format is documented
[here](docs/import-export-types.md).

Check changes with:

```sh
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
node --test tests/*.test.mjs
```

Run the integrated browser suite with:

```sh
uv run --with playwright playwright install chromium
uv run --with playwright pytest --browser tests/test_browser.py
```

It starts disposable local servers and makes no real model calls. Installed Chrome
is used on macOS; other platforms use Playwright Chromium. Set `BROWSER_EXECUTABLE`
only when a different browser executable is needed. The `browser_*.py` checks
are included, with synthetic model responses for report and reviewer views.
They can also be run individually using the arguments in their docstrings.

Historical storage tests use explicit adapters in `tests/legacy_api.py`. Their
removed library/retrieval/annotation routes are not exposed by the application.

Tests use disposable databases and mock models. Live model tests are opt-in;
passing software tests does not establish model accuracy. Preserve historical
results, exact quotations and the separation between generator inputs and expected
answers. Do not commit local databases, private documents or credentials.

Contributions use the project's GPL-3.0-only license.
