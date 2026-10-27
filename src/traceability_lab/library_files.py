"""Explicit replacement of editable libraries from JSON files, or vice versa."""

import json
import re
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

from pydantic import ValidationError

from traceability_lab.db import json_text, now
from traceability_lab.defined_experiments import list_definitions
from traceability_lab.initial_experiments import read_initial_experiments
from traceability_lab.prompts import PromptInput, list_prompts, read_prompt_files
from traceability_lab.transfers import export_experiment


def check_library_directories(settings):
    if settings.experiments_directory.resolve() == settings.prompts_directory.resolve():
        raise ValueError("Experiments and prompts must use different directories.")


def import_library_files(db, settings):
    check_library_directories(settings)
    for directory in (settings.experiments_directory, settings.prompts_directory):
        if not directory.is_dir():
            raise ValueError(f"Source directory does not exist: {directory.name}")
    # Validate both complete libraries before replacing either one.
    experiments = read_initial_experiments(settings.experiments_directory)
    try:
        prompts = read_prompt_files(settings.prompts_directory)
    except (AttributeError, TypeError) as error:
        raise ValueError("Prompt files must contain valid prompt objects.") from error
    with db.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM experiment_definitions")
        connection.execute("DELETE FROM prompts")
        for definition in experiments:
            timestamp = now()
            connection.execute(
                "INSERT INTO experiment_definitions VALUES (?,?,?,?,?)",
                ("saved-" + uuid4().hex, 1, json_text(definition.model_dump()), timestamp, timestamp),
            )
        for prompt in prompts:
            connection.execute("INSERT INTO prompts VALUES (?,?)", (prompt["id"], json_text(prompt)))
        connection.execute("INSERT OR REPLACE INTO prompt_library VALUES (1,1)")
    return {"experiments": len(experiments), "prompts": len(prompts)}


def export_library_files(db, settings):
    check_library_directories(settings)
    experiments, prompts = {}, {}
    values = []
    for value in list_prompts(db):
        try:
            values.append(PromptInput.model_validate(value).portable())
        except ValidationError as error:
            reason = "; ".join(item["msg"].removeprefix("Value error, ") for item in error.errors())
            raise ValueError(
                f'Cannot export prompt "{value.get("name", "Unnamed")}" ({value.get("id", "unknown")}): '
                f"{reason} Fix it in Manage prompts; no files were changed."
            ) from error
    for item in list_definitions(db):
        stem = (
            re.sub(r"[^a-z0-9]+", "-", item["definition"]["title"].lower()).strip("-")[:100] or "experiment"
        )
        filename, suffix = stem + ".json", 2
        while filename in experiments:
            filename = f"{stem}-{suffix}.json"
            suffix += 1
        experiments[filename] = export_experiment(db, item["id"])
    for value in values:
        category = [item for item in values if item["type"] == value["type"]]
        if value["type"] == "answer":
            filename = (
                "generation_prompt.json" if len(category) == 1 else f"generation_prompt-{value['id']}.json"
            )
            prompts[filename] = value
        else:
            for stage, field in (("facts", "fact_prompt"), ("answer", "answer_prompt")):
                stem = "fact_review_prompt" if stage == "facts" else "answer_review_prompt"
                filename = f"{stem}.json" if len(category) == 1 else f"{stem}-{value['id']}.json"
                prompts[filename] = {
                    "format": "traceability-review-stage-v1",
                    "id": value["id"],
                    "name": value["name"],
                    "stage": stage,
                    "system_prompt": value["system_prompt"],
                    "user_prompt": value[field],
                }
    replace_json_files({settings.experiments_directory: experiments, settings.prompts_directory: prompts})
    return {"experiments": len(experiments), "prompts": len(values)}


def replace_json_files(libraries: dict[Path, dict]):
    """Stage all writes first and restore replaced files if installation fails."""
    moved, installed = [], []
    with ExitStack() as stack:
        prepared = []
        for directory, values in libraries.items():
            if len({name.casefold() for name in values}) != len(values):
                raise ValueError(
                    "Export filenames collide. Give prompts distinct IDs, including letter case."
                )
            directory.mkdir(parents=True, exist_ok=True)
            temporary = Path(mkdtemp(prefix=".sync-", dir=directory))
            stack.callback(shutil.rmtree, temporary)
            incoming, backup = temporary / "incoming", temporary / "backup"
            incoming.mkdir()
            backup.mkdir()
            for filename, payload in values.items():
                (incoming / filename).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            prepared.append((directory, incoming, backup))
        try:
            for directory, incoming, backup in prepared:
                for original in directory.glob("*.json"):
                    target = backup / original.name
                    original.replace(target)
                    moved.append((target, original))
                for source in incoming.iterdir():
                    target = directory / source.name
                    source.replace(target)
                    installed.append(target)
        except OSError:
            try:
                for target in reversed(installed):
                    target.unlink()
                for backup, original in reversed(moved):
                    backup.replace(original)
            except OSError as error:
                stack.pop_all()
                locations = ", ".join(str(backup) for _, _, backup in prepared)
                raise OSError(
                    f"Could not restore every file. Recovery copies remain in: {locations}"
                ) from error
            raise
