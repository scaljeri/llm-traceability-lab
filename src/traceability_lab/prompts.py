"""Editable prompt sets, separate from experiment data and frozen in each run."""

import json
import re
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from traceability_lab.db import Database, json_text

SLOTS = re.compile(r"\{\{([^{}]+)\}\}")


class PromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["traceability-prompt-v1"] = "traceability-prompt-v1"
    id: str = Field(default_factory=lambda: "prompt-" + uuid4().hex, pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    type: Literal["answer", "review"]
    name: str = Field(min_length=1, max_length=200)
    system_prompt: str = Field(default="", max_length=20_000)
    user_prompt: str = Field(default="", max_length=10_000)
    fact_prompt: str = Field(default="", max_length=30_000)
    answer_prompt: str = Field(default="", max_length=30_000)

    @model_validator(mode="after")
    def valid_templates(self):
        if not self.name.strip():
            raise ValueError("Prompt name cannot be blank.")
        if set(SLOTS.findall(self.system_prompt)) - {"output_json"}:
            raise ValueError(
                "System prompts only support {{output_json}}; put input placeholders in user prompts."
            )
        if self.system_prompt.count("{{output_json}}") > 1:
            raise ValueError("Use {{output_json}} at most once in the system prompt.")
        if self.type == "answer":
            if self.fact_prompt or self.answer_prompt:
                raise ValueError("Answer prompts use user_prompt, not review fields.")
            fields = [self.user_prompt]
            allowed = {"user_question", "documents", "output_json"}
            if not fields[0].strip():
                raise ValueError("User prompt cannot be blank.")
        else:
            if self.user_prompt:
                raise ValueError("Review prompts use fact_prompt and answer_prompt.")
            fields = [self.fact_prompt, self.answer_prompt]
            allowed = {"review_input"}
            if any(field.count("{{review_input}}") != 1 for field in fields):
                raise ValueError("Both review prompts must include {{review_input}} exactly once.")
        for field in fields:
            if set(SLOTS.findall(field)) - allowed:
                raise ValueError("Unknown prompt placeholder.")
        if self.type == "answer" and self.user_prompt.count("{{documents}}") != 1:
            raise ValueError("Generation prompts must include {{documents}} exactly once.")
        return self

    def portable(self):
        excluded = {"fact_prompt", "answer_prompt"} if self.type == "answer" else {"user_prompt"}
        return self.model_dump(exclude=excluded)


def list_prompts(db: Database) -> list[dict]:
    return [
        json.loads(row["payload"])
        for row in db.rows(
            "SELECT payload FROM prompts ORDER BY json_extract(payload,'$.name') COLLATE NOCASE,id"
        )
    ]


def read_prompt(db: Database, identity: str, kind: str) -> dict:
    row = db.one("SELECT payload FROM prompts WHERE id=?", (identity,))
    try:
        value = PromptInput.model_validate_json(row["payload"]).portable()
    except ValidationError as error:
        stored = json.loads(row["payload"])
        name = stored.get("name", "Unnamed") if isinstance(stored, dict) else "Unnamed"
        reason = "; ".join(item["msg"].removeprefix("Value error, ") for item in error.errors())
        raise ValueError(
            f'Cannot use prompt "{name}" ({identity}): {reason} Fix it in Manage prompts.'
        ) from error
    if value["type"] != kind:
        raise ValueError(f"Select a {kind} prompt.")
    return value


def save_prompt(db: Database, data: PromptInput, identity: str | None = None) -> dict:
    if identity and identity != data.id:
        raise ValueError("A saved prompt's ID cannot change.")
    value = data.portable()
    with db.connect() as connection:
        if identity:
            cursor = connection.execute(
                "UPDATE prompts SET payload=? WHERE id=?", (json_text(value), identity)
            )
            if not cursor.rowcount:
                raise LookupError("Prompt not found.")
        else:
            connection.execute("INSERT INTO prompts VALUES (?,?)", (data.id, json_text(value)))
    return value


def delete_prompt(db: Database, identity: str) -> dict:
    with db.connect() as connection:
        if not connection.execute("DELETE FROM prompts WHERE id=?", (identity,)).rowcount:
            raise LookupError("Prompt not found.")
    return {"id": identity, "deleted": True}


class ReviewStageInput(BaseModel):
    """Two source files compose one editable review set with a shared system."""

    model_config = ConfigDict(extra="forbid")
    format: Literal["traceability-review-stage-v1"]
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    name: str = Field(min_length=1, max_length=200)
    stage: Literal["facts", "answer"]
    system_prompt: str = Field(max_length=20_000)
    user_prompt: str = Field(max_length=30_000)


def read_prompt_files(directory) -> list[dict]:
    values, reviews = [], {}
    for path in sorted(directory.glob("*.json")):
        value = json.loads(path.read_text())
        if value.get("format") == "traceability-review-stage-v1":
            stage = ReviewStageInput.model_validate(value)
            group = reviews.setdefault(stage.id, {})
            if stage.stage in group:
                raise ValueError(f"Duplicate {stage.stage} stage for review prompt {stage.id}.")
            group[stage.stage] = stage
        else:
            values.append(PromptInput.model_validate(value).portable())
    for identity, stages in reviews.items():
        if set(stages) != {"facts", "answer"}:
            raise ValueError(f"Review prompt {identity} needs both facts and answer source files.")
        fact, answer = stages["facts"], stages["answer"]
        if fact.name != answer.name or fact.system_prompt != answer.system_prompt:
            raise ValueError(f"Review prompt {identity} must share its name and system prompt.")
        values.append(
            PromptInput(
                id=identity,
                type="review",
                name=fact.name,
                system_prompt=fact.system_prompt,
                fact_prompt=fact.user_prompt,
                answer_prompt=answer.user_prompt,
            ).portable()
        )
    if len({value["id"] for value in values}) != len(values):
        raise ValueError("Prompt IDs must be unique across answer and review sets.")
    return values


def seed_prompts(db: Database, directory) -> None:
    if db.rows("SELECT initialized FROM prompt_library WHERE id=1"):
        return
    values = read_prompt_files(directory)
    if not values:
        return
    with db.connect() as connection:
        for value in values:
            connection.execute("INSERT OR IGNORE INTO prompts VALUES (?,?)", (value["id"], json_text(value)))
        connection.execute("INSERT INTO prompt_library VALUES (1,1)")


def apply_answer_prompt(db: Database, question):
    if not question.answer_prompt_id:
        return question
    value = read_prompt(db, question.answer_prompt_id, "answer")
    return question.model_copy(
        update={"system_prompt": value["system_prompt"], "prompt_instructions": value["user_prompt"]}
    )
