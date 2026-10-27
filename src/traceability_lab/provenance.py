"""Een PROV-JSON-export opbouwen uit de primaire, onveranderlijke records.

Geen tweede provenance-database: dezelfde bronrelaties worden bij export naar
W3C PROV vertaald. 'used' beschrijft aangeboden input, niet intern modeldenken.
"""

import hashlib
import json
from datetime import UTC, datetime

from prov.model import PROV, ProvDocument

from traceability_lab.db import Database, json_text
from traceability_lab.documents import digest, version_chunks
from traceability_lab.pipeline import read_run, read_trace


def export_provenance(db: Database, run_id: str) -> dict:
    run = read_run(db, run_id)
    document = ProvDocument()
    document.add_namespace("lab", "urn:ai-traceability-lab:")

    def entity(kind, identity, attributes=None):
        return document.entity(f"lab:{kind}-{identity}", attributes or {})

    software = document.agent(
        "lab:software-v0.1.0", {"prov:type": PROV["SoftwareAgent"], "lab:backend": run["backend"]}
    )
    activity = document.activity(
        f"lab:run-{run_id}",
        run["created_at"],
        run["finished_at"],
        {"lab:trace_id": run["trace_id"], "lab:status": run["status"]},
    )
    document.wasAssociatedWith(activity, software)
    question = entity("question", run_id, {"prov:value": run["question"]})
    document.used(activity, question)

    # Activiteiten krijgen alleen tijden die daadwerkelijk als span gemeten zijn.
    steps = {}
    for span in read_trace(db, run["trace_id"]):
        if span["name"] in {"vector.retrieve", "context.select", "prompt.build", "llm.generate"}:
            step = document.activity(
                f"lab:step-{span['span_id']}",
                datetime.fromtimestamp(span["start_ns"] / 1e9, UTC),
                datetime.fromtimestamp(span["end_ns"] / 1e9, UTC),
                {"prov:label": span["name"], "lab:span_id": span["span_id"]},
            )
            document.wasAssociatedWith(step, software)
            steps[span["name"]] = step

    # Alle toegestane bronversies opnemen, inclusief bewijs dat pas later bekeken is.
    versions = [row["id"] for row in run["snapshot"]["corpus"]]
    # Include ancestors as provenance entities, but never as supplied context.
    ancestry = list(versions)
    for identity in ancestry:
        parent = db.one("SELECT previous_version_id FROM source_versions WHERE id=?", (identity,))[
            "previous_version_id"
        ]
        if parent and parent not in ancestry:
            ancestry.append(parent)
    for version_id in ancestry:
        version = db.one("SELECT * FROM source_versions WHERE id=?", (version_id,))
        logical = entity("document", version["document_id"])
        source = entity(
            "version",
            version_id,
            {
                "lab:sha256": version["sha256"],
                "lab:version": version["version"],
                "lab:source": version["source"],
                "lab:sequence_number": version["sequence_number"],
                "lab:published_on": version["published_on"],
                "lab:effective_on": version["effective_on"],
            },
        )
        document.specializationOf(source, logical)
        if version["previous_version_id"]:
            document.wasRevisionOf(source, f"lab:version-{version['previous_version_id']}")
    for chunk in version_chunks(db, versions):
        node = entity(
            "chunk",
            chunk["id"],
            {
                "lab:sha256": chunk["sha256"],
                "lab:start": chunk["start"],
                "lab:end": chunk["end"],
                "prov:value": chunk["text"],
            },
        )
        document.wasDerivedFrom(node, f"lab:version-{chunk['version_id']}")
        if run["snapshot"].get("embedding_key") and "vector.retrieve" in steps:
            rows = db.rows(
                "SELECT * FROM embeddings WHERE chunk_id=? AND model_key=?",
                (chunk["id"], run["snapshot"]["embedding_key"]),
            )
            if rows:
                embedding = entity(
                    "embedding",
                    digest(chunk["id"] + rows[0]["model_key"]),
                    {
                        "lab:model_key": rows[0]["model_key"],
                        "lab:dimensions": rows[0]["dimensions"],
                        "lab:sha256": hashlib.sha256(rows[0]["vector"]).hexdigest(),
                    },
                )
                document.wasDerivedFrom(embedding, node)
                document.used(steps["vector.retrieve"], embedding)

    context = entity("context", run_id)
    if "vector.retrieve" in steps:
        query_vector = entity(
            "query-vector", run_id, {"prov:value": json_text(run["snapshot"]["query_vector"])}
        )
        document.wasDerivedFrom(query_vector, question)
        document.used(steps["vector.retrieve"], query_vector)
        if run["retrievals"]:
            document.wasGeneratedBy(context, steps["vector.retrieve"])
    if "context.select" in steps and run["context"]:
        document.wasGeneratedBy(context, steps["context.select"])
        for result in run["context"]:
            document.used(steps["context.select"], f"lab:chunk-{result['chunk_id']}")
    for result in run["context"]:
        document.wasDerivedFrom(
            context,
            f"lab:chunk-{result['chunk_id']}",
            other_attributes={
                "lab:rank": result["rank"],
                "lab:origin": result["origin"],
                **(
                    {"lab:distance": result["distance"], "lab:metric": result["metric"]}
                    if result.get("distance") is not None
                    else {}
                ),
            },
        )
    if run["context"]:
        document.used(activity, context)
    if run["messages"]:
        selected = (run["snapshot"].get("experiment") or {}).get("prompts", {}).get("answer") or {}
        prompt = entity(
            "prompt",
            run_id,
            {
                "prov:value": json_text(run["messages"]),
                "lab:template": run["snapshot"]["prompt_version"],
                **{f"lab:prompt_{key}": selected[key] for key in ("id", "name", "sha256") if key in selected},
            },
        )
        for index, item in enumerate(run["snapshot"].get("supplemental_contexts", []), 1):
            supplemental = entity(
                "operator-context",
                f"{run_id}-{index}",
                {
                    "prov:value": item["content"],
                    "prov:label": item["title"],
                    "lab:context_id": item["id"],
                    "lab:sha256": item["sha256"],
                    "lab:position": index,
                },
            )
            document.wasDerivedFrom(prompt, supplemental)
        document.wasDerivedFrom(prompt, context)
        document.used(activity, prompt)
        if "prompt.build" in steps:
            document.used(steps["prompt.build"], context)
            document.used(steps["prompt.build"], question)
            document.wasGeneratedBy(prompt, steps["prompt.build"])
        if "llm.generate" in steps:
            document.used(steps["llm.generate"], prompt)
            model_configuration = entity(
                "model-configuration", run_id, {"prov:value": json_text(run["snapshot"]["model"])}
            )
            document.used(steps["llm.generate"], model_configuration)
    if run["answer"] is not None:
        answer = entity("answer", run_id, {"prov:value": run["answer"]})
        document.wasGeneratedBy(answer, steps.get("llm.generate", activity))

    for evaluation in db.rows("SELECT * FROM run_evaluations WHERE run_id=?", (run_id,)):
        result = json.loads(evaluation["payload"])
        check = document.activity(
            f"lab:evaluation-activity-{run_id}",
            result.get("evaluation_started_at"),
            result.get("evaluation_finished_at"),
            other_attributes={"lab:recorded_at": evaluation["created_at"]},
        )
        document.wasAssociatedWith(check, software)
        if run["answer"] is not None:
            document.used(check, answer)
        assessment = entity(
            "evaluation",
            run_id,
            {
                "lab:sha256": digest(evaluation["payload"]),
                "lab:details_uri": f"/api/runs/{run_id}/evaluation",
                "lab:criteria": json_text(
                    [
                        {key: criterion[key] for key in ("id", "result") if key in criterion}
                        for criterion in result.get("criteria", [])
                    ]
                ),
                "lab:semantic_status": (result.get("semantic_evaluation") or {}).get("status"),
            },
        )
        document.wasGeneratedBy(assessment, check)
        semantic = result.get("semantic_evaluation") or {}
        calls = semantic.get("calls") or []
        if not calls:
            continue
        reviewer = document.agent(
            f"lab:reviewer-{run_id}",
            {
                "prov:type": PROV["SoftwareAgent"],
                "prov:label": "Reviewer",
                "lab:model": json_text(semantic.get("model") or {}),
            },
        )
        configuration = entity(
            "review-configuration",
            run_id,
            {
                "prov:value": json_text(semantic.get("configuration") or {}),
                "lab:prompt_version": semantic.get("prompt_version"),
            },
        )
        for index, call in enumerate(calls):
            identity = f"{run_id}-{index}"
            review = document.activity(
                f"lab:review-{identity}",
                call.get("started_at"),
                call.get("finished_at"),
                {
                    "lab:phase": call.get("phase"),
                    "lab:fact_index": call.get("fact_index"),
                    "lab:status": call.get("status"),
                    "lab:span_id": call.get("span_id"),
                },
            )
            document.wasAssociatedWith(review, reviewer)
            document.used(review, configuration)
            if call.get("request_body") is not None:
                template = semantic.get("selected_prompts", {}).get(call.get("phase")) or {}
                request = entity(
                    "review-request",
                    identity,
                    {
                        "prov:value": json_text(call["request_body"]),
                        **{
                            f"lab:prompt_{key}": template[key]
                            for key in ("id", "name", "sha256")
                            if key in template
                        },
                    },
                )
                document.used(review, request)
                document.wasDerivedFrom(request, question)
                if run["answer"] is not None:
                    document.wasDerivedFrom(request, answer)
            observation = entity(
                "review-result",
                identity,
                {
                    "prov:value": json_text(call["response"])
                    if call.get("response") is not None
                    else call.get("raw_answer"),
                    "lab:judgment": json_text(call["parsed_answer"])
                    if call.get("parsed_answer") is not None
                    else None,
                    "lab:status": call.get("status"),
                    "lab:error": call.get("error"),
                },
            )
            document.wasGeneratedBy(observation, review)
            document.used(check, observation)

    for claim in run["claims"]:
        claim_node = entity(
            "claim",
            claim["id"],
            {"prov:value": claim["text"], "lab:start": claim["start"], "lab:end": claim["end"]},
        )
        annotation = document.activity(
            f"lab:annotation-{claim['id']}", claim["created_at"], claim["created_at"]
        )
        human = document.agent(
            "lab:annotator-" + digest(claim["annotator"]), {"prov:label": claim["annotator"]}
        )
        document.used(annotation, f"lab:answer-{run_id}")
        document.wasAssociatedWith(annotation, human)
        document.wasGeneratedBy(claim_node, annotation)
        document.wasDerivedFrom(claim_node, f"lab:answer-{run_id}")
        for assessment in claim["assessments"]:
            node = entity(
                "assessment",
                str(assessment["id"]),
                {
                    "lab:classification": assessment["classification"],
                    "lab:scope": assessment["scope"],
                    "lab:method": "manual",
                    "lab:note": assessment["note"],
                },
            )
            check = document.activity(
                f"lab:assessment-activity-{assessment['id']}",
                assessment["created_at"],
                assessment["created_at"],
            )
            assessor = document.agent(
                "lab:annotator-" + digest(assessment["annotator"]), {"prov:label": assessment["annotator"]}
            )
            document.used(check, claim_node)
            document.wasAssociatedWith(check, assessor)
            document.wasGeneratedBy(node, check)
            if assessment["chunk_id"]:
                evidence = entity(
                    "evidence",
                    str(assessment["id"]),
                    {
                        "prov:value": assessment["quote"],
                        "lab:chunk_start": assessment["start"],
                        "lab:chunk_end": assessment["end"],
                    },
                )
                document.wasDerivedFrom(evidence, f"lab:chunk-{assessment['chunk_id']}")
                document.used(check, evidence)
    return json.loads(document.serialize(format="json"))
