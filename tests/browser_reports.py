"""Check report-only run selection and exports against a seeded isolated batch.

Usage: python tests/browser_reports.py BASE_URL BATCH_ID
Execution start is intercepted: no inference or new runs are started.
"""

import json
import re
import sys

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def test_reports(base_url, batch_id):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=browser_executable(),
            headless=True,
        )
        page = browser.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator, 'clipboard', {value: {writeText: async text => {window.copied = text;}}});"
        )
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = page.request.get(f"{base_url}/api/batches/{batch_id}")
        assert response.ok, response.text()
        batch = response.json()
        page.route(
            "**/api/executions",
            lambda route: route.fulfill(
                content_type="application/json",
                body=json.dumps(
                    {
                        "status": "completed",
                        "summary": batch,
                        "batch_id": batch_id,
                        "total": batch["counts"]["total"],
                        "completed": batch["counts"]["total"],
                        "title": batch.get("experiment_title", "Reports"),
                    }
                ),
            ),
        )
        page.goto(base_url)
        expect(page.locator("#saved-experiment option")).not_to_have_count(1)
        identity = page.locator("#saved-experiment option").nth(1).get_attribute("value")
        page.locator("#saved-experiment").select_option(identity)
        expect(page.locator("#run-button")).to_be_enabled()
        page.locator("#run-button").click()
        if any(run.get("json_repair_status") == "repaired" for run in batch["runs"]):
            expect(page.locator("#property-chart")).to_contain_text("JSON repaired")
        judgments = (
            batch["evaluations"][0].get("automatic", {}).get("semantic_evaluation", {}).get("judgments") or {}
        )
        for property_id in ("facts", "answer"):
            values = (
                judgments.get("facts", [])
                if property_id == "facts"
                else [judgments["answer"]]
                if judgments.get("answer")
                else []
            )
            if not values:
                continue
            segments = (
                page.locator(f'.property-responses[data-property="{property_id}"]')
                .locator("..")
                .locator(".correctness-segment")
            )
            expect(segments).not_to_have_count(0)
            for index, value in enumerate(values):
                score = value.get("correctness_score")
                segment = segments.nth(index)
                expect(segment).to_have_attribute("title", re.compile("Correctness:.*\\nConfidence:"))
                expect(segment).to_have_text("")
                if score == 0:
                    assert (
                        segment.evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(207, 75, 75)"
                    )
            if batch["counts"]["total"] <= 5:
                measured = [
                    item["correctness_score"] for item in values if item.get("correctness_score") is not None
                ]
                mean = str(int(sum(measured) / len(measured) + 0.5)) if measured else "—"
                expect(segments.first.locator("..").locator(".correctness-summary")).to_have_text(mean)
            panel = page.locator(f'.property-responses[data-property="{property_id}"]')
            panel.locator(".property-run-details > summary").click()
            table = panel.locator(".reviewer-scores").first
            expect(table).to_be_visible()
            expect(table.locator("th")).to_have_text(
                [
                    "Statement",
                    "Correctness",
                    "Confidence",
                    "Semantic meaning",
                    "Source support",
                    "Citation validation",
                ]
            )
            for index, judgment in enumerate(sorted(values, key=lambda item: item.get("fact_index", 0))):
                cells = table.locator("tbody tr").nth(index).locator("td")
                for position, key in ((1, "correctness_score"), (2, "confidence")):
                    value = judgment.get(key)
                    label = "Unavailable" if value is None else f"{value:g}/100"
                    expect(cells.nth(position)).to_have_text(label)
        checks = next(
            (
                item.get("citation_validations", [])
                for item in batch["evaluations"][0].get("automatic", {}).get("criteria", [])
                if item["id"] == "source_ids"
            ),
            [],
        )
        for check in checks:
            if check["status"] == "fail":
                panel = page.locator('.property-responses[data-property="facts"]')
                diagnostic = panel.locator(".citation-diagnostic.citation-failed")
                expect(diagnostic).not_to_have_count(0)
                for error in check["errors"]:
                    expect(diagnostic.first).to_contain_text(error["code"])
                assert panel.locator(".reviewer-scores tr.citation-failed").count() > 0
                segment = panel.locator("..").locator(".correctness-segment.citation-invalid-segment")
                expect(segment).not_to_have_count(0)
                assert (
                    segment.first.evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(207, 75, 75)"
                )
        reports = page.locator("#run-reports")
        expect(reports).to_be_visible()
        assert reports.get_attribute("open") is None
        assert page.locator("#inspect-batch-run").count() == 0
        reports.locator("summary").first.click()
        expect(page.locator("#copy-markdown")).to_be_enabled()
        expected_id = batch["runs"][0]["id"]
        expect(page.locator("#download")).to_have_attribute("href", f"/api/runs/{expected_id}/report")
        page.locator("#copy-markdown").click()
        expect(page.locator("#snackbar")).to_have_text("Run Markdown copied to clipboard")
        markdown = page.evaluate("window.copied")
        assert expected_id in markdown and "Exact recorded request" in markdown
        report = page.request.get(f"{base_url}/api/runs/{expected_id}/report").json()
        for identity in report.get("models", {}).values():
            if identity:
                assert identity["display_name"] in markdown
                assert f"Model:** {identity['technical'].get('api_model_id')}" not in markdown
        if report.get("reproducibility"):
            assert report["reproducibility"]["configuration_sha256"] in markdown
        for check in checks:
            for error in check["errors"]:
                assert error["code"] in markdown
        with page.expect_download() as event:
            page.locator("#download-markdown").click()
        download = event.value
        assert download.suggested_filename == f"run-{expected_id}.md"
        assert open(download.path()).read() == markdown
        page.locator("#copy-provenance").click()
        expect(page.locator("#snackbar")).to_have_text("Provenance JSON copied to clipboard")
        provenance = json.loads(page.evaluate("window.copied"))
        assert provenance == page.request.get(f"{base_url}/api/runs/{expected_id}/provenance").json()
        assert page.locator("#provenance-json").is_hidden()
        if len(batch["runs"]) > 1:
            expected_id = batch["runs"][1]["id"]
            page.locator("#batch-run-selection").select_option(expected_id)
            expect(page.locator("#download")).to_have_attribute("href", f"/api/runs/{expected_id}/report")
            expect(page.locator("#copy-markdown")).to_be_enabled()
        with page.expect_download() as event:
            page.locator("#download-batch-markdown").click()
        assert event.value.suggested_filename == "experiment-analysis.md"
        assert "Raw responses per run" in open(event.value.path()).read()
        for width in (1280, 390):
            page.set_viewport_size({"width": width, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), page.evaluate(
                "Array.from(document.querySelectorAll('body *')).filter(el=>el.getBoundingClientRect().right>innerWidth+1).map(el=>[el.tagName,el.id,el.className,el.getBoundingClientRect().right]).slice(0,12)"
            )
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    test_reports(sys.argv[1], sys.argv[2])
