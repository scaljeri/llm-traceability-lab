"""Inspect average run labels and per-fact colors on a disposable app copy."""

import math
import re
import sys

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def main(address, batch_id):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_executable(), headless=True)
        page = browser.new_page()
        batch = page.request.get(f"{address}/api/batches/{batch_id}").json()
        page.route(
            "**/api/executions",
            lambda route: route.fulfill(
                json={
                    "status": "completed",
                    "summary": batch,
                    "batch_id": batch_id,
                    "total": len(batch["runs"]),
                    "completed": len(batch["runs"]),
                }
            ),
        )
        page.goto(address)
        identity = page.locator("#saved-experiment option").nth(1).get_attribute("value")
        page.locator("#saved-experiment").select_option(identity)
        page.locator("#run-button").click()
        panel = page.locator('.property-responses[data-property="facts"]')
        blocks = panel.locator("..").locator(".correctness-run")
        expect(blocks).to_have_count(len(batch["runs"]))
        for index, evaluation in enumerate(batch["evaluations"]):
            scores = evaluation["automatic"]["semantic_evaluation"]["judgments"]["facts"]
            block = blocks.nth(index)
            expect(block.locator(".correctness-segment")).to_have_count(len(scores))
            measured = [s["correctness_score"] for s in scores if s.get("correctness_score") is not None]
            text = str(math.floor(sum(measured) / len(measured) + 0.5)) if measured else "—"
            expect(block.locator(".correctness-summary")).to_have_text(text)
            expect(block.locator(".correctness-segment")).to_have_text([""] * len(scores))
            for segment_index, score in enumerate(scores):
                segment = block.locator(".correctness-segment").nth(segment_index)
                expect(segment).to_have_attribute("title", re.compile("Correctness:"))
                citations = [
                    check
                    for criterion in evaluation["automatic"].get("criteria", [])
                    for check in criterion.get("citation_validations", [])
                    if check.get("fact_index") == score["fact_index"]
                ]
                citation_failed = any(check.get("status") == "fail" for check in citations)
                if citation_failed:
                    assert segment.evaluate("el => getComputedStyle(el).boxShadow") == "none"
                if score.get("correctness_score") == 0 or citation_failed:
                    assert (
                        segment.evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(207, 75, 75)"
                    )
                elif score.get("correctness_score") == 100:
                    assert segment.evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(25, 97, 86)"
        details = panel.locator(".property-run-details")
        assert details.get_attribute("open") is None
        details.locator(":scope > summary").click()
        expect(panel.locator(".reviewer-scores").first).to_be_visible()
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
