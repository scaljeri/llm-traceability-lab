"""Isolated UI regression: reload reconnects; Cancel does not start another batch.

Run with Playwright against an isolated app. Execution endpoints are mocked;
no model calls are made.
"""

import json
import sys
from copy import deepcopy

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def test_reconnect(base_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=browser_executable(),
            headless=True,
        )
        page = browser.new_page()
        errors, starts = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        definition = {
            "title": "Reconnect UI regression",
            "question": "Who manages the project?",
            "documents": [],
            "context": "{{user_question}}\n{{documents}}\n{{output_json}}",
            "output_mode": "facts_and_answer",
            "semantic_evaluator": None,
            "semantic_reference": {"facts": [], "answer": None},
        }
        saved = page.request.post(base_url + "/api/experiment-definitions", data=definition)
        assert saved.status == 201, saved.text()
        identity = saved.json()["id"]
        seeded = page.request.post(
            base_url + "/api/experiment-definitions/run",
            data={"definition": definition, "execution": {"repetitions": 2}},
        )
        assert seeded.status == 201, seeded.text()
        full_summary = page.request.get(base_url + "/api/batches/" + seeded.json()["batch_id"]).json()
        partial_summary = deepcopy(full_summary)
        partial_summary["runs"] = partial_summary["runs"][:1]
        partial_summary["evaluations"] = partial_summary["evaluations"][:1]
        partial_summary["counts"].update(total=1, completed=1, failed=0)
        for property_ in partial_summary["property_counts"]:
            property_["total"] = 1
        state = {"job": None}

        def execution_route(route):
            path = route.request.url.split("/api/executions", 1)[1]
            if route.request.method == "POST" and not path:
                payload = route.request.post_data_json
                starts.append(payload)
                state["job"] = {
                    "batch_id": "reload-test",
                    "saved_id": identity,
                    "title": definition["title"],
                    "total": 5,
                    "completed": 1,
                    "status": "running",
                    "summary": partial_summary,
                    "error": None,
                    "execution": payload["execution"],
                    "reviewer": None,
                }
            elif path.endswith("/cancel"):
                state["job"]["status"] = "cancelled"
            value = state["job"]
            route.fulfill(
                status=202 if route.request.method == "POST" and not path else 200,
                content_type="application/json",
                body=json.dumps(value),
            )

        page.route("**/api/executions**", execution_route)
        page.goto(base_url)
        expect(page.locator(f'#saved-experiment option[value="{identity}"]')).to_have_count(1)
        page.locator("#saved-experiment").select_option(identity)
        page.locator("#repetitions").fill("5")
        try:
            expect(page.locator("#run-button")).to_be_enabled(timeout=10000)
        except AssertionError:
            print(page.locator("#runner-selection-status").inner_text(), errors)
            raise
        page.locator("#run-button").click()
        expect(page.locator("#progress")).to_contain_text("1 / 5")
        expect(page.locator("#cancel-execution")).to_be_enabled()
        page.evaluate("localStorage.removeItem('active-execution')")
        page.reload()
        expect(page.locator("#progress")).to_contain_text("1 / 5")
        expect(page.locator("#saved-experiment")).to_have_value(identity)
        expect(page.locator("#saved-experiment")).to_be_disabled()
        expect(page.locator("#execution-spinner")).to_be_visible()
        assert len(starts) == 1
        expect(page.locator("#batch-results")).to_be_visible()
        expect(page.locator("#batch-run-selection option")).to_have_count(1)
        responses = page.locator(".property-run-details").first
        responses.locator("summary").click()
        expect(responses).to_have_attribute("open", "")
        state["job"]["completed"] = 2
        state["job"]["summary"] = full_summary
        expect(page.locator("#progress")).to_contain_text("2 / 5")
        expect(page.locator("#batch-run-selection option")).to_have_count(2)
        expect(page.locator(".property-run-details").first).to_have_attribute("open", "")
        expect(page.locator(".property-run-list").first.locator(":scope > li")).to_have_count(2)
        expect(page.locator("#execution-spinner")).to_be_visible()
        page.locator("#cancel-execution").click()
        expect(page.locator("#progress")).to_contain_text("cancelled", timeout=1500)
        expect(page.locator("#execution-spinner")).to_be_hidden()
        expect(page.locator("#cancel-execution")).to_be_hidden()
        expect(page.locator("#saved-experiment")).to_be_enabled()
        assert page.evaluate("localStorage.getItem('active-execution')") is None
        assert len(starts) == 1
        assert not errors, errors
        browser.close()
        print("Reload reconnected, no duplicate start, Cancel restored controls.")


if __name__ == "__main__":
    test_reconnect(sys.argv[1])
