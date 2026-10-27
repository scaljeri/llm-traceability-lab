"""Reviewer prompt modal smoke; intercept execution on a disposable database."""

import sys

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def main(address, batch_id):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_executable(), headless=True)
        page = browser.new_page()
        page.add_init_script("""Object.defineProperty(navigator, 'clipboard', {
            value: {writeText: async text => {window.copiedReviewerPrompt = text;}}
        });""")
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
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
        page.locator("#saved-experiment").select_option(
            page.locator("#saved-experiment option").nth(1).get_attribute("value")
        )
        page.locator("#run-button").click()
        source = page.get_by_role("button", name="View reviewer prompt")
        expect(source).to_be_visible()
        source.click()
        dialog = page.get_by_role("dialog", name="Reviewer prompt", exact=True)
        expect(dialog).to_be_visible()
        expect(page.locator("#reviewer-prompt-text")).to_contain_text("__ANSWER_JSON__")
        page.locator("#copy-reviewer-prompt").click()
        expect(page.locator("#snackbar")).to_have_text("Answer reviewer prompt copied to clipboard")
        assert (
            page.evaluate("window.copiedReviewerPrompt")
            == page.locator("#reviewer-prompt-text").text_content()
        )
        compact = (
            batch["evaluations"][0]["automatic"]["semantic_evaluation"].get("answer_input_format")
            == "compact-v1"
        )
        if compact:
            expect(page.locator("#reviewer-prompt-text")).to_contain_text('"documents"')
            expect(page.locator("#reviewer-prompt-text")).to_contain_text('"expected_answer"')
            expect(page.locator("#reviewer-prompt-text")).not_to_contain_text('"output_schema"')
        else:
            expect(page.locator("#reviewer-prompt-text")).to_contain_text("output_schema")
            expect(page.locator("#reviewer-prompt-text")).to_contain_text("supplied_sources")
        expect(page.locator("#reviewer-prompt-text")).to_contain_text("SYSTEM")
        if len(batch["runs"]) > 1:
            page.locator("#reviewer-prompt-run").select_option("1")
            expect(page.locator("#reviewer-prompt-text")).to_contain_text("__ANSWER_JSON__")
        page.set_viewport_size({"width": 390, "height": 844})
        assert dialog.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
        expect(page.locator("#close-reviewer-prompt")).to_be_in_viewport()
        page.keyboard.press("Escape")
        expect(dialog).not_to_be_visible()
        page.get_by_role("button", name="View fact reviewer prompt", exact=True).click()
        expect(dialog).to_be_visible()
        expect(page.locator("#reviewer-prompt-fact")).to_be_visible()
        expect(page.locator("#reviewer-prompt-text")).to_contain_text("generated_fact")
        expect(page.locator("#reviewer-prompt-text")).to_contain_text("supplied_sources")
        fact_calls = [
            call
            for call in batch["evaluations"][0]["automatic"]["semantic_evaluation"]["calls"]
            if call["phase"] == "fact"
        ]
        if len(fact_calls) > 1:
            page.locator("#reviewer-prompt-fact").select_option(str(fact_calls[1]["fact_index"]))
            expect(page.locator("#reviewer-prompt-text")).to_contain_text(
                fact_calls[1]["request_body"]["messages"][1]["content"]
            )
        page.locator("#copy-reviewer-prompt").click()
        expect(page.locator("#snackbar")).to_have_text("Fact reviewer prompt copied to clipboard")
        assert (
            page.evaluate("window.copiedReviewerPrompt")
            == page.locator("#reviewer-prompt-text").text_content()
        )
        page.keyboard.press("Escape")
        if any(item.get("evaluation_error") for item in batch["property_counts"]):
            answers = page.locator('.property-responses[data-property="answer"]')
            answers.locator(".property-run-details > summary").click()
            expect(answers.locator(".property-run-status").first).to_contain_text("Evaluation error")
            criterion = next(
                item for item in batch["evaluations"][0]["result_criteria"] if item["id"] == "answer"
            )
            expect(answers).to_contain_text(criterion["reason"])
            expect(answers.locator("..")).to_contain_text("0 failed checks")
            expect(answers.locator("..")).to_contain_text("evaluation errors")
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
