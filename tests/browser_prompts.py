"""Prompt management and runner selection against an isolated local backend."""

import json
import re

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def test_prompts(base_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_executable(), headless=True)
        page = browser.new_page()
        page.add_init_script("""Object.defineProperty(navigator, 'clipboard', {
            value: {writeText: async text => {window.copiedPrompt = text;}}
        });""")
        page.goto(base_url)
        expect(page.locator("#answer-prompt option")).to_have_count(1)
        expect(page.locator("#review-facts-prompt option")).to_have_count(1)
        expect(page.locator("#answer-prompt")).to_have_value("answer-default")
        expect(page.locator("#review-facts-prompt")).to_have_value("review-default")
        expect(page.locator("#review-answer-prompt option")).to_have_count(1)
        expect(page.locator("#review-answer-prompt")).to_have_value("review-default")
        for removed in (
            "experiment-system-prompt",
            "experiment-instructions",
            "prefill-context",
            "context-prefill-dialog",
        ):
            expect(page.locator(f"#{removed}")).to_have_count(0)
        for selector, name in (
            ("#answer-prompt", "generation_prompt"),
            ("#review-facts-prompt", "fact_review_prompt"),
            ("#review-answer-prompt", "answer_review_prompt"),
        ):
            expect(page.locator(f"{selector} option:checked")).to_have_text(name)
            style = page.locator(selector).evaluate(
                "node => { const s = getComputedStyle(node); return {appearance:s.appearance, padding:s.paddingRight, image:s.backgroundImage, position:s.backgroundPositionX}; }"
            )
            assert style["appearance"] == "none"
            assert float(style["padding"].removesuffix("px")) >= 36
            assert "chevron-down.svg" in style["image"] and "12px" in style["position"]
        expect(page.locator("#saved-experiment")).to_have_value("")
        selected = next(
            prompt
            for prompt in page.request.get(f"{base_url}/api/prompts").json()
            if prompt["id"] == page.locator("#answer-prompt").input_value()
        )
        page.locator("#input-preview summary").click()
        expect(page.locator("#run-system-prompt")).to_have_value(selected["system_prompt"])
        expect(page.locator("#run-context")).to_have_value(selected["user_prompt"])
        expect(page.locator("#run-system-prompt")).to_be_enabled()
        expect(page.locator("#run-context")).to_be_enabled()
        expect(page.locator("#run-button")).to_be_disabled()
        page.locator("#manage-prompts").click()
        dialog = page.locator("#prompts-manager")
        expect(dialog).to_be_visible()
        expect(page.locator("#managed-prompt")).to_have_value("answer-default")
        expect(page.locator("#prompt-user-field")).to_be_visible()
        page.locator('label[for="prompt-user"] button').click()
        expect(page.locator("#parameter-help")).to_be_visible()
        for wildcard in ["{{user_question}}", "{{documents}}", "{{output_json}}"]:
            expect(page.locator("#parameter-help-text")).to_contain_text(wildcard)
        page.locator("#close-parameter-help").click()
        page.locator("#prompt-category-review").click()
        expect(page.locator("#managed-prompt")).to_have_value("review-default")
        expect(page.locator("#prompt-tabs")).to_have_attribute("data-active", "review")
        assert (
            page.locator("#prompt-tabs").evaluate(
                "node => getComputedStyle(node, '::after').transitionProperty"
            )
            == "transform"
        )
        page.wait_for_function("""() => {
            const tabs = document.getElementById('prompt-tabs');
            return new DOMMatrix(getComputedStyle(tabs, '::after').transform).m41 >= tabs.clientWidth / 2 - 1;
        }""")
        page.locator("#prompt-category-review").press("ArrowLeft")
        expect(page.locator("#prompt-category-answer")).to_be_focused()
        expect(page.locator("#prompt-tabs")).to_have_attribute("data-active", "answer")
        page.locator("#prompt-category-answer").press("End")
        expect(page.locator("#prompt-category-review")).to_be_focused()
        expect(page.locator("#prompt-tabs")).to_have_attribute("data-active", "review")
        expect(page.locator("#prompt-system")).to_have_attribute("readonly", "")
        expect(page.locator("#prompt-review-fields")).to_be_visible()
        expect(page.locator("#prompt-user-field")).to_be_hidden()
        page.locator("#prompt-edit").click()
        page.locator("#prompt-name").fill("Shared review set")
        page.locator("#prompt-category-answer").click()
        expect(page.locator("#prompt-name")).to_have_value("generation_prompt")
        page.locator("#prompt-category-review").click()
        expect(page.locator("#prompt-name")).to_have_value("Shared review set")
        expect(page.locator("#prompt-save")).to_be_enabled()
        page.locator("#prompt-system").fill("Shared reviewer system")
        page.locator("#prompt-json-view").click()
        value = json.loads(page.locator("#prompt-json").input_value())
        assert value["type"] == "review"
        assert "fact_prompt" in value and "answer_prompt" in value
        page.locator("#prompt-form-view").click()
        page.locator("#prompt-save").click()
        expect(page.locator("#prompt-status")).to_have_text("Prompt saved.")
        expect(page.locator("#review-facts-prompt option:checked")).to_have_text("fact_Shared review set")
        page.locator("#prompt-category-answer").click()
        page.locator("#prompt-new").click()
        answer_id = page.locator("#prompt-id").input_value()
        page.locator("#prompt-name").fill("Browser answer")
        page.locator("#prompt-system").fill("Answer from supplied evidence only.")
        page.locator("#prompt-save").click()
        expect(page.locator("#prompt-status")).to_have_text("Prompt saved.")
        page.locator("#close-prompts").click()
        page.locator("#answer-prompt").select_option(answer_id)
        expect(page.locator("#run-system-prompt")).to_have_value("Answer from supplied evidence only.")
        expect(page.locator("#run-context")).to_have_value(re.compile(r"\{\{user_question\}\}"))
        expect(page.locator("#run-button")).to_be_disabled()
        experiment = page.locator("#saved-experiment option").nth(1).get_attribute("value")
        page.locator("#saved-experiment").select_option(experiment)
        expect(page.locator("#run-system-prompt")).to_have_value(
            re.compile("^Answer from supplied evidence only.")
        )
        page.reload()
        expect(page.locator("#answer-prompt")).to_have_value(answer_id)
        page.locator("#manage-prompts").click()
        expect(page.locator("#prompt-import")).to_have_count(0)
        expect(page.locator("#prompt-file")).to_have_count(0)
        page.locator("#prompt-category-review").click()
        page.locator("#prompt-copy").click()
        expect(page.locator("#snackbar")).to_have_text("Review prompt copied to clipboard")
        expect(page.locator("#snackbar")).to_be_visible()
        assert page.locator("#snackbar").evaluate("node => node.parentElement.id") == "prompts-manager"
        assert page.locator("#snackbar").evaluate("node => getComputedStyle(node).bottom") == "24px"
        copied = json.loads(page.evaluate("window.copiedPrompt"))
        assert copied["type"] == "review" and copied["name"] == "Shared review set"
        page.locator("#prompt-category-answer").click()
        page.locator("#prompt-copy").click()
        expect(page.locator("#snackbar")).to_have_text("Generation prompt copied to clipboard")
        expect(page.locator("#snackbar")).to_be_hidden(timeout=5000)
        page.evaluate(
            "() => { navigator.clipboard.writeText = async () => { throw new Error('Clipboard denied'); }; }"
        )
        page.locator("#prompt-copy").click()
        expect(page.locator("#prompt-status")).to_have_text("Clipboard denied")
        expect(page.locator("#snackbar")).to_be_hidden()
        page.locator("#prompt-category-review").click()
        page.locator("#prompt-new").click()
        page.locator("#prompt-json-view").click()
        imported = {
            "format": "traceability-prompt-v1",
            "id": "browser-import",
            "type": "review",
            "name": "Imported review",
            "system_prompt": "Review system",
            "fact_prompt": "FACT\n{{review_input}}",
            "answer_prompt": "ANSWER\n{{review_input}}",
        }
        page.locator("#prompt-json").fill(json.dumps(imported))
        page.locator("#prompt-save").click()
        expect(page.locator("#prompt-status")).to_have_text("Prompt saved.")
        expect(page.locator("#prompt-category-review")).to_have_attribute("aria-selected", "true")
        page.locator("#close-prompts").click()
        page.locator("#review-facts-prompt").select_option("browser-import")
        page.locator("#review-answer-prompt").select_option("review-default")
        page.reload()
        expect(page.locator("#review-facts-prompt")).to_have_value("browser-import")
        expect(page.locator("#review-answer-prompt")).to_have_value("review-default")
        page.locator("#manage-prompts").click()
        page.locator("#prompt-category-review").click()
        page.locator("#managed-prompt").select_option("browser-import")
        page.on("dialog", lambda confirmation: confirmation.accept())
        page.locator("#prompt-delete").click()
        expect(page.locator("#review-facts-prompt option")).to_have_count(1)
        expect(page.locator("#review-facts-prompt")).to_have_value("review-default")
        expect(page.locator("#review-answer-prompt")).to_have_value("review-default")
        page.locator("#prompt-category-review").click()
        page.locator("#managed-prompt").select_option("review-default")
        page.locator("#prompt-edit").click()
        page.locator("#prompt-form-view").click()
        page.locator("#prompt-fact").fill("Missing review payload")
        page.locator("#prompt-save").click()
        expect(page.locator("#prompt-status")).to_contain_text("review_input")
        expect(page.locator("#prompt-save")).to_be_enabled()
        for prompt in page.request.get(f"{base_url}/api/prompts").json():
            assert page.request.delete(f"{base_url}/api/prompts/{prompt['id']}").ok
        page.reload()
        for selector in ("#answer-prompt", "#review-facts-prompt", "#review-answer-prompt"):
            expect(page.locator(f"{selector} option")).to_have_count(0)
            expect(page.locator(selector)).to_be_disabled()
        expect(page.locator("#run-button")).to_be_disabled()
        browser.close()
