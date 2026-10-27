"""File-sync controls against disposable copies of the example libraries."""

import json

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def test_file_sync(base_url, directory):
    experiment = directory / "experiments/01-complete-experiment.json"
    value = json.loads(experiment.read_text())
    value["experiment"]["title"] = "Changed on disk"
    experiment.write_text(json.dumps(value))
    prompt = directory / "prompts/generation_prompt.json"
    value = json.loads(prompt.read_text())
    value["name"] = "Prompt from disk"
    prompt.write_text(json.dumps(value))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_executable(), headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base_url)
        expect(page.locator("#answer-prompt option:checked")).to_have_text("generation_prompt")
        page.locator("#sync-files").click()
        expect(page.locator("#file-sync-dialog")).to_be_visible()
        page.once("dialog", lambda dialog: dialog.dismiss())
        page.locator("#import-library-files").click()
        expect(page.locator("#answer-prompt option:checked")).to_have_text("generation_prompt")
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator("#import-library-files").click()
        expect(page.locator("#file-sync-status")).to_have_text("Loaded 7 experiments and 2 prompt sets.")
        expect(page.locator("#saved-experiment option", has_text="Changed on disk")).to_have_count(1)
        expect(page.locator("#answer-prompt option:checked")).to_have_text("Prompt from disk")
        expect(page.locator("#run-system-prompt")).not_to_have_value("")
        response = page.request.post(
            f"{base_url}/api/prompts",
            data={
                "id": "browser-new",
                "name": "New app prompt",
                "type": "answer",
                "user_prompt": "{{user_question}}\n{{documents}}",
            },
        )
        assert response.ok, response.text()
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator("#export-library-files").click()
        expect(page.locator("#file-sync-status")).to_have_text("Exported 7 experiments and 3 prompt sets.")
        assert any(
            json.loads(path.read_text()).get("id") == "browser-new"
            for path in (directory / "prompts").glob("*.json")
        )
        (directory / "experiments/broken.json").write_text("{")
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator("#import-library-files").click()
        expect(page.locator("#file-sync-status")).to_contain_text(
            "Cannot load initial experiment broken.json"
        )
        assert len(page.request.get(f"{base_url}/api/prompts").json()) == 3
        page.locator("#close-file-sync").click()
        expect(page.locator("#file-sync-dialog")).not_to_be_visible()
        assert not errors, errors
        browser.close()
