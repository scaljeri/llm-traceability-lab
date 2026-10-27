"""Verify pasted experiment JSON survives save, reload and form switching.

Run against an isolated app: python tests/browser_experiment_save.py BASE_URL
"""

import json
import re
import sys
from pathlib import Path

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def test_save(base_url):
    portable = json.loads(Path("experiments/03-conflcting-sources.json").read_text())
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=browser_executable(),
            headless=True,
        )
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        old = {**portable["experiment"], "system_prompt": "OLD SAVED PROMPT"}
        response = page.request.post(base_url + "/api/experiment-definitions", data=old)
        assert response.status == 201, response.text()
        identity = response.json()["id"]
        for switch_form in (False, True):
            page.goto(base_url)
            expect(page.locator(f'#saved-experiment option[value="{identity}"]')).to_have_count(1)
            page.locator("#saved-experiment").select_option(identity)
            page.locator("#manage-experiments").click()
            page.locator("#edit-experiment").click()
            if not switch_form:
                expect(page.locator("#legacy-prompt-warning")).to_be_visible()
            page.locator("#experiment-json-view").click()
            question = "Which budget is approved?" + (" Explain the evidence." if switch_form else "")
            portable["experiment"]["question"] = question
            portable["experiment"].pop("system_prompt", None)
            portable["experiment"].pop("context", None)
            page.locator("#experiment-json").fill(json.dumps(portable))
            if switch_form:
                page.locator("#experiment-form-view").click()
                expect(page.locator("#question")).to_have_value(question)
            page.locator("#save-experiment").click()
            expect(page.locator("#edit-experiment")).to_be_enabled()
            expect(page.locator("#legacy-prompt-warning")).to_be_hidden()
            saved = page.request.get(base_url + "/api/experiment-definitions").json()
            definition = next(item for item in saved if item["id"] == identity)
            assert definition["definition"]["question"] == question
            assert definition["revision"] == (3 if switch_form else 2)
            page.reload()
            expect(page.locator(f'#saved-experiment option[value="{identity}"]')).to_have_count(1)
            page.locator("#saved-experiment").select_option(identity)
            expect(page.locator("#run-context")).to_have_value(re.compile(re.escape(question)), timeout=10000)
            assert "OLD SAVED PROMPT" not in page.locator("#run-system-prompt").input_value()
        assert not errors, errors
        browser.close()
        print("JSON save/reload/form roundtrip preserves experiment data independently of prompts.")


if __name__ == "__main__":
    test_save(sys.argv[1])
