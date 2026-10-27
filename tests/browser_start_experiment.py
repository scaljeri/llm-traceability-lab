"""Regression: selecting a saved experiment enables Start without editing or inference.

Run against an isolated app: python tests/browser_start_experiment.py BASE_URL
Requires Playwright and a local Chrome installation.
"""

import sys

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def test_saved_experiment_can_start(base_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=browser_executable(),
            headless=True,
        )
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base_url)
        expect(page.locator("#saved-experiment option")).not_to_have_count(1)
        options = page.locator("#saved-experiment option").evaluate_all(
            "options => options.filter(option => option.value).map(option => option.value)"
        )
        assert options, "Seed at least one saved experiment in the isolated database."
        for identity in options:
            page.locator("#saved-experiment").select_option(identity)
            expect(page.locator("#run-button"), str(errors)).to_be_enabled(timeout=10000)
        preview_temperatures = []
        page.on(
            "request",
            lambda request: (
                preview_temperatures.append(request.post_data_json["temperature"])
                if request.url.endswith("/api/context-preview")
                else None
            ),
        )
        for identity in ["temperature", "evaluator-temperature"]:
            slider = page.locator(f"#{identity}")
            assert slider.get_attribute("type") == "range"
            assert slider.get_attribute("step") == "0.2"
            if identity == "evaluator-temperature":
                page.locator("summary", has_text="Advanced model options").click()
            slider.focus()
            page.keyboard.press("Home")
            page.keyboard.press("ArrowRight")
            expect(slider).to_have_value("0.2")
            expect(page.locator(f"#{identity}-value")).to_have_text("0.2")
            page.keyboard.press("End")
            expect(slider).to_have_value("2")
        # Retain a regression for the validation boundary even if the UI is bypassed.
        page.locator("#temperature").evaluate("input => input.type = 'number'")
        page.locator("#temperature").fill("80")
        expect(page.locator("#runner-selection-status")).to_contain_text(
            "Temperature must be between 0 and 2"
        )
        expect(page.locator("#run-button")).to_be_disabled()
        page.wait_for_timeout(500)
        assert 80 not in preview_temperatures, "Invalid temperature must not be sent to the preview API."
        page.locator("#temperature").fill("0.4")
        expect(page.locator("#run-button")).to_be_enabled()
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    test_saved_experiment_can_start(sys.argv[1])
