"""JSON save errors stay visible. Run only against an isolated disposable app."""

import json
import sys
from pathlib import Path

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def main(address):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=browser_executable(),
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        page.goto(address)
        page.locator("#manage-experiments").click()
        page.locator("#new-experiment").click()
        page.locator("#experiment-json-view").click()
        page.locator("#experiment-json").fill('{"title": "Missing question", "documents": []}')
        page.locator("#experiments-manager .manager-body").evaluate("el => el.scrollTop = el.scrollHeight")
        page.locator("#save-experiment").click()
        error = page.locator("#experiment-json-error")
        expect(error).to_be_visible()
        expect(error).to_contain_text("Cannot save experiment")
        expect(error).to_be_in_viewport()
        expect(page.locator("#save-experiment")).to_be_enabled()
        assert (
            page.locator("#experiment-json").input_value() == '{"title": "Missing question", "documents": []}'
        )
        page.locator("#experiment-json").fill('{"title": "Truncated",')
        page.locator("#save-experiment").click()
        expect(error).to_contain_text("Cannot save experiment")
        expect(error).to_be_in_viewport()
        assert page.locator("#experiment-json").input_value() == '{"title": "Truncated",'
        valid = json.loads(Path("experiments/04-perfect-citation.json").read_text())["experiment"]
        valid["title"] = "JSON save regression"
        page.locator("#experiment-json").fill(json.dumps(valid))
        page.locator("#save-experiment").click()
        expect(page.locator("#edit-experiment")).to_be_enabled()
        expect(page.locator("#experiment-json")).to_have_attribute("readonly", "")
        stored = page.request.get(address + "/api/experiment-definitions").json()
        saved = next(item for item in stored if item["definition"]["title"] == valid["title"])
        assert saved["definition"]["semantic_reference"] == valid["semantic_reference"]
        assert page.locator("#experiment-json-view").get_attribute("aria-pressed") == "true"
        # JSON edits must update the existing experiment rather than create a duplicate.
        page.locator("#edit-experiment").click()
        valid["title"] = "Updated JSON save regression"
        page.locator("#experiment-json").fill(json.dumps(valid))
        update_url = address + "/api/experiment-definitions/" + saved["id"]
        page.route(
            update_url, lambda route: route.fulfill(status=503, json={"detail": "Temporary save failure"})
        )
        page.locator("#save-experiment").click()
        expect(error).to_contain_text("Cannot save experiment: Temporary save failure")
        expect(error).to_be_in_viewport()
        assert json.loads(page.locator("#experiment-json").input_value()) == valid
        expect(page.locator("#save-experiment")).to_be_enabled()
        page.unroute(update_url)
        page.locator("#save-experiment").click()
        expect(page.locator("#edit-experiment")).to_be_enabled()
        updated = page.request.get(address + "/api/experiment-definitions").json()
        assert len(updated) == len(stored)
        assert (
            next(item for item in updated if item["id"] == saved["id"])["definition"]["title"]
            == valid["title"]
        )
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1])
