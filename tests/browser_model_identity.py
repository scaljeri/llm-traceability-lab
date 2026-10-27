"""Server manager smoke test. Use only an isolated disposable server/database."""

import sys

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def test_manifest(base_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_executable(), headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base_url)
        page.locator("#manage-servers").click()
        bounds = page.locator("#settings-menu").bounding_box()
        viewport = page.viewport_size
        assert abs(bounds["x"] + bounds["width"] / 2 - viewport["width"] / 2) < 2
        assert abs(bounds["y"] + bounds["height"] / 2 - viewport["height"] / 2) < 2
        page.route(
            "**/api/profile-models",
            lambda route: route.fulfill(json={"reachable": True, "models": ["test-model-a", "test-model-b"]}),
        )
        page.locator("#new-profile").click()
        expect(page.locator("#profile-max-tokens")).to_have_value("2048")
        page.locator("#profile-name").fill("My loaded server")
        page.locator("#profile-url").fill("http://localhost:8080/v1")
        page.locator("#load-profile-models").click()
        expect(page.locator("#profile-model-status")).to_have_text("2 models available.")
        page.locator("#profile-model").select_option("test-model-b")
        page.locator("#save-profile").click()
        expect(page.locator("#profile-status")).to_have_text("Server saved.")
        created_id = page.locator("#edit-profile").input_value()
        expect(page.locator(f'#llm-profile option[value="{created_id}"]')).to_have_text("My loaded server")
        expect(page.locator(f'#semantic-evaluator option[value="{created_id}"]')).to_have_text(
            "My loaded server"
        )
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator("#delete-profile").click()
        expect(page.locator("#profile-management")).to_be_hidden()
        assert not any(p["id"] == created_id for p in page.request.get(f"{base_url}/api/profiles").json())
        page.locator("#edit-profile").select_option("test-reviewer")
        expect(page.locator("#profile-model")).to_have_value("test-reviewer-model")
        expect(page.locator("#profile-name")).to_be_disabled()
        page.locator("#new-profile").click()
        page.locator("#profile-name").fill("Discarded draft")
        page.locator("#cancel-profile").click()
        expect(page.locator("#profile-management")).to_be_hidden()
        page.locator("#edit-profile").select_option("test-generator")
        page.locator("#edit-server").click()
        expect(page.locator("#profile-hardware")).to_be_hidden()
        expect(page.get_by_text("Version identity and embedding prefixes", exact=True)).to_be_hidden()
        before = next(
            p for p in page.request.get(f"{base_url}/api/profiles").json() if p["id"] == "test-generator"
        )
        page.locator("#profile-name").fill("Registered fixture model")
        page.locator("#profile-max-tokens").fill("4096")
        page.locator("#save-profile").click()
        expect(page.locator("#profile-status")).to_have_text("Server saved.")
        saved = next(
            p for p in page.request.get(f"{base_url}/api/profiles").json() if p["id"] == "test-generator"
        )
        assert saved["name"] == "Registered fixture model"
        assert saved["max_tokens"] == 4096
        expect(page.locator("#save-profile")).to_be_disabled()
        assert page.locator("#profile-status").evaluate("el => el.closest('.manager-footer') !== null")
        expect(page.locator("#profile-status")).to_be_in_viewport()
        page.locator("#edit-server").click()
        page.route(
            "**/api/profiles/test-generator",
            lambda route: route.fulfill(status=500, json={"detail": "Test storage error"}),
        )
        page.locator("#save-profile").click()
        expect(page.locator("#profile-status")).to_contain_text("Save failed:")
        expect(page.locator("#profile-status")).to_be_in_viewport()
        page.unroute("**/api/profiles/test-generator")
        page.locator("#cancel-profile").click()
        for key in (
            "hardware",
            "precision",
            "revision",
            "deployment_manifest",
            "capture_server_metadata",
            "trust_server_identity",
        ):
            assert saved[key] == before[key], key
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.locator("#settings-menu").evaluate("el => el.scrollWidth <= el.clientWidth + 1")
        page.locator("#close-settings").click()
        page.set_viewport_size({"width": 1280, "height": 900})
        seed_box = page.locator("#seed").bounding_box()
        repetitions_box = page.locator("#repetitions").bounding_box()
        assert abs(seed_box["y"] - repetitions_box["y"]) < 1
        assert abs(seed_box["height"] - repetitions_box["height"]) < 1
        advanced = page.locator("details").filter(has=page.locator("#max-tokens"))
        advanced.locator("summary").click()
        expect(page.locator("#max-tokens")).to_be_visible()
        expect(page.locator("#max-tokens")).to_have_value("")
        page.locator("#max-tokens").fill("4096")
        page.locator("#evaluator-max-tokens").fill("6144")
        page.locator("#evaluator-max-tokens").press("Tab")
        review_settings = page.evaluate("JSON.parse(localStorage.getItem('review-settings'))")
        assert review_settings is None or "max_tokens" not in review_settings
        profiles_after = page.request.get(f"{base_url}/api/profiles").json()
        assert next(p for p in profiles_after if p["id"] == "test-generator")["max_tokens"] == 4096
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    test_manifest(sys.argv[1])
