"""First-server onboarding; point only at a disposable application database."""

import sys

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def test_first_server(base_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_executable(), headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        assert page.request.get(f"{base_url}/api/profiles").json() == []
        page.goto(base_url)
        dialog = page.locator("#settings-menu")
        expect(dialog).to_be_visible()
        expect(page.locator("#profile-name")).to_be_editable()
        page.locator("#profile-name").fill("My first server")
        page.keyboard.press("Escape")
        expect(dialog).to_be_visible()
        expect(page.locator("#profile-name")).to_have_value("My first server")
        expect(page.locator("#close-settings")).to_be_disabled()
        page.reload()
        expect(dialog).to_be_visible()
        expect(page.locator("#profile-name")).to_be_editable()
        page.locator("#profile-name").fill("My first server")
        page.locator("#profile-url").fill("http://localhost:8080/v1")
        page.route(
            "**/api/profile-models",
            lambda route: route.fulfill(json={"reachable": True, "models": ["example-model"]}),
        )
        page.locator("#load-profile-models").click()
        expect(page.locator("#profile-model-status")).to_have_text("1 models available.")
        page.locator("#save-profile").click()
        expect(page.locator("#profile-status")).to_have_text("Server saved.")
        expect(page.locator("#close-settings")).to_be_enabled()
        page.locator("#close-settings").click()
        expect(dialog).not_to_be_visible()
        page.reload()
        expect(page.locator("#llm-profile option")).to_have_count(1)
        first = page.request.get(f"{base_url}/api/profiles").json()[0]["id"]
        for selector in ("#llm-profile", "#semantic-evaluator"):
            expect(page.locator(selector)).to_have_value(first)
            expect(page.locator(f"{selector} option")).to_have_count(1)
        page.evaluate(
            "localStorage.setItem('review-settings', JSON.stringify({llm_profile_id: 'deleted-server'}))"
        )
        page.reload()
        expect(page.locator("#semantic-evaluator")).to_have_value(first)
        expect(page.locator("#semantic-evaluator option")).to_have_count(1)
        response = page.request.post(
            f"{base_url}/api/profiles",
            data={
                "name": "Second server",
                "kind": "llm",
                "url": "http://localhost:8081/v1",
                "model": "second",
            },
        )
        assert response.ok, response.text()
        second = response.json()["id"]
        page.reload()
        for selector in ("#llm-profile", "#semantic-evaluator"):
            expect(page.locator(f"{selector} option")).to_have_count(2)
            page.locator(selector).select_option(second)
        page.reload()
        for selector in ("#llm-profile", "#semantic-evaluator"):
            expect(page.locator(selector)).to_have_value(second)
        page.locator("#saved-experiment").select_option(index=1)
        expect(page.locator("#semantic-evaluator")).to_have_value(second)
        assert page.request.delete(f"{base_url}/api/profiles/{second}").ok
        page.reload()
        for selector in ("#llm-profile", "#semantic-evaluator"):
            expect(page.locator(selector)).to_have_value(first)
            expect(page.locator(f"{selector} option")).to_have_count(1)
        expect(dialog).not_to_be_visible()
        page.locator("#manage-servers").click()
        page.once("dialog", lambda confirmation: confirmation.accept())
        page.locator("#delete-profile").click()
        expect(page.locator("#profile-name")).to_be_editable()
        expect(page.locator("#profile-name")).to_have_value("")
        expect(page.locator("#close-settings")).to_be_disabled()
        page.keyboard.press("Escape")
        expect(dialog).to_be_visible()
        assert page.request.get(f"{base_url}/api/profiles").json() == []
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    test_first_server(sys.argv[1])
