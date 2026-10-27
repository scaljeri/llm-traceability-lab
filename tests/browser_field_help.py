"""Only the info button opens help; clicking a field label must not open it."""

import sys

from browser_support import browser_executable
from playwright.sync_api import expect, sync_playwright


def test_help_requires_button_click(base_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=browser_executable(),
            headless=True,
        )
        page = browser.new_page()
        page.goto(base_url)
        for identity in ("temperature", "llm-profile", "seed"):
            field = page.locator(f"#{identity}")
            label = page.locator(f'label[for="{identity}"]')
            if not label.count():
                label = page.locator("label").filter(has=field)
            expect(label.locator(".field-info")).to_have_count(1)
            box = label.bounding_box()
            page.mouse.click(box["x"] + 3, box["y"] + 8)
            expect(page.locator("#parameter-help")).not_to_be_visible()
            assert label.evaluate("label => label.control.id") == identity
            label.locator(".field-info").click()
            expect(page.locator("#parameter-help")).to_be_visible()
            page.keyboard.press("Escape")
        browser.close()


if __name__ == "__main__":
    test_help_requires_button_click(sys.argv[1])
