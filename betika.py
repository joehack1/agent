"""
Betika Selenium Auto-Bet Helper

Flow:
1. Open Betika homepage
2. Log in with configured username/password
3. Click first N odds between min/max on the main content area
4. Fill stake input
5. Place bet only when --execute is passed


"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable

try:
    from selenium import webdriver
    from selenium.common.exceptions import (
        StaleElementReferenceException,
        TimeoutException,
        WebDriverException,
    )
    from selenium.webdriver import ChromeOptions
    from selenium.webdriver.chrome.webdriver import WebDriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.remote.webelement import WebElement
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
except ImportError as exc:
    raise SystemExit(
        "Selenium is not installed. Run: .\\.venv\\Scripts\\python.exe -m pip install selenium"
    ) from exc


DEFAULT_HOME_URL = "https://www.betika.com/en-ke"
DEFAULT_LOGIN_URL = "https://www.betika.com/en-ke/login?next=%2F"
DEFAULT_USERNAME = "0795080677"
DEFAULT_PASSWORD = "4133"


class BotError(RuntimeError):
    """Raised when the browser automation flow fails."""


@dataclass
class BotConfig:
    home_url: str
    login_url: str
    username: str
    password: str
    stake: float
    count: int
    min_odds: float
    max_odds: float
    timeout: int
    max_scrolls: int
    execute: bool
    headless: bool
    keep_open: bool
    only_1x2: bool
    manual_login_wait: int
    debug_login: bool


def parse_args() -> BotConfig:
    parser = argparse.ArgumentParser(
        description="Betika Selenium bot: login, pick low odds, and place bet."
    )
    parser.add_argument("--home-url", default=os.getenv("BETIKA_HOME_URL", DEFAULT_HOME_URL))
    parser.add_argument("--login-url", default=os.getenv("BETIKA_LOGIN_URL", DEFAULT_LOGIN_URL))
    parser.add_argument("--username", default=os.getenv("BETIKA_USERNAME", DEFAULT_USERNAME))
    parser.add_argument("--password", default=os.getenv("BETIKA_PASSWORD", DEFAULT_PASSWORD))
    parser.add_argument("--stake", type=float, default=2.0)
    parser.add_argument("--count", type=int, default=39)
    parser.add_argument("--min-odds", type=float, default=1.01)
    parser.add_argument("--max-odds", type=float, default=1.35)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--max-scrolls", type=int, default=45)
    parser.add_argument("--execute", action="store_true", help="Actually click the Place Bet button")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true", help="Do not close browser after run")
    parser.add_argument(
        "--manual-login-wait",
        type=int,
        default=90,
        help="Seconds to wait for manual login if auto-login does not complete.",
    )
    parser.add_argument(
        "--debug-login",
        action="store_true",
        help="Save login page screenshot/source when auto-login fails.",
    )
    parser.add_argument(
        "--all-markets",
        action="store_true",
        help="Allow any market. Default behavior prefers 1/X/2 outcomes.",
    )

    args = parser.parse_args()

    if args.stake <= 0:
        raise SystemExit("--stake must be greater than 0")
    if args.count <= 0:
        raise SystemExit("--count must be greater than 0")
    if args.min_odds <= 0 or args.max_odds <= 0:
        raise SystemExit("--min-odds and --max-odds must be greater than 0")
    if args.min_odds > args.max_odds:
        raise SystemExit("--min-odds must be <= --max-odds")

    return BotConfig(
        home_url=args.home_url,
        login_url=args.login_url,
        username=args.username,
        password=args.password,
        stake=args.stake,
        count=args.count,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        timeout=args.timeout,
        max_scrolls=args.max_scrolls,
        execute=args.execute,
        headless=args.headless,
        keep_open=args.keep_open,
        only_1x2=not args.all_markets,
        manual_login_wait=args.manual_login_wait,
        debug_login=args.debug_login,
    )


class BetikaSeleniumBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.driver = self._build_driver()
        self.wait = WebDriverWait(self.driver, self.config.timeout)
        self.bet_confirmed = False
        self.bet_attempted = False

    def _build_driver(self) -> WebDriver:
        options = ChromeOptions()
        if self.config.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--start-maximized")

        try:
            return webdriver.Chrome(options=options)
        except WebDriverException as exc:
            raise BotError(
                "Could not start Chrome WebDriver. Ensure Chrome is installed and Selenium can download/use a driver."
            ) from exc

    def close(self) -> None:
        if self.config.keep_open:
            return
        if self.bet_attempted and not self.bet_confirmed:
            print("Browser left open because bet confirmation was not detected.")
            return
        self.driver.quit()

    def run(self) -> None:
        self.driver.get(self.config.login_url)
        self._dismiss_cookie_banner()
        self._login_if_needed()
        self.driver.get(self.config.home_url)
        self._dismiss_cookie_banner()

        picked = self.pick_low_odds()
        if not picked:
            raise BotError(
                f"No odds found in range {self.config.min_odds:.2f}-{self.config.max_odds:.2f}."
            )

        self._set_stake(self.config.stake)

        if self.config.execute:
            confirmation_text = self._place_bet()
            self.bet_confirmed = True
            print(f"Bet confirmed: {confirmation_text}")
        else:
            print("Dry-run mode: selections added and stake set, but Place Bet was not clicked.")

    def _dismiss_cookie_banner(self) -> None:
        candidates = [
            (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]"),
            (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]"),
        ]
        for by, selector in candidates:
            try:
                elements = self.driver.find_elements(by, selector)
            except WebDriverException:
                continue
            for element in elements:
                if self._is_displayed_safe(element):
                    self._safe_click(element)
                    time.sleep(0.4)
                    return

    def _login_if_needed(self) -> None:
        if self._is_logged_in():
            print("Already logged in.")
            return

        if "/login" not in self.driver.current_url:
            self.driver.get(self.config.login_url)
            self._dismiss_cookie_banner()

        user_input = self._find_first_visible(
            [
                (By.CSS_SELECTOR, "input[name*='mobile']"),
                (By.CSS_SELECTOR, "input[name*='phone']"),
                (By.CSS_SELECTOR, "input[type='tel']"),
                (By.CSS_SELECTOR, "input[name*='username']"),
                (By.CSS_SELECTOR, "input[id*='phone']"),
                (By.CSS_SELECTOR, "input[placeholder*='phone']"),
                (By.CSS_SELECTOR, "input[placeholder*='Phone']"),
                (By.CSS_SELECTOR, "input[aria-label*='phone']"),
                (By.CSS_SELECTOR, "input[aria-label*='Phone']"),
                (
                    By.XPATH,
                    "//label[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'phone number')]/following::input[1]",
                ),
                (
                    By.XPATH,
                    "//input[contains(translate(@placeholder, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'enter your phone number') or contains(translate(@placeholder, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'phone number') or contains(translate(@placeholder, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'e.g. 0712')]",
                ),
                (
                    By.XPATH,
                    "//input[contains(translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'phone') or contains(translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'mobile') or contains(translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'username')]",
                ),
            ],
            timeout=12,
        )
        if user_input is None:
            raise BotError("Could not find username/mobile input on login page.")

        pass_input = self._find_first_visible(
            [
                (By.CSS_SELECTOR, "input[type='password']"),
                (By.CSS_SELECTOR, "input[name*='password']"),
                (By.CSS_SELECTOR, "input[id*='password']"),
                (By.CSS_SELECTOR, "input[placeholder*='password']"),
                (By.CSS_SELECTOR, "input[placeholder*='Password']"),
                (
                    By.XPATH,
                    "//label[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'password')]/following::input[1]",
                ),
                (
                    By.XPATH,
                    "//input[contains(translate(@placeholder, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'enter your password')]",
                ),
            ],
            timeout=10,
        )
        if pass_input is None:
            raise BotError("Could not find password input.")

        self._set_input_value(user_input, self.config.username)
        self._set_input_value(pass_input, self.config.password)

        submit = self._find_first_visible(
            [
                (By.CSS_SELECTOR, "button[type='submit']"),
                (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login')]"),
                (By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in')]"),
            ],
            timeout=6,
        )
        if submit is None:
            raise BotError("Could not find login submit button.")

        self._safe_click(submit)

        if not self._wait_until_logged_in(timeout=self.config.timeout):
            raise BotError("Login did not complete. Credentials or selectors may be incorrect.")

        print("Login successful.")

    def _is_logged_in(self) -> bool:
        # If login form or login CTA is visible, user is not authenticated yet.
        not_logged_in_markers = [
            (By.CSS_SELECTOR, "input[type='password']"),
            (
                By.XPATH,
                "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login') or contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in')]",
            ),
            (
                By.XPATH,
                "//a[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'login') or contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'sign in')]",
            ),
        ]
        for by, selector in not_logged_in_markers:
            try:
                elements = self.driver.find_elements(by, selector)
            except WebDriverException:
                continue
            for el in elements:
                if self._is_displayed_safe(el):
                    return False

        # Stronger logged-in indicators.
        indicators = [
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'logout')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'my bets')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'profile')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'balance')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'notifications')]",
            ),
        ]
        matched = 0
        for by, selector in indicators:
            try:
                elements = self.driver.find_elements(by, selector)
            except WebDriverException:
                continue
            for el in elements:
                if self._is_displayed_safe(el):
                    matched += 1
                    break

        # Require at least two independent indicators to avoid false positives.
        return matched >= 2

    def _wait_until_logged_in(self, timeout: int) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self._is_logged_in():
                    return True
            except StaleElementReferenceException:
                time.sleep(0.2)
                continue
            except WebDriverException:
                time.sleep(0.2)
                continue
            if "/login" not in self.driver.current_url:
                return True
            time.sleep(0.5)
        return False

    def pick_low_odds(self) -> list[dict[str, str | float]]:
        clicked_ids: set[str] = set()
        picked: list[dict[str, str | float]] = []

        for scroll_step in range(self.config.max_scrolls):
            new_clicks = 0
            buttons = self._find_low_odd_buttons()
            for button, odd, label in buttons:
                try:
                    if button.id in clicked_ids:
                        continue
                    if self._is_already_selected(button):
                        clicked_ids.add(button.id)
                        continue

                    clicked = self._safe_click(button)
                    if not clicked:
                        continue
                    clicked_ids.add(button.id)
                    picked.append({"odd": odd, "label": label})
                    new_clicks += 1
                    print(f"Picked {len(picked)}/{self.config.count}: {label} @ {odd:.2f}")
                    time.sleep(0.15)
                except StaleElementReferenceException:
                    continue

                if len(picked) >= self.config.count:
                    return picked

            if new_clicks == 0:
                print(
                    f"Scroll step {scroll_step + 1}: no new picks from {len(buttons)} candidates, scrolling..."
                )
                self.driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight * 0.85));")
                time.sleep(0.7)
            else:
                self.driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight * 0.35));")
                time.sleep(0.4)

        return picked

    def _find_low_odd_buttons(self) -> list[tuple[WebElement, float, str]]:
        selectors: list[tuple[str, str]] = [
            (By.CSS_SELECTOR, "[data-odd], [data-odds], [class*='odd-btn'], button[class*='odd']"),
            (By.CSS_SELECTOR, "[class*='outcome'] button, [class*='market'] button"),
            (By.CSS_SELECTOR, "button"),
            (By.CSS_SELECTOR, "[role='button']"),
        ]

        window_width = self.driver.execute_script("return window.innerWidth")
        max_main_x = float(window_width) * 0.72

        found: list[tuple[WebElement, float, str]] = []
        seen: set[str] = set()
        for by, selector in selectors:
            elements = self.driver.find_elements(by, selector)
            for element in elements:
                try:
                    if element.id in seen:
                        continue
                    seen.add(element.id)

                    if not element.is_displayed() or not element.is_enabled():
                        continue

                    rect = element.rect
                    center_x = rect.get("x", 0) + rect.get("width", 0) / 2
                    if center_x > max_main_x:
                        continue

                    text = (element.text or "").strip()
                    odd = parse_odd(text)
                    if odd is None:
                        continue
                    if odd < self.config.min_odds or odd > self.config.max_odds:
                        continue
                    if rect.get("width", 0) < 24 or rect.get("height", 0) < 18:
                        continue

                    label = text.replace("\n", " ")
                    if self.config.only_1x2 and not looks_like_1x2(label):
                        continue

                    found.append((element, odd, label))
                except StaleElementReferenceException:
                    continue
                except WebDriverException:
                    continue

        return found

    def _is_already_selected(self, element: WebElement) -> bool:
        try:
            class_name = (element.get_attribute("class") or "").lower()
        except StaleElementReferenceException:
            return True
        selected_tokens = ("selected", "active", "picked", "highlight")
        return any(token in class_name for token in selected_tokens)

    def _set_stake(self, stake: float) -> None:
        stake_input = self._find_first_visible(
            [
                (By.XPATH, "//input[contains(@placeholder, 'Amount') or contains(@placeholder, 'KES')]") ,
                (By.CSS_SELECTOR, "input[name*='stake'], input[name*='amount']"),
                (By.CSS_SELECTOR, "input[type='number']"),
            ],
            timeout=10,
            prefer_right_panel=True,
        )
        if stake_input is None:
            raise BotError("Could not find stake input in betslip.")

        self._set_input_value(stake_input, str(int(stake) if float(stake).is_integer() else stake))
        print(f"Stake set to {stake}.")

    def _place_bet(self) -> str:
        self.bet_attempted = True
        deadline = time.time() + self.config.timeout + 15
        click_count = 0

        while time.time() < deadline:
            removed = self._click_remove_expired()
            if removed:
                time.sleep(0.7)
                confirmation = self._wait_for_bet_confirmation(timeout=2)
                if confirmation:
                    return confirmation
                continue

            button, label = self._find_place_bet_action()
            if button is None:
                confirmation = self._wait_for_bet_confirmation(timeout=2)
                if confirmation:
                    return confirmation
                time.sleep(0.4)
                continue

            clicked = self._safe_click(button)
            if not clicked:
                time.sleep(0.2)
                continue

            click_count += 1
            print(f"{label} clicked ({click_count}). Waiting for confirmation...")

            confirmation = self._wait_for_bet_confirmation(timeout=4)
            if confirmation:
                return confirmation

            # If confirmation is delayed or odds changed, loop and retry action.
            time.sleep(0.5)

        raise BotError("Bet action was clicked but no final confirmation was detected.")

    def _click_remove_expired(self) -> bool:
        remove_locators = [
            (
                By.XPATH,
                "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'remove') and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'expired')]",
            ),
            (
                By.XPATH,
                "//*[@role='button' and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'remove') and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'expired')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'remove expired')]",
            ),
        ]
        button = self._find_first_visible(
            remove_locators,
            timeout=1,
            prefer_right_panel=True,
        )
        if button is None:
            return False
        if not self._safe_click(button):
            return False
        print("Clicked remove expired selections.")
        return True

    def _find_place_bet_action(self) -> tuple[WebElement | None, str]:
        accept_then_place = [
            (
                By.XPATH,
                "//button[(contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept') and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'place bet')) or contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept & place bet')]",
            ),
            (
                By.XPATH,
                "//*[(@role='button' or self::button) and ((contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept') and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'place bet')) or contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept & place bet'))]",
            ),
        ]
        place_only = [
            (
                By.XPATH,
                "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'place bet')]",
            ),
            (
                By.XPATH,
                "//*[(@role='button' or self::button) and contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'place bet')]",
            ),
        ]

        button = self._find_first_visible(
            accept_then_place,
            timeout=1,
            prefer_right_panel=True,
        )
        if button is not None:
            return button, "Accept and Place Bet"

        button = self._find_first_visible(
            place_only,
            timeout=1,
            prefer_right_panel=True,
        )
        if button is not None:
            return button, "Place Bet"

        return None, ""

    def _wait_for_bet_confirmation(self, timeout: int) -> str | None:
        success_locators = [
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'bet placed')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'successfully')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'booking code')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'bet id')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'receipt')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accepted')]",
            ),
        ]
        error_locators = [
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'insufficient balance')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'minimum stake')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'maximum stake')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'market suspended')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'selection is no longer available')]",
            ),
            (
                By.XPATH,
                "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'session expired')]",
            ),
        ]

        end = time.time() + timeout
        while time.time() < end:
            error_text = self._find_visible_text(error_locators, prefer_right_panel=True)
            if error_text:
                raise BotError(f"Bet rejected: {error_text}")

            success_text = self._find_visible_text(success_locators, prefer_right_panel=True)
            if success_text:
                return success_text

            if "/my-bets" in self.driver.current_url.lower():
                return "navigated to My Bets page"

            time.sleep(0.4)

        return None

    def _find_visible_text(
        self, locators: Iterable[tuple[str, str]], prefer_right_panel: bool = False
    ) -> str | None:
        right_threshold = None
        if prefer_right_panel:
            try:
                window_width = self.driver.execute_script("return window.innerWidth")
                right_threshold = float(window_width) * 0.58
            except WebDriverException:
                right_threshold = None

        for by, selector in locators:
            try:
                elements = self.driver.find_elements(by, selector)
            except WebDriverException:
                continue

            for element in elements:
                try:
                    if not self._is_displayed_safe(element):
                        continue
                    if right_threshold is not None:
                        rect = element.rect
                        center_x = rect.get("x", 0) + (rect.get("width", 0) / 2)
                        if center_x < right_threshold:
                            continue
                    text = " ".join((element.text or "").split())
                    # Ignore giant containers (like full page body) that produce false positives.
                    if len(text) > 280:
                        continue
                    if text:
                        return text
                except StaleElementReferenceException:
                    continue
                except WebDriverException:
                    continue

        return None

    def _find_first_visible(
        self,
        locators: Iterable[tuple[str, str]],
        timeout: int,
        prefer_right_panel: bool = False,
    ) -> WebElement | None:
        end = time.time() + timeout
        while time.time() < end:
            candidates: list[WebElement] = []
            for by, selector in locators:
                try:
                    elements = self.driver.find_elements(by, selector)
                except WebDriverException:
                    continue
                for element in elements:
                    if self._is_displayed_safe(element):
                        candidates.append(element)

            if prefer_right_panel and candidates:
                window_width = self.driver.execute_script("return window.innerWidth")
                threshold = float(window_width) * 0.58
                right_side = [
                    el
                    for el in candidates
                    if (el.rect.get("x", 0) + (el.rect.get("width", 0) / 2)) > threshold
                ]
                if right_side:
                    return right_side[0]

            if candidates:
                return candidates[0]

            time.sleep(0.25)

        return None

    def _is_displayed_safe(self, element: WebElement) -> bool:
        try:
            return element.is_displayed()
        except StaleElementReferenceException:
            return False
        except WebDriverException:
            return False

    def _safe_click(self, element: WebElement) -> bool:
        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'instant', block: 'center'});", element
            )
            self.wait.until(EC.element_to_be_clickable(element))
            element.click()
            return True
        except StaleElementReferenceException:
            return False
        except Exception:
            try:
                self.driver.execute_script("arguments[0].click();", element)
                return True
            except StaleElementReferenceException:
                return False
            except WebDriverException:
                return False

    def _set_input_value(self, element: WebElement, value: str) -> None:
        element.click()
        element.send_keys(Keys.CONTROL, "a")
        element.send_keys(Keys.BACKSPACE)
        element.send_keys(value)
        self.driver.execute_script(
            "arguments[0].dispatchEvent(new Event('input', {bubbles: true}));"
            "arguments[0].dispatchEvent(new Event('change', {bubbles: true}));",
            element,
        )


def parse_odd(text: str) -> float | None:
    if not text:
        return None

    compact = " ".join(text.replace("\n", " ").split())
    raw_numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", compact)
    if not raw_numbers:
        return None

    parsed: list[tuple[str, float]] = []
    for raw in raw_numbers:
        try:
            parsed.append((raw, float(raw.replace(",", "."))))
        except ValueError:
            continue

    if not parsed:
        return None

    # Odds chips often look like "1 1.48", where the first token is outcome label.
    # Prefer decimal-looking tokens and return the last plausible one.
    decimal_values = [value for raw, value in parsed if "." in raw or "," in raw]
    candidates = decimal_values or [value for _, value in parsed]
    plausible = [value for value in candidates if 1.01 <= value <= 1000]
    if plausible:
        return plausible[-1]
    return candidates[-1]


def looks_like_1x2(label: str) -> bool:
    compact = " ".join(label.lower().split())
    normalized = compact.replace(" ", "")

    if "1x2" in normalized or "matchresult" in normalized:
        return True

    # Many homepage odds chips are just a single odd value under 1/X/2 columns.
    if re.fullmatch(r"\d+(?:[.,]\d+)?", compact):
        return True

    tokens = re.findall(r"[a-z]+|x|\d+", normalized)
    accepted = {"1", "x", "2", "home", "draw", "away"}
    return any(token in accepted for token in tokens)


def main() -> int:
    config = parse_args()

    bot = BetikaSeleniumBot(config)
    try:
        bot.run()
        return 0
    except BotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except StaleElementReferenceException:
        print(
            "ERROR: Page refreshed while elements were being read. Please run the command again.",
            file=sys.stderr,
        )
        return 1
    except TimeoutException as exc:
        print(f"ERROR: Timed out waiting for page elements: {exc}", file=sys.stderr)
        return 1
    finally:
        bot.close()


if __name__ == "__main__":
    raise SystemExit(main())
