"""
Betika API helper (educational)

Features:
- Session-based authentication
- Fetch prematch odds from API JSON
- Place a bet with explicit --execute (dry-run by default)

Before real usage, inspect Betika network calls in your browser and update endpoint
paths/payload fields to match the live API contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any

import requests

DEFAULT_BASE_URL = "https://www.betika.com"
DEFAULT_TIMEOUT = 20

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

TOKEN_KEYS = {"token", "access_token", "jwt", "auth_token", "api_token"}


class BetikaApiError(RuntimeError):
    """Raised when the Betika flow fails."""


@dataclass
class ApiPaths:
    login: str = "/api/v1/login"
    markets: str = "/api/v1/prematch/matches"
    place_bet: str = "/api/v1/bets/place"


class BetikaClient:
    def __init__(
        self,
        base_url: str,
        paths: ApiPaths,
        timeout: int = DEFAULT_TIMEOUT,
        csrf_token: str | None = None,
        bearer_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.paths = paths
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

        if csrf_token:
            self.session.headers["X-CSRF-Token"] = csrf_token
        if bearer_token:
            self.session.headers["Authorization"] = f"Bearer {bearer_token}"

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request_json(
        self,
        method: str,
        path: str,
        expected_status: tuple[int, ...] = (200,),
        **kwargs: Any,
    ) -> dict[str, Any] | list[Any]:
        url = self._url(path)
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise BetikaApiError(f"Request failed for {url}: {exc}") from exc

        if response.status_code not in expected_status:
            preview = response.text[:400].replace("\n", " ")
            raise BetikaApiError(
                f"{method} {url} failed with {response.status_code}: {preview}"
            )

        try:
            return response.json()
        except ValueError as exc:
            preview = response.text[:300].replace("\n", " ")
            raise BetikaApiError(f"Response from {url} is not valid JSON: {preview}") from exc

    def login(self, payload: dict[str, Any]) -> dict[str, Any] | list[Any]:
        data = self._request_json(
            "POST",
            self.paths.login,
            expected_status=(200, 201),
            json=payload,
        )

        token = _find_token(data)
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

        return data

    def fetch_markets(
        self,
        sport: str = "soccer",
        page: int = 1,
        per_page: int = 100,
    ) -> dict[str, Any] | list[Any]:
        params = {"sport": sport, "page": page, "per_page": per_page}
        return self._request_json("GET", self.paths.markets, params=params)

    def place_bet(
        self,
        payload: dict[str, Any],
        dry_run: bool = True,
    ) -> dict[str, Any] | list[Any]:
        if dry_run:
            return {"dry_run": True, "payload": payload}

        return self._request_json(
            "POST",
            self.paths.place_bet,
            expected_status=(200, 201),
            json=payload,
        )


def _find_token(data: Any) -> str | None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in TOKEN_KEYS and isinstance(value, str) and value.strip():
                return value.strip()
        for value in data.values():
            token = _find_token(value)
            if token:
                return token
    elif isinstance(data, list):
        for item in data:
            token = _find_token(item)
            if token:
                return token
    return None


def _format_payload_template(template: Any, username: str, password: str) -> Any:
    if isinstance(template, dict):
        return {
            key: _format_payload_template(value, username, password)
            for key, value in template.items()
        }
    if isinstance(template, list):
        return [_format_payload_template(item, username, password) for item in template]
    if isinstance(template, str):
        return template.format(username=username, password=password)
    return template


def _extract_matches(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return [item for item in data["data"] if isinstance(item, dict)]
        if isinstance(data.get("matches"), list):
            return [item for item in data["matches"] if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def flatten_outcomes(markets_data: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matches = _extract_matches(markets_data)

    for match_index, match in enumerate(matches):
        match_id = match.get("id") or match.get("match_id")
        home = match.get("home_team") or match.get("home") or "?"
        away = match.get("away_team") or match.get("away") or "?"

        direct_outcomes = match.get("outcomes") or match.get("odds") or []
        if isinstance(direct_outcomes, list):
            for outcome_index, outcome in enumerate(direct_outcomes):
                if not isinstance(outcome, dict):
                    continue
                rows.append(
                    {
                        "match_order": match_index,
                        "match_id": match_id,
                        "home": home,
                        "away": away,
                        "market_id": outcome.get("market_id") or outcome.get("market"),
                        "market_name": outcome.get("market_name") or outcome.get("market"),
                        "outcome_id": outcome.get("id") or outcome.get("outcome_id"),
                        "selection_id": outcome.get("selection_id") or outcome.get("id"),
                        "outcome_order": outcome_index,
                        "name": outcome.get("name") or outcome.get("label") or "?",
                        "odd": _to_float(outcome.get("odd") or outcome.get("value")),
                    }
                )

        markets = match.get("markets") or []
        if isinstance(markets, list):
            for market in markets:
                if not isinstance(market, dict):
                    continue
                market_id = market.get("id") or market.get("market_id")
                market_name = market.get("name") or market.get("market_name") or market.get("label")
                outcomes = market.get("outcomes") or market.get("selections") or []
                if not isinstance(outcomes, list):
                    continue
                for outcome_index, outcome in enumerate(outcomes):
                    if not isinstance(outcome, dict):
                        continue
                    rows.append(
                        {
                            "match_order": match_index,
                            "match_id": match_id,
                            "home": home,
                            "away": away,
                            "market_id": market_id,
                            "market_name": market_name,
                            "outcome_id": outcome.get("id") or outcome.get("outcome_id"),
                            "selection_id": outcome.get("selection_id") or outcome.get("id"),
                            "outcome_order": outcome_index,
                            "name": outcome.get("name") or outcome.get("label") or "?",
                            "odd": _to_float(outcome.get("odd") or outcome.get("value")),
                        }
                    )

    return dedupe_outcomes(rows)


def dedupe_outcomes(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for row in outcomes:
        key = (
            str(row.get("match_id") or ""),
            str(row.get("selection_id") or row.get("outcome_id") or ""),
            str(row.get("name") or ""),
            str(row.get("odd") if row.get("odd") is not None else ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    return unique


def find_selection(
    outcomes: list[dict[str, Any]],
    selection_id: str | None,
    match_id: str | None,
    outcome_id: str | None,
) -> dict[str, Any] | None:
    if selection_id:
        for row in outcomes:
            if str(row.get("selection_id")) == str(selection_id):
                return row

    if match_id and outcome_id:
        for row in outcomes:
            if str(row.get("match_id")) == str(match_id) and str(row.get("outcome_id")) == str(outcome_id):
                return row

    return None


def build_bet_payload(selection: dict[str, Any], stake: float, currency: str) -> dict[str, Any]:
    pick: dict[str, Any] = {
        "selection_id": selection.get("selection_id"),
        "outcome_id": selection.get("outcome_id"),
        "match_id": selection.get("match_id"),
        "odd": selection.get("odd"),
        "name": selection.get("name"),
    }
    pick = {k: v for k, v in pick.items() if v is not None}

    return {
        "stake": stake,
        "currency": currency,
        "selections": [pick],
    }


def build_multi_bet_payload(
    selections: list[dict[str, Any]], stake: float, currency: str
) -> dict[str, Any]:
    picks: list[dict[str, Any]] = []
    for selection in selections:
        pick = {
            "selection_id": selection.get("selection_id"),
            "outcome_id": selection.get("outcome_id"),
            "match_id": selection.get("match_id"),
            "odd": selection.get("odd"),
            "name": selection.get("name"),
        }
        pick = {key: value for key, value in pick.items() if value is not None}
        picks.append(pick)

    return {
        "stake": stake,
        "currency": currency,
        "selections": picks,
    }


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def print_outcomes(outcomes: list[dict[str, Any]], max_odds: float | None = None) -> None:
    filtered = outcomes
    if max_odds is not None:
        filtered = [row for row in outcomes if row.get("odd") is not None and row["odd"] <= max_odds]

    if not filtered:
        print("No outcomes found for the applied filters.")
        return

    filtered = sorted(filtered, key=lambda row: (row.get("odd") is None, row.get("odd") or 9999))

    print("odd    match_id  selection_id  outcome_id  teams  name")
    print("-" * 90)
    for row in filtered:
        odd = "?" if row.get("odd") is None else f"{row['odd']:.2f}"
        teams = f"{row.get('home', '?')} vs {row.get('away', '?')}"
        print(
            f"{odd:<6} {str(row.get('match_id', '?')):<9} "
            f"{str(row.get('selection_id', '?')):<12} {str(row.get('outcome_id', '?')):<10} "
            f"{teams:<30} {row.get('name', '?')}"
        )


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def fetch_outcomes_pages(
    client: BetikaClient,
    sport: str,
    start_page: int,
    pages: int,
    per_page: int,
) -> list[dict[str, Any]]:
    all_outcomes: list[dict[str, Any]] = []
    for page in range(start_page, start_page + pages):
        data = client.fetch_markets(sport=sport, page=page, per_page=per_page)
        page_outcomes = flatten_outcomes(data)
        if not page_outcomes:
            break
        all_outcomes.extend(page_outcomes)
    return dedupe_outcomes(all_outcomes)


def _is_1x2_outcome(outcome: dict[str, Any]) -> bool:
    name = str(outcome.get("name") or "").strip().lower()
    market_name = str(outcome.get("market_name") or "").strip().lower()

    if name in {"1", "x", "2", "home", "draw", "away"}:
        return True

    return "1x2" in market_name or "match result" in market_name


def pick_first_low_odds(
    outcomes: list[dict[str, Any]],
    max_count: int,
    max_odds: float,
    min_odds: float = 1.01,
    only_1x2: bool = True,
) -> list[dict[str, Any]]:
    by_match: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    selected: list[dict[str, Any]] = []

    ordered = sorted(
        outcomes,
        key=lambda row: (
            int(row.get("match_order") or 0),
            str(row.get("match_id") or ""),
            int(row.get("outcome_order") or 0),
        ),
    )

    for row in ordered:
        match_key = str(row.get("match_id") or "")
        if not match_key:
            continue
        by_match.setdefault(match_key, []).append(row)

    for rows in by_match.values():
        if len(selected) >= max_count:
            break

        candidate: dict[str, Any] | None = None
        for row in rows:
            odd = row.get("odd")
            if odd is None:
                continue
            if odd < min_odds or odd > max_odds:
                continue
            if only_1x2 and not _is_1x2_outcome(row):
                continue
            candidate = row
            break

        if candidate:
            selected.append(candidate)

    return selected


def print_selected(selected: list[dict[str, Any]]) -> None:
    if not selected:
        print("No selections were auto-picked.")
        return

    print("Auto-picked selections:")
    print("odd    match_id  selection_id  teams  outcome")
    print("-" * 90)
    for row in selected:
        odd = "?" if row.get("odd") is None else f"{row['odd']:.2f}"
        teams = f"{row.get('home', '?')} vs {row.get('away', '?')}"
        print(
            f"{odd:<6} {str(row.get('match_id', '?')):<9} "
            f"{str(row.get('selection_id', '?')):<12} {teams:<35} {row.get('name', '?')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Betika API helper: list odds and place bets (dry-run by default)."
    )
    parser.add_argument("--base-url", default=os.getenv("BETIKA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--csrf-token", default=os.getenv("BETIKA_CSRF_TOKEN"))
    parser.add_argument("--bearer-token", default=os.getenv("BETIKA_BEARER_TOKEN"))

    parser.add_argument("--login-path", default=os.getenv("BETIKA_LOGIN_PATH", "/api/v1/login"))
    parser.add_argument(
        "--markets-path",
        default=os.getenv("BETIKA_MARKETS_PATH", "/api/v1/prematch/matches"),
    )
    parser.add_argument(
        "--place-bet-path",
        default=os.getenv("BETIKA_PLACE_BET_PATH", "/api/v1/bets/place"),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_cmd = subparsers.add_parser("list", help="Fetch and display available outcomes")
    list_cmd.add_argument("--sport", default="soccer")
    list_cmd.add_argument("--page", type=int, default=1)
    list_cmd.add_argument("--per-page", type=int, default=100)
    list_cmd.add_argument("--max-odds", type=float, default=None)
    list_cmd.add_argument("--save-json", default=None, help="Optional file to save flattened outcomes")

    bet_cmd = subparsers.add_parser("bet", help="Login and place a single bet")
    bet_cmd.add_argument("--username", default=os.getenv("BETIKA_USERNAME"))
    bet_cmd.add_argument("--password", default=os.getenv("BETIKA_PASSWORD"))
    bet_cmd.add_argument(
        "--login-template",
        default='{"mobile":"{username}","password":"{password}"}',
        help="JSON template for login payload. Use {username} and {password} placeholders.",
    )
    bet_cmd.add_argument("--sport", default="soccer")
    bet_cmd.add_argument("--page", type=int, default=1)
    bet_cmd.add_argument("--per-page", type=int, default=100)
    bet_cmd.add_argument("--stake", type=float, required=True)
    bet_cmd.add_argument("--currency", default="KES")
    bet_cmd.add_argument("--selection-id", default=None)
    bet_cmd.add_argument("--match-id", default=None)
    bet_cmd.add_argument("--outcome-id", default=None)
    bet_cmd.add_argument("--odd", type=float, default=None, help="Override odd in the payload")
    bet_cmd.add_argument(
        "--extra-bet-payload",
        default=None,
        help="Optional JSON object merged at top-level into bet payload.",
    )
    bet_cmd.add_argument(
        "--execute",
        action="store_true",
        help="Actually place the bet. Without this flag, the script prints payload only.",
    )

    auto_cmd = subparsers.add_parser(
        "auto-bet",
        help="Login, auto-pick first N low-odds selections, and place one multi-bet",
    )
    auto_cmd.add_argument("--username", default=os.getenv("BETIKA_USERNAME"))
    auto_cmd.add_argument("--password", default=os.getenv("BETIKA_PASSWORD"))
    auto_cmd.add_argument(
        "--login-template",
        default='{"mobile":"{username}","password":"{password}"}',
        help="JSON template for login payload. Use {username} and {password} placeholders.",
    )
    auto_cmd.add_argument("--sport", default="soccer")
    auto_cmd.add_argument("--page", type=int, default=1, help="First page to scan from")
    auto_cmd.add_argument("--pages", type=int, default=5, help="How many pages to scan")
    auto_cmd.add_argument("--per-page", type=int, default=100)
    auto_cmd.add_argument("--count", type=int, default=30, help="Maximum selections to pick")
    auto_cmd.add_argument("--max-odds", type=float, default=1.5)
    auto_cmd.add_argument("--min-odds", type=float, default=1.01)
    auto_cmd.add_argument(
        "--all-markets",
        action="store_true",
        help="Pick from all markets. By default only 1X2-like outcomes are considered.",
    )
    auto_cmd.add_argument(
        "--stake",
        type=float,
        default=100.0,
        help="Stake amount. Defaults to 100.0 for no-subcommand runs.",
    )
    auto_cmd.add_argument("--currency", default="KES")
    auto_cmd.add_argument("--save-picks", default=None, help="Optional JSON file for picked selections")
    auto_cmd.add_argument(
        "--extra-bet-payload",
        default=None,
        help="Optional JSON object merged at top-level into bet payload.",
    )
    auto_cmd.add_argument(
        "--execute",
        action="store_true",
        help="Actually place the bet. Without this flag, the script prints payload only.",
    )

    return parser.parse_args()


def run_list(client: BetikaClient, args: argparse.Namespace) -> int:
    markets = client.fetch_markets(sport=args.sport, page=args.page, per_page=args.per_page)
    outcomes = flatten_outcomes(markets)

    print_outcomes(outcomes, max_odds=args.max_odds)

    if args.save_json:
        save_json(outcomes, args.save_json)
        print(f"Saved {len(outcomes)} outcomes to {args.save_json}")

    return 0


def _assert_credentials(username: str | None, password: str | None) -> None:
    if not username or not password:
        raise BetikaApiError(
            "Missing credentials. Provide --username/--password or set BETIKA_USERNAME/BETIKA_PASSWORD."
        )


def _login_with_template(client: BetikaClient, args: argparse.Namespace) -> dict[str, Any] | list[Any]:
    _assert_credentials(args.username, args.password)

    try:
        login_template = json.loads(args.login_template)
    except json.JSONDecodeError as exc:
        raise BetikaApiError(f"Invalid --login-template JSON: {exc}") from exc

    login_payload = _format_payload_template(login_template, args.username, args.password)
    login_response = client.login(login_payload)
    print("Login request sent.")
    return login_response


def _merge_extra_bet_payload(args: argparse.Namespace, bet_payload: dict[str, Any]) -> dict[str, Any]:
    if not args.extra_bet_payload:
        return bet_payload

    try:
        extra_payload = json.loads(args.extra_bet_payload)
    except json.JSONDecodeError as exc:
        raise BetikaApiError(f"Invalid --extra-bet-payload JSON: {exc}") from exc
    if not isinstance(extra_payload, dict):
        raise BetikaApiError("--extra-bet-payload must be a JSON object.")

    merged = dict(bet_payload)
    merged.update(extra_payload)
    return merged


def _submit_or_preview_bet(
    client: BetikaClient, bet_payload: dict[str, Any], execute: bool
) -> dict[str, Any] | list[Any]:
    response = client.place_bet(payload=bet_payload, dry_run=not execute)
    if isinstance(response, dict) and response.get("dry_run"):
        print("Dry-run mode: no live bet submitted. Use --execute to place the bet.")
        print(json.dumps(response["payload"], indent=2))
    else:
        print("Bet request submitted.")
        print(json.dumps(response, indent=2))
    return response


def run_bet(client: BetikaClient, args: argparse.Namespace) -> int:
    if args.stake <= 0:
        raise BetikaApiError("Stake must be greater than 0.")

    login_response = _login_with_template(client, args)

    markets = client.fetch_markets(sport=args.sport, page=args.page, per_page=args.per_page)
    outcomes = flatten_outcomes(markets)

    selection = find_selection(
        outcomes,
        selection_id=args.selection_id,
        match_id=args.match_id,
        outcome_id=args.outcome_id,
    )

    if not selection:
        raise BetikaApiError(
            "Selection not found. Provide --selection-id or both --match-id and --outcome-id."
        )

    if args.odd is not None:
        selection["odd"] = args.odd

    bet_payload = build_bet_payload(selection, stake=args.stake, currency=args.currency)
    bet_payload = _merge_extra_bet_payload(args, bet_payload)
    _submit_or_preview_bet(client, bet_payload, execute=args.execute)

    if os.getenv("BETIKA_DEBUG_LOGIN_RESPONSE") == "1":
        print("Login response (debug):")
        print(json.dumps(login_response, indent=2))

    return 0


def run_auto_bet(client: BetikaClient, args: argparse.Namespace) -> int:
    if args.stake <= 0:
        raise BetikaApiError("Stake must be greater than 0.")
    if args.count <= 0:
        raise BetikaApiError("--count must be greater than 0.")
    if args.pages <= 0:
        raise BetikaApiError("--pages must be greater than 0.")
    if args.max_odds <= 0:
        raise BetikaApiError("--max-odds must be greater than 0.")
    if args.min_odds <= 0:
        raise BetikaApiError("--min-odds must be greater than 0.")
    if args.min_odds > args.max_odds:
        raise BetikaApiError("--min-odds must be less than or equal to --max-odds.")

    login_response = _login_with_template(client, args)

    outcomes = fetch_outcomes_pages(
        client=client,
        sport=args.sport,
        start_page=args.page,
        pages=args.pages,
        per_page=args.per_page,
    )
    selected = pick_first_low_odds(
        outcomes=outcomes,
        max_count=args.count,
        max_odds=args.max_odds,
        min_odds=args.min_odds,
        only_1x2=not args.all_markets,
    )

    if not selected:
        raise BetikaApiError(
            "No matching selections found for the configured odds range and market filter."
        )

    print_selected(selected)
    print(f"Picked {len(selected)} selections (target {args.count}).")

    if args.save_picks:
        save_json(selected, args.save_picks)
        print(f"Saved picked selections to {args.save_picks}")

    bet_payload = build_multi_bet_payload(selected, stake=args.stake, currency=args.currency)
    bet_payload = _merge_extra_bet_payload(args, bet_payload)
    _submit_or_preview_bet(client, bet_payload, execute=args.execute)

    if os.getenv("BETIKA_DEBUG_LOGIN_RESPONSE") == "1":
        print("Login response (debug):")
        print(json.dumps(login_response, indent=2))

    return 0


def main() -> int:
    if len(sys.argv) == 1:
        sys.argv.append("auto-bet")

    args = parse_args()

    paths = ApiPaths(
        login=args.login_path,
        markets=args.markets_path,
        place_bet=args.place_bet_path,
    )
    client = BetikaClient(
        base_url=args.base_url,
        paths=paths,
        timeout=args.timeout,
        csrf_token=args.csrf_token,
        bearer_token=args.bearer_token,
    )

    try:
        if args.command == "list":
            return run_list(client, args)
        if args.command == "bet":
            return run_bet(client, args)
        if args.command == "auto-bet":
            return run_auto_bet(client, args)
        raise BetikaApiError(f"Unknown command: {args.command}")
    except BetikaApiError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
