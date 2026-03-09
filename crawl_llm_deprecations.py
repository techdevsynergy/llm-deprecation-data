#!/usr/bin/env python3
"""
Crawler + merger for llm_deprecation_data.json.

This script:
1) Crawls fixed sources from crawl_sources.json.
2) Extracts provider/model lifecycle signals from tables and page text.
3) Merges only lifecycle fields into the existing dataset, without deleting
   rows that are not mentioned in the crawl.
4) Writes a crawl report and merged output file.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests


USER_AGENT = "LLMDeprecationCrawler/1.0 (compliance)"
REQUEST_TIMEOUT_SECONDS = 30
HOST_DELAY_SECONDS = 1.0
FETCH_FALLBACK_STATUSES = {401, 403, 429}

ALLOWED_STATUSES = {"active", "deprecated", "legacy", "retired"}
STATUS_PRIORITY = {"active": 0, "legacy": 1, "deprecated": 2, "retired": 3}

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MODEL_REGEX_BY_PROVIDER = {
    "openai": re.compile(
        r"\b(?:gpt|o[0-9]|text-embedding|text-moderation|omni-moderation|"
        r"whisper|dall-e|sora|babbage|davinci|codex)"
        r"[a-z0-9._:@-]*\b",
        re.IGNORECASE,
    ),
    "anthropic": re.compile(r"\bclaude-[a-z0-9.-]+\b", re.IGNORECASE),
    "gemini": re.compile(
        r"\b(?:gemini|veo|nano-banana|imagen|imagetext|virtual-try-on|"
        r"textembedding-gecko|imagegeneration|text-bison|chat-bison|code-gecko|"
        r"gemini-embedding|text-embedding)"
        r"[a-z0-9._:@-]*\*?\b",
        re.IGNORECASE,
    ),
}

REPLACEMENT_PATTERNS = [
    re.compile(
        r"(?:migrate to|use|replacement(?: model)?|recommended replacement|"
        r"recommended upgrade|successor|superseded by|instead(?: use)?|replace(?:d)? by)\s+"
        r"(?P<value>[a-z0-9][a-z0-9._:@-]*(?:\s+or\s+[a-z0-9][a-z0-9._:@-]*)*)",
        re.IGNORECASE,
    ),
]


def normalize_model_id(token: str) -> str:
    cleaned = token.replace("‑", "-").replace("–", "-").replace("—", "-")
    cleaned = cleaned.strip().strip("`'\"()[]{}.,;:").lower()
    cleaned = cleaned.rstrip("*")
    return cleaned


def is_probable_model_id(provider: str, token: str) -> bool:
    token = normalize_model_id(token)
    if not token or token in {"claude", "gemini", "gpt", "model", "models"}:
        return False
    if provider == "openai":
        return bool(re.match(r"^(gpt|o[0-9]|text-|omni-|whisper|dall-e|sora|babbage|davinci|codex)", token))
    if provider == "anthropic":
        return bool(re.match(r"^claude-[a-z0-9.-]+$", token))
    if provider == "gemini":
        return bool(
            re.match(
                r"^(gemini-|veo-|nano-banana|imagen|imagetext|virtual-try-on|"
                r"textembedding-gecko|imagegeneration|text-bison|chat-bison|"
                r"code-gecko|gemini-embedding|text-embedding)",
                token,
            )
        )
    return False


def extract_model_ids(text: str, provider: str) -> List[str]:
    regex = MODEL_REGEX_BY_PROVIDER[provider]
    out = []
    for match in regex.findall(text):
        model_id = normalize_model_id(match)
        if is_probable_model_id(provider, model_id):
            out.append(model_id)
    # Preserve order while deduplicating.
    seen = set()
    deduped = []
    for mid in out:
        if mid not in seen:
            seen.add(mid)
            deduped.append(mid)
    return deduped


def parse_date_yyyy_mm_dd(text: str) -> Optional[str]:
    if not text:
        return None
    s = (
        text.replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("\u00a0", " ")
    )
    s = " ".join(s.strip().split())
    sl = s.lower()
    if any(x in sl for x in ["n/a", "no retirement date announced", "unknown", "tbd"]):
        return None

    # YYYY-MM-DD
    m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Month DD, YYYY  OR Month DD YYYY
    m = re.search(
        r"\b(" + "|".join(MONTHS.keys()) + r")\s+(\d{1,2})(?:,)?\s+(20\d{2})\b",
        sl,
    )
    if m:
        mo = MONTHS[m.group(1)]
        d = int(m.group(2))
        y = int(m.group(3))
        try:
            return date(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # DD Month YYYY
    m = re.search(
        r"\b(\d{1,2})\s+(" + "|".join(MONTHS.keys()) + r")(?:,)?\s+(20\d{2})\b",
        sl,
    )
    if m:
        d = int(m.group(1))
        mo = MONTHS[m.group(2)]
        y = int(m.group(3))
        try:
            return date(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None


def choose_status(text: str, sunset_date: Optional[str]) -> str:
    l = text.lower()
    status = "active"
    if "retired" in l or "no longer available" in l:
        status = "retired"
    elif "legacy" in l:
        status = "legacy"
    elif any(k in l for k in ["deprecated", "deprecation", "sunset", "discontinuation", "retirement", "removed"]):
        status = "deprecated"

    if sunset_date:
        today_s = date.today().strftime("%Y-%m-%d")
        if sunset_date < today_s:
            status = "retired"
    return status


def extract_replacement(text: str, provider: str) -> Optional[str]:
    for pattern in REPLACEMENT_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        raw = m.group("value").strip().strip(".")
        # Keep "x or y" when present; otherwise normalize single model ID.
        if " or " in raw.lower():
            parts = [normalize_model_id(p) for p in re.split(r"\s+or\s+", raw, flags=re.IGNORECASE)]
            parts = [p for p in parts if is_probable_model_id(provider, p)]
            if parts:
                return " or ".join(parts)
        else:
            cand = normalize_model_id(raw)
            if is_probable_model_id(provider, cand):
                return cand
    return None


def later_date(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a and b:
        return a if a > b else b
    return a or b


def parse_robots_txt(text: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse robots.txt into:
      {user_agent_token_lower: [(directive, path), ...], ...}
    """
    groups: Dict[str, List[Tuple[str, str]]] = {}
    active_agents: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field, value = line.split(":", 1)
        field = field.strip().lower()
        value = value.strip()

        if field == "user-agent":
            token = value.lower()
            active_agents = [token]
            groups.setdefault(token, [])
            continue

        if field in {"allow", "disallow"}:
            if not active_agents:
                # Ignore malformed directives before any user-agent.
                continue
            for agent in active_agents:
                groups.setdefault(agent, []).append((field, value))

    return groups


def build_jina_reader_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return f"https://r.jina.ai/http://{parsed.netloc}{path}"


def parse_markdown_tables(markdown_text: str) -> List[List[List[str]]]:
    """
    Parse basic GitHub-style markdown tables into the same shape as HTML tables:
      [ [header_cells], [row_cells], ... ]
    """
    def split_row(raw: str) -> List[str]:
        parts = [p.strip() for p in raw.strip().strip("|").split("|")]
        return [re.sub(r"\s+", " ", p).strip() for p in parts]

    lines = markdown_text.splitlines()
    out: List[List[List[str]]] = []
    i = 0
    while i < len(lines):
        header_line = lines[i].strip()
        if not (header_line.startswith("|") and header_line.endswith("|")):
            i += 1
            continue
        if i + 1 >= len(lines):
            i += 1
            continue
        sep_line = lines[i + 1].strip()
        if not (sep_line.startswith("|") and sep_line.endswith("|")):
            i += 1
            continue

        sep_cells = [c.strip() for c in sep_line.strip("|").split("|")]
        if not sep_cells or not all(re.match(r"^:?-{3,}:?$", cell) for cell in sep_cells):
            i += 1
            continue

        header_cells = split_row(header_line)
        table: List[List[str]] = [header_cells]
        j = i + 2
        while j < len(lines):
            row_line = lines[j].strip()
            if not (row_line.startswith("|") and row_line.endswith("|")):
                break
            row_cells = split_row(row_line)
            if len(row_cells) < len(header_cells):
                row_cells.extend([""] * (len(header_cells) - len(row_cells)))
            table.append(row_cells[: len(header_cells)])
            j += 1

        if len(table) > 1:
            out.append(table)
            i = j
        else:
            i += 1
    return out


def robots_allows(user_agent: str, target_url: str, rules: Dict[str, List[Tuple[str, str]]]) -> bool:
    """
    Apply longest-match rule precedence:
      - pick rules for matching user-agent token groups; if none, fallback '*'
      - longest path match wins; allow wins ties
    """
    parsed = urlparse(target_url)
    target_path = parsed.path or "/"
    if parsed.query:
        target_path = f"{target_path}?{parsed.query}"

    ua = user_agent.lower()
    matched_tokens = [token for token in rules.keys() if token != "*" and token in ua]
    if not matched_tokens and "*" in rules:
        matched_tokens = ["*"]
    if not matched_tokens:
        return True

    best_len = -1
    best_decision = True

    for token in matched_tokens:
        for directive, path in rules.get(token, []):
            if directive == "disallow" and path == "":
                continue
            if not target_path.startswith(path):
                continue
            match_len = len(path)
            decision = directive == "allow"
            if match_len > best_len or (match_len == best_len and decision):
                best_len = match_len
                best_decision = decision

    return best_decision


class PageParser(HTMLParser):
    """Extracts plain text + HTML tables without external dependencies."""

    BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "li",
        "tr",
        "br",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "td",
        "th",
        "code",
        "pre",
    }

    def __init__(self) -> None:
        super().__init__()
        self.in_ignored_tag = 0
        self.text_parts: List[str] = []
        self.tables: List[List[List[str]]] = []
        self._in_table = False
        self._table: List[List[str]] = []
        self._in_row = False
        self._row: List[str] = []
        self._in_cell = False
        self._cell: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript"}:
            self.in_ignored_tag += 1
            return
        if self.in_ignored_tag:
            return
        if t in self.BLOCK_TAGS:
            self.text_parts.append("\n")
        if t == "table":
            self._in_table = True
            self._table = []
        elif t == "tr" and self._in_table:
            self._in_row = True
            self._row = []
        elif t in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._cell = []
        elif t == "br" and self._in_cell:
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript"}:
            self.in_ignored_tag = max(0, self.in_ignored_tag - 1)
            return
        if self.in_ignored_tag:
            return
        if t in self.BLOCK_TAGS:
            self.text_parts.append("\n")
        if t in {"td", "th"} and self._in_cell:
            cell = html.unescape("".join(self._cell))
            cell = re.sub(r"\s+", " ", cell).strip()
            self._row.append(cell)
            self._in_cell = False
        elif t == "tr" and self._in_row:
            if any(c.strip() for c in self._row):
                self._table.append(self._row)
            self._in_row = False
        elif t == "table" and self._in_table:
            if self._table:
                self.tables.append(self._table)
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self.in_ignored_tag:
            return
        self.text_parts.append(data)
        if self._in_cell:
            self._cell.append(data)

    def text(self) -> str:
        text = html.unescape("".join(self.text_parts))
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text


def merge_candidate(
    store: Dict[Tuple[str, str], Dict[str, object]],
    provider: str,
    model_id: str,
    status: str,
    deprecated_date: Optional[str],
    sunset_date: Optional[str],
    replacement: Optional[str],
    notes: str,
    source_url: str,
) -> None:
    key = (provider, model_id)
    existing = store.get(key)
    if existing is None:
        store[key] = {
            "provider": provider,
            "model_id": model_id,
            "status": status if status in ALLOWED_STATUSES else "active",
            "deprecated_date": deprecated_date,
            "sunset_date": sunset_date,
            "replacement": replacement,
            "notes": notes,
            "_source_urls": {source_url},
        }
        return

    if STATUS_PRIORITY.get(status, 0) > STATUS_PRIORITY.get(existing["status"], 0):
        existing["status"] = status
    if deprecated_date:
        existing["deprecated_date"] = later_date(existing["deprecated_date"], deprecated_date)
    if sunset_date:
        existing["sunset_date"] = later_date(existing["sunset_date"], sunset_date)
    if replacement and not existing["replacement"]:
        existing["replacement"] = replacement
    if notes and ("crawl" in str(existing["notes"]).lower() or not existing["notes"]):
        existing["notes"] = notes
    existing["_source_urls"].add(source_url)


def parse_tables(
    provider: str,
    url: str,
    tables: List[List[List[str]]],
    text_blob: str,
    out: Dict[Tuple[str, str], Dict[str, object]],
    parse_warnings: List[str],
) -> None:
    for table in tables:
        if not table:
            continue
        header = [c.lower() for c in table[0]]
        header_str = " | ".join(header)

        # Anthropic deprecations history table:
        # Retirement Date | Deprecated Model | Recommended Replacement
        if "deprecated model" in header_str and ("retirement date" in header_str or "discontinuation" in header_str):
            date_idx = next((i for i, h in enumerate(header) if "retirement" in h or "discontinuation" in h or "sunset" in h), None)
            model_idx = next((i for i, h in enumerate(header) if "deprecated model" in h or "model id" in h or h == "model"), None)
            repl_idx = next((i for i, h in enumerate(header) if "replacement" in h or "upgrade" in h), None)
            for row in table[1:]:
                if model_idx is None or model_idx >= len(row):
                    continue
                model_text = row[model_idx]
                model_ids = extract_model_ids(model_text, provider)
                if not model_ids:
                    continue
                sunset = parse_date_yyyy_mm_dd(row[date_idx]) if date_idx is not None and date_idx < len(row) else None
                replacement = row[repl_idx] if repl_idx is not None and repl_idx < len(row) else ""
                replacement_norm = extract_replacement("replacement " + replacement, provider) or (
                    " or ".join(extract_model_ids(replacement, provider)) if extract_model_ids(replacement, provider) else None
                )
                for model_id in model_ids:
                    status = "retired" if sunset and sunset < date.today().strftime("%Y-%m-%d") else "deprecated"
                    note = f"Crawled from {url}: deprecated model lifecycle table."
                    merge_candidate(
                        store=out,
                        provider=provider,
                        model_id=model_id,
                        status=status,
                        deprecated_date=None,
                        sunset_date=sunset,
                        replacement=replacement_norm,
                        notes=note,
                        source_url=url,
                    )
            continue

        # Anthropic current model-state table:
        # API Model Name | Current State | Deprecated | Tentative Retirement Date
        if "current state" in header_str and ("api model name" in header_str or "model name" in header_str):
            model_idx = next((i for i, h in enumerate(header) if "model" in h), 0)
            state_idx = next((i for i, h in enumerate(header) if "current state" in h), None)
            dep_idx = next((i for i, h in enumerate(header) if h.strip() == "deprecated" or "deprecated" in h), None)
            retirement_idx = next((i for i, h in enumerate(header) if "retirement" in h), None)
            for row in table[1:]:
                if model_idx >= len(row):
                    continue
                model_ids = extract_model_ids(row[model_idx], provider)
                if not model_ids:
                    continue
                state_text = row[state_idx] if state_idx is not None and state_idx < len(row) else ""
                deprecated_date = row[dep_idx] if dep_idx is not None and dep_idx < len(row) else ""
                retirement_text = row[retirement_idx] if retirement_idx is not None and retirement_idx < len(row) else ""
                status = choose_status(state_text, parse_date_yyyy_mm_dd(retirement_text))
                for model_id in model_ids:
                    merge_candidate(
                        store=out,
                        provider=provider,
                        model_id=model_id,
                        status=status,
                        deprecated_date=parse_date_yyyy_mm_dd(deprecated_date),
                        sunset_date=parse_date_yyyy_mm_dd(retirement_text) if status != "active" else None,
                        replacement=None,
                        notes=f"Crawled from {url}: model status table.",
                        source_url=url,
                    )
            continue

        # OpenAI or Gemini style lifecycle table:
        # Model ID | Release date | Retirement date | Recommended upgrade
        if (
            ("recommended upgrade" in header_str and "retirement date" in header_str)
            or ("shutdown date" in header_str and "recommended replacement" in header_str)
        ):
            model_idx = next(
                (i for i, h in enumerate(header) if "model id" in h or "model / system" in h or h.strip() == "model"),
                0,
            )
            retirement_idx = next(
                (
                    i
                    for i, h in enumerate(header)
                    if "retirement date" in h or "discontinuation date" in h or "shutdown date" in h or "sunset" in h
                ),
                None,
            )
            repl_idx = next(
                (i for i, h in enumerate(header) if "recommended upgrade" in h or "replacement" in h),
                None,
            )
            for row in table[1:]:
                if model_idx >= len(row):
                    continue
                model_ids = extract_model_ids(row[model_idx], provider)
                if not model_ids:
                    continue
                retirement = row[retirement_idx] if retirement_idx is not None and retirement_idx < len(row) else ""
                sunset = parse_date_yyyy_mm_dd(retirement)
                replacement = row[repl_idx] if repl_idx is not None and repl_idx < len(row) else ""
                replacement_norm = extract_replacement("recommended upgrade " + replacement, provider) or (
                    " or ".join(extract_model_ids(replacement, provider)) if extract_model_ids(replacement, provider) else None
                )
                for model_id in model_ids:
                    status = choose_status("deprecated retirement", sunset)
                    merge_candidate(
                        store=out,
                        provider=provider,
                        model_id=model_id,
                        status=status,
                        deprecated_date=None,
                        sunset_date=sunset,
                        replacement=replacement_norm,
                        notes=f"Crawled from {url}: lifecycle table.",
                        source_url=url,
                    )
            continue

        # Gemini key-value model details table:
        # row0: Model ID | <actual-id>
        if len(table) > 2 and len(table[0]) >= 2 and table[0][0].strip().lower() == "model id":
            model_id = normalize_model_id(table[0][1])
            if not is_probable_model_id(provider, model_id):
                continue
            details = {}
            for row in table[1:]:
                if len(row) >= 2:
                    details[row[0].strip().lower()] = row[1]
            date_val = details.get("discontinuation date") or details.get("retirement date")
            sunset = parse_date_yyyy_mm_dd(date_val or "")
            if sunset:
                status = "retired" if sunset < date.today().strftime("%Y-%m-%d") else "deprecated"
                replacement = None
                model_mention = re.search(
                    re.escape(model_id) + r".{0,300}",
                    text_blob,
                    re.IGNORECASE | re.DOTALL,
                )
                if model_mention:
                    replacement = extract_replacement(model_mention.group(0), provider)
                merge_candidate(
                    store=out,
                    provider=provider,
                    model_id=model_id,
                    status=status,
                    deprecated_date=None,
                    sunset_date=sunset,
                    replacement=replacement,
                    notes=f"Crawled from {url}: model details table.",
                    source_url=url,
                )
            continue

    # Targeted text extraction for explicit lifecycle sentences only.
    # Example:
    # "gemini-live-... will be deprecated and removed on March 19, 2026."
    explicit = re.compile(
        r"(?P<model>[a-z0-9][a-z0-9._:@-]{2,})\s+"
        r"(?:will be|is|was)\s+"
        r"(?:deprecated(?:\s+and\s+removed)?|retired|removed|discontinued)"
        r"[^.]{0,180}?\s(?:on|by)\s"
        r"(?P<date>(?:[A-Za-z]+\s+\d{1,2},\s+20\d{2}|20\d{2}-\d{2}-\d{2}))",
        re.IGNORECASE,
    )
    for match in explicit.finditer(text_blob):
        model_id = normalize_model_id(match.group("model"))
        if not is_probable_model_id(provider, model_id):
            continue
        sunset = parse_date_yyyy_mm_dd(match.group("date"))
        if not sunset:
            parse_warnings.append(f"{provider}:{model_id} explicit lifecycle sentence missing date ({url})")
            continue
        sentence = text_blob[match.start(): min(len(text_blob), match.end() + 240)]
        replacement = extract_replacement(sentence, provider)
        status = "retired" if sunset < date.today().strftime("%Y-%m-%d") else "deprecated"
        merge_candidate(
            store=out,
            provider=provider,
            model_id=model_id,
            status=status,
            deprecated_date=None,
            sunset_date=sunset,
            replacement=replacement,
            notes=f"Crawled from {url}: explicit lifecycle text sentence.",
            source_url=url,
        )


def crawl_sources(sources: Dict[str, List[str]]) -> Tuple[Dict[Tuple[str, str], Dict[str, object]], List[Dict[str, object]], List[str]]:
    results: Dict[Tuple[str, str], Dict[str, object]] = {}
    url_logs: List[Dict[str, object]] = []
    parse_warnings: List[str] = []

    robots_cache: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    robots_errors: Dict[str, Optional[str]] = {}
    host_last_request_ts: Dict[str, float] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for provider, urls in sources.items():
        for url in urls:
            parsed = urlparse(url)
            host = parsed.netloc
            robots_allowed = True
            robots_error = None
            if host not in robots_cache:
                robots_url = f"{parsed.scheme}://{host}/robots.txt"
                try:
                    robots_resp = session.get(robots_url, timeout=REQUEST_TIMEOUT_SECONDS)
                    if robots_resp.status_code >= 400:
                        robots_cache[host] = {}
                        robots_errors[host] = f"robots HTTP {robots_resp.status_code}; treated as allow"
                    else:
                        robots_cache[host] = parse_robots_txt(robots_resp.text)
                        robots_errors[host] = None
                except Exception as exc:
                    robots_cache[host] = {}
                    robots_errors[host] = f"robots read failed: {exc}; treated as allow"
            rules = robots_cache[host]
            robots_error = robots_errors.get(host)
            try:
                robots_allowed = True if not rules else robots_allows(USER_AGENT, url, rules)
            except Exception as exc:
                robots_allowed = True
                robots_error = f"robots parse failed: {exc}"

            if not robots_allowed:
                url_logs.append(
                    {
                        "provider": provider,
                        "url": url,
                        "http_status": None,
                        "error": "blocked_by_robots",
                        "robots_error": robots_error,
                        "records_extracted": 0,
                    }
                )
                continue

            last_ts = host_last_request_ts.get(host)
            if last_ts is not None:
                since = time.time() - last_ts
                if since < HOST_DELAY_SECONDS:
                    time.sleep(HOST_DELAY_SECONDS - since)

            before_count = len(results)
            http_status = None
            final_url = None
            fallback_url = None
            fetched_via = "direct"
            error = None
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
                http_status = resp.status_code
                final_url = resp.url
                response_text = resp.text
                host_last_request_ts[host] = time.time()
                if resp.status_code in FETCH_FALLBACK_STATUSES:
                    fallback_url = build_jina_reader_url(url)
                    fb_resp = session.get(fallback_url, timeout=REQUEST_TIMEOUT_SECONDS)
                    if fb_resp.status_code < 400:
                        fetched_via = "jina_reader"
                        http_status = fb_resp.status_code
                        final_url = fb_resp.url
                        response_text = fb_resp.text
                    else:
                        raise requests.HTTPError(f"HTTP {resp.status_code}")
                elif resp.status_code >= 400:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                parser = PageParser()
                parser.feed(response_text)
                text_blob = parser.text()
                parsed_tables = list(parser.tables)
                parsed_tables.extend(parse_markdown_tables(response_text))
                parse_tables(
                    provider,
                    url if fetched_via != "direct" else (final_url or url),
                    parsed_tables,
                    text_blob or response_text,
                    results,
                    parse_warnings,
                )
            except Exception as exc:
                error = str(exc)

            after_count = len(results)
            url_logs.append(
                {
                    "provider": provider,
                    "url": url,
                    "final_url": final_url,
                    "http_status": http_status,
                    "fallback_url": fallback_url,
                    "fetched_via": fetched_via,
                    "error": error,
                    "robots_error": robots_error,
                    "records_extracted": max(0, after_count - before_count),
                }
            )
    return results, url_logs, parse_warnings


def merge_with_existing(
    existing_rows: List[Dict[str, object]],
    crawl_rows: Dict[Tuple[str, str], Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, int]]]:
    merged_rows = [dict(r) for r in existing_rows]
    key_to_index = {(r["provider"], r["model_id"]): i for i, r in enumerate(merged_rows)}
    touched_existing_keys = set()

    summary = {
        provider: {"updated": 0, "added": 0, "unchanged": 0}
        for provider in ["openai", "anthropic", "gemini"]
    }

    for key, crawl_row in crawl_rows.items():
        provider, model_id = key
        if key in key_to_index:
            touched_existing_keys.add(key)
            row = merged_rows[key_to_index[key]]
            changed = False
            for field in ("status",):
                new_val = crawl_row.get(field)
                if new_val in ALLOWED_STATUSES and row.get(field) != new_val:
                    row[field] = new_val
                    changed = True
            for field in ("deprecated_date", "sunset_date", "replacement"):
                new_val = crawl_row.get(field)
                if new_val is not None and row.get(field) != new_val:
                    row[field] = new_val
                    changed = True
            crawl_note = str(crawl_row.get("notes") or "").strip()
            if crawl_note and not str(row.get("notes") or "").strip():
                row["notes"] = crawl_note
                changed = True
            if changed:
                summary[provider]["updated"] += 1
            else:
                summary[provider]["unchanged"] += 1
            merged_rows[key_to_index[key]] = row
        else:
            crawl_note = str(crawl_row.get("notes") or "").strip()
            new_row = {
                "provider": provider,
                "model_id": model_id,
                "status": crawl_row.get("status") if crawl_row.get("status") in ALLOWED_STATUSES else "active",
                "deprecated_date": crawl_row.get("deprecated_date"),
                "sunset_date": crawl_row.get("sunset_date"),
                "replacement": crawl_row.get("replacement"),
                "notes": (
                    f"Added by crawl; verify. {crawl_note}".strip()
                    if crawl_note
                    else "Added by crawl; verify."
                ),
            }
            key_to_index[key] = len(merged_rows)
            merged_rows.append(new_row)
            summary[provider]["added"] += 1

    # Rows not mentioned by crawl: keep exactly as-is.
    for provider, model_id in key_to_index.keys():
        if (provider, model_id) not in crawl_rows and (provider, model_id) not in touched_existing_keys:
            summary[provider]["unchanged"] += 1

    return merged_rows, summary


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
        f.write("\n")


def write_dataset_json(path: Path, rows: List[Dict[str, object]]) -> None:
    """
    Preserve compact one-row-per-line format used by llm_deprecation_data.json.
    """
    lines = ["["]
    for idx, row in enumerate(rows):
        row_json = json.dumps(row, ensure_ascii=True, separators=(", ", ": "))
        trailing = "," if idx < len(rows) - 1 else ""
        lines.append(f"  {row_json}{trailing}")
    lines.append("]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl provider lifecycle pages and merge deprecation data.")
    parser.add_argument("--sources", default="crawl_sources.json", help="Path to provider source URL config JSON.")
    parser.add_argument("--input", default="llm_deprecation_data.json", help="Existing deprecation data JSON file.")
    parser.add_argument("--output", default="llm_deprecation_data_crawled.json", help="Merged output JSON file.")
    parser.add_argument("--report", default="llm_deprecation_crawl_report.json", help="Crawl report JSON output.")
    parser.add_argument("--apply", action="store_true", help="Also overwrite --input with merged output.")
    args = parser.parse_args()

    sources_path = Path(args.sources)
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    sources = read_json(sources_path)
    existing_rows = read_json(input_path)
    crawl_rows, url_logs, parse_warnings = crawl_sources(sources)
    merged_rows, summary = merge_with_existing(existing_rows, crawl_rows)

    # Strip private fields before writing/reporting.
    cleaned_crawl_rows = []
    for row in crawl_rows.values():
        r = {k: v for k, v in row.items() if not k.startswith("_")}
        cleaned_crawl_rows.append(r)
    cleaned_crawl_rows.sort(key=lambda r: (r["provider"], r["model_id"]))

    write_dataset_json(output_path, merged_rows)
    if args.apply:
        write_dataset_json(input_path, merged_rows)

    report = {
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user_agent": USER_AGENT,
        "sources": sources,
        "url_results": url_logs,
        "crawl_record_count": len(cleaned_crawl_rows),
        "crawl_records": cleaned_crawl_rows,
        "provider_summary": summary,
        "parse_warnings": sorted(set(parse_warnings)),
    }
    write_json(report_path, report)

    print(f"Wrote merged output: {output_path}")
    if args.apply:
        print(f"Applied merged output to input file: {input_path}")
    print(f"Wrote crawl report: {report_path}")
    print("Provider summary:")
    for provider in ["openai", "anthropic", "gemini"]:
        s = summary[provider]
        print(f"  - {provider}: updated={s['updated']}, added={s['added']}, unchanged={s['unchanged']}")
    print(f"Crawl records: {len(cleaned_crawl_rows)}")
    print(f"Parse warnings: {len(report['parse_warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
