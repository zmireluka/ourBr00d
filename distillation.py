"""
Distillation — post-session self-critique for Mother.

Stage 1 — Judge:
    Reads Mother's system prompt + session transcript, extracts
    `what_worked` + `what_to_avoid` as JSON. Empty lists allowed.
    Appends findings to the bottom of the session transcript file
    (distillates/sessions/session_{id}.txt) — one file per session.

Stage 2 — Merger:
    Takes fresh findings + existing distillates/lessons.md, calls the
    merger LLM, writes updated lessons.md back.
    Hard cap per list: 4 critical + 4 notable. Severity-aware, no
    duplicates, no paraphrasing of existing entries.

Standalone CLI for testing against old transcripts:
    python3 distillation.py path/to/transcript.txt

Called from server.py at session end (awaited in the finally block).
"""

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import ollama

from mother_prompt import MOTHER_SYSTEM_PROMPT
from config import (
    DISTILL_JUDGE_NUM_CTX,
    DISTILL_JUDGE_NUM_PREDICT,
    DISTILL_JUDGE_TEMPERATURE,
    DISTILL_MERGER_NUM_CTX,
    DISTILL_MERGER_NUM_PREDICT,
    DISTILL_MERGER_TEMPERATURE,
    DISTILL_MODEL,
    DISTILLATES_DIR,
    LESSONS_PATH,
    MAX_CRITICAL_PER_LIST,
    MAX_FINDINGS_PER_LIST,
    MAX_NOTABLE_PER_LIST,
    MIN_TURNS_FOR_DISTILLATION,
    OLLAMA_HOST,
)

# All constants live in config.py. Edit there — both consumers
# (server.py + this file) pick up changes automatically.


# ── Judge prompt ──────────────────────────────────────────────────────────
# Severity thresholds made explicit so "critical" is not handed out
# too freely. Empty-list permission mentioned twice on purpose.

JUDGE_SYSTEM_PROMPT = """You evaluate Mother's performance in one conversation session.

Mother is an empathetic conversation companion. Her full operating instructions are provided below. Judge each of Mother's responses against those instructions and identify:

1. Responses that exemplified her intended character → `what_worked`
2. Responses that drifted from her intended character → `what_to_avoid`

Focus on Mother's lines (start with "Mother:"). The user lines are context.

Severity levels:
- "critical": breaks Mother's core identity. Examples: AI disclaimers ("as an AI", "I'm just a model"), therapist-mode, assistant-mode, lecturing, summarizing the user's words back at them, refusing to engage.
- "notable": stylistic drift, not catastrophic. Examples: response a bit too long, slightly preachy tone, missed an obvious follow-up question, awkward phrasing.

Picking examples:
- Surface the CLEAREST, most representative cases — not every minor issue.
- Quotes must be verbatim from Mother (not paraphrased).
- Reasons must be one short sentence.
- Max 3 items per list. Both lists may be empty.
- EMPTY IS ALLOWED. If Mother was consistently in character with no drift, return what_to_avoid: []. If nothing stood out as exemplary, return what_worked: []. Do NOT fabricate findings to fill slots.

Output ONLY valid JSON in this exact schema, no markdown, no commentary:

{
  "what_worked": [
    {"quote": "verbatim Mother text", "reason": "one sentence", "severity": "critical" | "notable"}
  ],
  "what_to_avoid": [
    {"quote": "verbatim Mother text", "reason": "one sentence", "severity": "critical" | "notable"}
  ]
}
"""


# Mother's system prompt is imported from mother_prompt.py above.
# server.py uses the same string live — the judge is therefore held to the
# exact same standard Mother is currently running against. One source, no drift.


# ── Merger prompt ─────────────────────────────────────────────────────────
# Lower temperature than the judge — merger should decide deterministically,
# not write creatively. Verbatim preservation of existing entries explicitly
# required, otherwise LLMs silently paraphrase and state drifts.

MERGER_SYSTEM_PROMPT = """You curate Mother's rolling list of behavioral lessons across sessions.

You receive two JSON inputs:
1. CURRENT_LESSONS — the existing curated lessons.
2. NEW_FINDINGS — fresh observations from the latest session (may have empty lists).

Decide what the UPDATED_LESSONS should be.

Rules:

1. Hard caps PER LIST:
   - max 4 entries with severity="critical"
   - max 4 entries with severity="notable"
   - so max 8 total entries per list

2. If BOTH NEW_FINDINGS lists are empty: return CURRENT_LESSONS unchanged.

3. NO duplicates. If a new finding addresses the SAME PATTERN as an existing entry (e.g. both about disclaimers, both about lecturing, both about summarizing), keep only ONE — the entry with the clearer, more representative quote. Do not keep both.

4. Severity priority: critical entries take precedence over notable. When a new critical entry needs to be added but the critical-cap is already full, replace the weakest existing critical (the one with the least clear quote). If the list is also at total-cap, drop the weakest notable to make room.

5. Do NOT rewrite or paraphrase existing entries. Keep them VERBATIM. The only operations allowed on existing entries are: keep them, drop them (when replaced or making room), or replace one entry by another entry's exact content.

6. Quotes MUST come verbatim from either CURRENT_LESSONS or NEW_FINDINGS. Never invent quotes. Never combine fragments.

Output ONLY valid JSON in this exact schema, no markdown, no commentary:

{
  "what_worked": [{"quote": "...", "reason": "...", "severity": "critical" | "notable"}],
  "what_to_avoid": [{"quote": "...", "reason": "...", "severity": "critical" | "notable"}]
}
"""


# ── Helpers ───────────────────────────────────────────────────────────────

def format_transcript(messages):
    """messages list → readable markdown transcript text.

    Format per turn:
        [Speaker A]: ...user text...

        Mother: ...mother text...

    System message is skipped — the judge receives it separately
    as its own system prompt context.
    """
    lines = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "system" or not content:
            continue
        if role == "user":
            # content already starts with "[Speaker X]: …" from server.py
            lines.append(content)
        elif role == "assistant":
            lines.append(f"Mother: {content}")
        lines.append("")  # blank line between turns
    return "\n".join(lines).strip()


def parse_transcript_file(path):
    """Reads a transcript written by server.py back into a messages list.
    Expected format:

        [USER]
        [Speaker A]: hello

        [ASSISTANT]
        Hi there

    Returns a list of role/content dicts — without system message.
    """
    text = Path(path).read_text(encoding="utf-8")
    messages = []
    current_role = None
    current_content = []

    def flush():
        if current_role and current_content:
            content = "\n".join(current_content).strip()
            if content:
                messages.append({"role": current_role, "content": content})

    for line in text.splitlines():
        if line.strip() == "[USER]":
            flush()
            current_role = "user"
            current_content = []
        elif line.strip() == "[ASSISTANT]":
            flush()
            current_role = "assistant"
            current_content = []
        else:
            if current_role is not None:
                current_content.append(line)

    flush()
    return messages


def count_turn_pairs(messages):
    """Number of complete user→assistant pairs. Incomplete turns
    (e.g. last user utterance without a Mother reply) are not counted."""
    n_user = sum(1 for m in messages if m["role"] == "user")
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")
    return min(n_user, n_assistant)


# ── lessons.md I/O ────────────────────────────────────────────────────────
# Parser + renderer for the rolling lessons.md. Source of truth is the file
# itself (git-tracked, hand-editable). On merge: read file → JSON repr →
# LLM merge → validate JSON → write back as markdown. This keeps the
# human-readable format stable while the LLM works in structured JSON.

# Regex for a bullet start: -  **"quote"** *(severity)*
# Allows single and double quotes in case someone hand-edits.
_BULLET_RE = re.compile(r'^- \*\*[\"\'](.+?)[\"\']\*\*\s*\*\((critical|notable)\)\*\s*$')


def parse_lessons_md(path):
    """Reads lessons.md back into a structure:
        {"what_worked": [{quote, reason, severity}, ...],
         "what_to_avoid": [...]}

    Tolerant: missing file or empty file → empty lists.
    Survives hand-edits as long as the bullet format stays intact.
    """
    path = Path(path)
    if not path.exists():
        return {"what_worked": [], "what_to_avoid": []}

    text = path.read_text(encoding="utf-8")

    # Delimit sections — cut at H2 headers.
    # ".*?" with DOTALL captures everything up to the next "## " or end of file.
    sections = {}
    for key, header in (("what_worked", "What worked"),
                        ("what_to_avoid", "What to avoid")):
        pat = re.compile(rf"##\s+{re.escape(header)}\s*\n(.*?)(?=^##\s|\Z)",
                         re.DOTALL | re.MULTILINE)
        m = pat.search(text)
        sections[key] = m.group(1) if m else ""

    state = {}
    for key, body in sections.items():
        state[key] = _parse_entries(body)
    return state


def _parse_entries(section_text):
    """Extract bullet entries from a section text."""
    entries = []
    current = None

    for line in section_text.splitlines():
        m = _BULLET_RE.match(line)
        if m:
            if current:
                entries.append(current)
            current = {
                "quote": m.group(1),
                "severity": m.group(2),   # guaranteed critical|notable by regex
                "reason": "",
            }
        elif current is not None and line.startswith("  ") and line.strip():
            # Indented continuation line = reason. Multiple reason lines are
            # joined with a space (should not occur, handled defensively).
            extra = line.strip()
            current["reason"] = (current["reason"] + " " + extra).strip()

    if current:
        entries.append(current)
    return entries


def render_lessons_for_prompt(state):
    """Compact variant for injection into Mother's system prompt.
    No headers, no meta text — just the lists. Returns empty string
    if both lists are empty (caller can then omit the whole block).
    """
    if not state.get("what_worked") and not state.get("what_to_avoid"):
        return ""

    def _line(entry):
        q = (entry.get("quote") or "").strip()
        r = (entry.get("reason") or "").strip()
        return f'- "{q}" — {r}' if r else f'- "{q}"'

    def _ordered(entries):
        # critical first, then notable — deterministic
        return ([e for e in entries if e.get("severity") == "critical"] +
                [e for e in entries if e.get("severity") != "critical"])

    parts = []
    if state.get("what_worked"):
        parts.append("What worked:")
        parts.extend(_line(e) for e in _ordered(state["what_worked"]))
    if state.get("what_to_avoid"):
        if parts:
            parts.append("")
        parts.append("What to avoid:")
        parts.extend(_line(e) for e in _ordered(state["what_to_avoid"]))
    return "\n".join(parts)


def render_lessons_md(state):
    """state dict → full markdown content for lessons.md.

    Order per list: all critical first, then all notable
    (deterministic, easier to read diffs).
    """
    lines = [
        "# Mother — Lessons across sessions",
        "",
        "Rolling, curated behavioral lessons distilled from past sessions by the Merger.",
        "Capped at 4 critical + 4 notable per list. Hand-edit allowed — format must stay parseable.",
        "",
    ]

    for header, key in (("What worked", "what_worked"),
                        ("What to avoid", "what_to_avoid")):
        lines.append(f"## {header}")
        lines.append("")
        entries = state.get(key, [])
        if not entries:
            lines.append("_(empty — fills up after first session ≥ 10 turns)_")
            lines.append("")
            continue

        ordered = ([e for e in entries if e.get("severity") == "critical"] +
                   [e for e in entries if e.get("severity") != "critical"])
        for entry in ordered:
            quote = (entry.get("quote") or "").strip()
            reason = (entry.get("reason") or "").strip()
            severity = entry.get("severity", "notable")
            lines.append(f'- **"{quote}"** *({severity})*')
            if reason:
                lines.append(f"  {reason}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ── Judge ─────────────────────────────────────────────────────────────────

async def judge_session(mother_system_prompt, transcript_text):
    """Calls the judge LLM, returns a findings dict.

    Returns:
        {"what_worked": [...], "what_to_avoid": [...]}
        On parse error additionally "_parse_error" + "_raw" fields.
    """
    client = ollama.AsyncClient(host=OLLAMA_HOST)

    user_message = (
        f"Mother's full operating instructions:\n---\n{mother_system_prompt}\n---\n\n"
        f"Session transcript to evaluate:\n---\n{transcript_text}\n---"
    )

    response = await client.chat(
        model=DISTILL_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        format="json",   # Ollama enforces JSON output
        options={
            "temperature": DISTILL_JUDGE_TEMPERATURE,
            "num_ctx": DISTILL_JUDGE_NUM_CTX,
            "num_predict": DISTILL_JUDGE_NUM_PREDICT,
        },
    )

    raw = response["message"]["content"]

    try:
        findings = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "what_worked": [],
            "what_to_avoid": [],
            "_parse_error": str(e),
            "_raw": raw,
        }

    findings.setdefault("what_worked", [])
    findings.setdefault("what_to_avoid", [])
    # Enforce hard cap at judge level in case LLM returns more than allowed
    findings["what_worked"] = findings["what_worked"][:MAX_FINDINGS_PER_LIST]
    findings["what_to_avoid"] = findings["what_to_avoid"][:MAX_FINDINGS_PER_LIST]
    return findings


# ── Merger ────────────────────────────────────────────────────────────────

def _enforce_caps(items):
    """Enforces per-list caps (max 4 critical + max 4 notable). Preserves
    the LLM's ordering within severity groups — the merger should place
    the most important entries first itself."""
    crit = [e for e in items if e.get("severity") == "critical"][:MAX_CRITICAL_PER_LIST]
    notable = [e for e in items if e.get("severity") == "notable"][:MAX_NOTABLE_PER_LIST]
    return crit + notable


def _clean_entry(item):
    """Clean one LLM output entry: trim, validate severity, drop empty
    quotes. Returns None if the entry is unusable."""
    if not isinstance(item, dict):
        return None
    quote = (item.get("quote") or "").strip()
    if not quote:
        return None
    reason = (item.get("reason") or "").strip()
    severity = item.get("severity", "notable")
    if severity not in ("critical", "notable"):
        severity = "notable"   # fall back to the milder label
    return {"quote": quote, "reason": reason, "severity": severity}


def _validate_merger_output(state, fallback):
    """Validate merger output + enforce caps. On broken schema:
    return fallback (= previous state), lessons.md stays untouched."""
    if not isinstance(state, dict):
        return fallback

    result = {}
    for key in ("what_worked", "what_to_avoid"):
        items = state.get(key, [])
        if not isinstance(items, list):
            return fallback
        cleaned = [c for c in (_clean_entry(it) for it in items) if c]
        result[key] = _enforce_caps(cleaned)
    return result


async def merge_findings(current_lessons, new_findings):
    """Call the merger LLM. Returns updated lessons state.

    Shortcuts:
    - Both new_findings lists empty → no LLM call, return current as-is.
    - LLM error or parse error → return current as-is (safe fallback,
      lessons.md stays unchanged).
    """
    n_worked = len(new_findings.get("what_worked") or [])
    n_avoid = len(new_findings.get("what_to_avoid") or [])
    if n_worked == 0 and n_avoid == 0:
        print("Merger: NEW_FINDINGS empty — lessons.md unchanged.")
        return current_lessons

    client = ollama.AsyncClient(host=OLLAMA_HOST)

    user_message = (
        "CURRENT_LESSONS:\n"
        f"{json.dumps(current_lessons, indent=2, ensure_ascii=False)}\n\n"
        "NEW_FINDINGS:\n"
        f"{json.dumps(new_findings, indent=2, ensure_ascii=False)}"
    )

    response = await client.chat(
        model=DISTILL_MODEL,
        messages=[
            {"role": "system", "content": MERGER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        format="json",
        options={
            "temperature": DISTILL_MERGER_TEMPERATURE,
            "num_ctx": DISTILL_MERGER_NUM_CTX,
            "num_predict": DISTILL_MERGER_NUM_PREDICT,
        },
    )

    raw = response["message"]["content"]
    try:
        updated = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Merger parse error: {e} — lessons.md unchanged.")
        return current_lessons

    return _validate_merger_output(updated, current_lessons)


async def update_lessons_file(new_findings, lessons_path=None):
    """Orchestrator: read lessons.md → merger → write lessons.md back.

    Logs before/after counts per list. Never raises outward — a distillation
    failure must not affect the main pipeline.
    """
    if lessons_path is None:
        lessons_path = LESSONS_PATH

    current = parse_lessons_md(lessons_path)
    b_worked = len(current["what_worked"])
    b_avoid = len(current["what_to_avoid"])

    try:
        updated = await merge_findings(current, new_findings)
    except Exception as e:
        print(f"Merger call failed: {e} — lessons.md unchanged.")
        return current

    a_worked = len(updated["what_worked"])
    a_avoid = len(updated["what_to_avoid"])

    lessons_path.parent.mkdir(parents=True, exist_ok=True)
    lessons_path.write_text(render_lessons_md(updated), encoding="utf-8")
    print(f"lessons.md updated: worked {b_worked}→{a_worked}, avoid {b_avoid}→{a_avoid}")
    return updated


# ── File output ───────────────────────────────────────────────────────────

def _format_finding_block(items):
    """A list of findings → markdown bullet points."""
    if not items:
        return "_(none flagged this session)_"
    lines = []
    for item in items:
        quote = (item.get("quote") or "").strip()
        reason = (item.get("reason") or "").strip()
        severity = item.get("severity", "notable")
        lines.append(f'- **"{quote}"** *({severity})*')
        lines.append(f"  {reason}")
        lines.append("")
    return "\n".join(lines).strip()


def format_findings(findings):
    """findings dict → markdown block for the per-session file."""
    if "_parse_error" in findings:
        return (
            f"_Parse error: {findings['_parse_error']}_\n\n"
            f"Raw judge output:\n```\n{findings.get('_raw', '')}\n```"
        )
    if "_error" in findings:
        return f"_Judge call failed: {findings['_error']}_"

    return (
        "### What worked\n\n"
        f"{_format_finding_block(findings['what_worked'])}\n\n"
        "### What to avoid\n\n"
        f"{_format_finding_block(findings['what_to_avoid'])}"
    )


def write_session_file(session_id, transcript_text, findings, n_turns):
    """Appends judge findings to the bottom of the existing transcript file
    distillates/sessions/session_{session_id}.txt — one file per session:
    transcript at the top (written by server.py), findings at the bottom.
    If the file does not exist (edge case, e.g. CLI run), creates it with
    transcript + findings."""
    DISTILLATES_DIR.mkdir(parents=True, exist_ok=True)
    path = DISTILLATES_DIR / f"session_{session_id}.txt"

    findings_block = (
        f"\n{'=' * 60}\n"
        f"JUDGE FINDINGS  ·  {n_turns} turns  ·  model: {DISTILL_MODEL}\n"
        f"{'=' * 60}\n\n"
        f"{format_findings(findings)}\n"
    )

    if path.exists():
        with open(path, "a", encoding="utf-8") as f:
            f.write(findings_block)
    else:
        path.write_text(f"{transcript_text}\n{findings_block}", encoding="utf-8")
    return path


def _append_distill_marker(session_id, text):
    """Writes a completion marker to the end of the session file.
    Signal for the client (run_session.sh): distillation for this session
    is done (including lessons.md). Called in EVERY code path — even when
    the floor is not reached — so the client never hits its timeout."""
    path = DISTILLATES_DIR / f"session_{session_id}.txt"
    try:
        DISTILLATES_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n--- distillation: {text} ---\n")
    except Exception as e:
        print(f"Could not write distillation marker: {e}")


# ── Orchestrator (entry point for server.py) ──────────────────────────────

async def run_distillation(messages, mother_system_prompt, session_id=None):
    """Main flow: check floor → judge → write per-session file → merger.

    Args:
        messages: list as in server.py — [{role, content}, ...]
        mother_system_prompt: Mother's full system prompt as string
        session_id: optional, defaults to current timestamp

    Returns:
        findings dict, or None if floor not reached.

    Never raises outward — everything in try/except so a distillation
    failure can never affect the main pipeline (server.py).
    """
    if session_id is None:
        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    n_turns = count_turn_pairs(messages)
    if n_turns < MIN_TURNS_FOR_DISTILLATION:
        print(f"Distillation skipped: {n_turns} turns < {MIN_TURNS_FOR_DISTILLATION} required.")
        _append_distill_marker(session_id, f"skipped ({n_turns} turns < {MIN_TURNS_FOR_DISTILLATION})")
        return None

    transcript_text = format_transcript(messages)

    print(f"Distillation start: {n_turns} turns · model={DISTILL_MODEL} · host={OLLAMA_HOST}")

    # Stage 1 — Judge
    try:
        findings = await judge_session(mother_system_prompt, transcript_text)
    except Exception as e:
        print(f"Judge call failed: {e}")
        findings = {"what_worked": [], "what_to_avoid": [], "_error": str(e)}

    path = write_session_file(session_id, transcript_text, findings, n_turns)
    print(f"Wrote: {path}")

    # Stage 2 — Merger (skip if judge failed; merge_findings() also skips
    # if both lists are empty).
    if "_error" in findings or "_parse_error" in findings:
        print("Merger skipped — judge had an error, lessons.md unchanged.")
        _append_distill_marker(session_id, "complete (judge error, lessons.md unchanged)")
        return findings

    try:
        await update_lessons_file(findings)
    except Exception as e:
        print(f"Merger failed: {e} — lessons.md unchanged.")

    # Marker written LAST — once this line appears in the session file,
    # findings + lessons.md are fully written. run_session.sh polls for it.
    _append_distill_marker(session_id, "complete")
    return findings


# ── CLI ───────────────────────────────────────────────────────────────────

def _cli():
    if len(sys.argv) < 2:
        print("Usage: python3 distillation.py <transcript-file>")
        print("Example: python3 distillation.py sessions/session_2026-05-21_18-30-00.txt")
        sys.exit(1)

    transcript_path = sys.argv[1]
    if not Path(transcript_path).exists():
        print(f"File not found: {transcript_path}")
        sys.exit(1)

    messages = parse_transcript_file(transcript_path)
    n_turns = count_turn_pairs(messages)
    print(f"Parsed {n_turns} turn-pairs from {transcript_path}")

    if n_turns == 0:
        print("No turns found — is the format right? Expected [USER]/[ASSISTANT] blocks.")
        sys.exit(1)

    # Derive session ID from filename (e.g. "session_2026-05-21_18-30-00.txt"
    # → "2026-05-21_18-30-00"). Fallback: current timestamp.
    stem = Path(transcript_path).stem
    session_id = stem.replace("session_", "") if stem.startswith("session_") else stem

    findings = asyncio.run(
        run_distillation(messages, MOTHER_SYSTEM_PROMPT, session_id)
    )

    if findings is not None:
        print("\n--- Findings (JSON) ---")
        clean = {k: v for k, v in findings.items() if not k.startswith("_")}
        print(json.dumps(clean, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
