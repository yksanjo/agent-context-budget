#!/usr/bin/env python3
"""context_budget - audit and trim an agent's conversation transcript against
a context-window budget.

Reads a JSONL transcript (one message per line), estimates token usage with a
documented heuristic, plans which messages survive a fixed budget, and reports
where the window is going. No model calls, no network, stdlib only.

Transcript line format:
    {"role": "system"|"user"|"assistant"|"tool", "content": "...", "pin": false}

Library usage:
    import context_budget as cb
    msgs = cb.load_transcript("session.jsonl")
    plan = cb.plan(msgs, cb.Budget(max_tokens=8000, keep_last=2))
    print(cb.report(msgs, plan))
"""

import argparse
import hashlib
import json
import math
import sys

CHARS_PER_TOKEN = 4  # documented heuristic; see README limits

ROLES = ("system", "user", "assistant", "tool")


class TranscriptError(Exception):
    """Bad transcript input."""


class Budget:
    def __init__(self, max_tokens, keep_last=2, keep_system=True):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if keep_last < 0:
            raise ValueError("keep_last must be >= 0")
        self.max_tokens = int(max_tokens)
        self.keep_last = int(keep_last)
        self.keep_system = bool(keep_system)


def estimate_tokens(text):
    """Rough token estimate: ceil(len/4) with a floor of 1 per message part.

    A heuristic for planning, never an exact model count. See README limits.
    """
    if not text:
        return 1
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def load_transcript(path):
    """Load and validate a JSONL transcript. Raises TranscriptError."""
    messages = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TranscriptError("line %d: invalid JSON: %s" % (lineno, exc))
            if not isinstance(obj, dict):
                raise TranscriptError("line %d: message must be a JSON object" % lineno)
            role = obj.get("role")
            if role not in ROLES:
                raise TranscriptError("line %d: role must be one of %s, got %r"
                                      % (lineno, ", ".join(ROLES), role))
            content = obj.get("content")
            if not isinstance(content, str):
                raise TranscriptError("line %d: content must be a string" % lineno)
            messages.append({
                "role": role,
                "content": content,
                "pin": bool(obj.get("pin", False)),
                "tokens": estimate_tokens(content),
                "digest": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
            })
    return messages


def plan(messages, budget):
    """Decide which messages survive the budget.

    Mandatory set: pinned messages, system messages (when keep_system), and
    the last keep_last conversation turns (a turn = one non-system message).
    Remaining space is filled with the newest unpinned, non-mandatory
    messages first; identical content is never kept twice (the newer copy
    wins). Returns a dict with keep flags, findings and totals.
    """
    n = len(messages)
    keep = [False] * n
    reasons = [None] * n

    # mandatory: pinned and system
    for i, m in enumerate(messages):
        if m["pin"]:
            keep[i], reasons[i] = True, "pinned"
        elif m["role"] == "system" and budget.keep_system:
            keep[i], reasons[i] = True, "system"

    # mandatory: last keep_last non-system messages
    seen = 0
    for i in range(n - 1, -1, -1):
        if messages[i]["role"] == "system":
            continue
        if seen < budget.keep_last:
            if not keep[i]:
                keep[i], reasons[i] = True, "recent"
            seen += 1
        else:
            break

    findings = []
    mandatory_tokens = sum(messages[i]["tokens"] for i in range(n) if keep[i])
    if mandatory_tokens > budget.max_tokens:
        findings.append({
            "severity": "error",
            "rule": "mandatory-over-budget",
            "detail": "pinned/system/recent messages need %d tokens but the budget is %d"
                      % (mandatory_tokens, budget.max_tokens),
        })

    # fill remaining space, newest first, never keeping the same content twice
    kept_digests = set(messages[i]["digest"] for i in range(n) if keep[i])
    room = budget.max_tokens - mandatory_tokens
    for i in range(n - 1, -1, -1):
        if keep[i]:
            continue
        t = messages[i]["tokens"]
        if messages[i]["digest"] in kept_digests:
            reasons[i] = "duplicate"
        elif t <= room:
            keep[i], reasons[i] = True, "fits"
            kept_digests.add(messages[i]["digest"])
            room -= t
        else:
            reasons[i] = "dropped"

    # lint findings over the whole transcript
    for i, m in enumerate(messages):
        if m["tokens"] > 0.25 * budget.max_tokens:
            findings.append({
                "severity": "warning",
                "rule": "oversized-message",
                "detail": "message %d (%s) uses ~%d tokens, over 25%% of the budget"
                          % (i, m["role"], m["tokens"]),
            })
    seen_digests = {}
    for i, m in enumerate(messages):
        if m["digest"] in seen_digests and m["tokens"] > 32:
            findings.append({
                "severity": "warning",
                "rule": "duplicate-content",
                "detail": "message %d repeats the content of message %d (~%d tokens each)"
                          % (i, seen_digests[m["digest"]], m["tokens"]),
            })
        else:
            seen_digests.setdefault(m["digest"], i)

    kept_tokens = sum(messages[i]["tokens"] for i in range(n) if keep[i])
    total_tokens = sum(m["tokens"] for m in messages)
    return {
        "keep": keep,
        "reasons": reasons,
        "findings": findings,
        "total_messages": n,
        "total_tokens": total_tokens,
        "kept_tokens": kept_tokens,
        "dropped_tokens": total_tokens - kept_tokens,
        "dropped_messages": sum(1 for k in keep if not k),
        "fits": mandatory_tokens <= budget.max_tokens,
        "budget": budget.max_tokens,
    }


def report(messages, result):
    """Human-readable analysis."""
    lines = []
    lines.append("Budget: %d tokens (est.) | transcript: %d messages, ~%d tokens"
                 % (result["budget"], result["total_messages"], result["total_tokens"]))
    lines.append("Plan: keep %d (~%d tokens), drop %d (~%d tokens) -> %s"
                 % (result["total_messages"] - result["dropped_messages"],
                    result["kept_tokens"], result["dropped_messages"],
                    result["dropped_tokens"],
                    "FITS" if result["fits"] else "DOES NOT FIT"))
    by_role = {}
    for m in messages:
        by_role[m["role"]] = by_role.get(m["role"], 0) + m["tokens"]
    lines.append("By role: " + ", ".join("%s ~%d" % (r, by_role[r])
                                         for r in ROLES if r in by_role))
    if messages:
        biggest = max(range(len(messages)), key=lambda i: messages[i]["tokens"])
        lines.append("Largest message: #%d (%s, ~%d tokens, %s)"
                     % (biggest, messages[biggest]["role"],
                        messages[biggest]["tokens"], result["reasons"][biggest]))
    if result["fits"]:
        util = 100.0 * result["kept_tokens"] / result["budget"]
        lines.append("Utilization after trim: %.0f%%" % util)
    for f in result["findings"]:
        lines.append("%s [%s] %s" % (f["severity"].upper(), f["rule"], f["detail"]))
    return "\n".join(lines)


def trimmed(messages, result):
    """The messages that survive the plan, in original order."""
    return [m for i, m in enumerate(messages) if result["keep"][i]]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="context_budget",
        description="Audit and trim an agent transcript against a context budget.")
    parser.add_argument("command", choices=["analyze", "trim"])
    parser.add_argument("transcript", help="JSONL transcript file")
    parser.add_argument("--max-tokens", type=int, required=True,
                        help="context window budget in estimated tokens")
    parser.add_argument("--keep-last", type=int, default=2,
                        help="always keep this many most recent messages (default 2)")
    parser.add_argument("--no-keep-system", action="store_true",
                        help="do not force-keep system messages")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    try:
        messages = load_transcript(args.transcript)
        budget = Budget(args.max_tokens, keep_last=args.keep_last,
                        keep_system=not args.no_keep_system)
    except (TranscriptError, ValueError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    result = plan(messages, budget)
    if args.command == "analyze":
        if args.json:
            out = dict(result)
            del out["keep"], out["reasons"]
            print(json.dumps(out, indent=2, sort_keys=True))
        else:
            print(report(messages, result))
        return 0 if result["fits"] and not any(
            f["severity"] == "error" for f in result["findings"]) else 1

    # trim
    kept = trimmed(messages, result)
    for m in kept:
        print(json.dumps({"role": m["role"], "content": m["content"],
                          **({"pin": True} if m["pin"] else {})},
                         ensure_ascii=False))
    print("# kept %d of %d messages (~%d of ~%d est. tokens)"
          % (len(kept), result["total_messages"],
             result["kept_tokens"], result["total_tokens"]), file=sys.stderr)
    return 0 if result["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())
