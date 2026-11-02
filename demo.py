#!/usr/bin/env python3
"""Synthetic demo: a research agent's session transcript is about to overflow
its context window. Audit it against a budget, then trim. All content is
fictional."""

import json
import os
import tempfile

import context_budget as cb

SYSTEM = ("You are ResearchBot, a synthetic demo agent. Answer questions "
          "about fictional companies only. Never invent financial figures.")
TOOL_DUMP = ("ACME-DEMO-CORP quarterly report extract. " * 120)  # big paste
TRANSCRIPT = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Summarize what we know about Acme Demo Corp."},
    {"role": "assistant", "content": "I'll pull the two demo filings."},
    {"role": "tool", "content": TOOL_DUMP},
    {"role": "assistant", "content": "Got the filing. Key figures next."},
    {"role": "tool", "content": TOOL_DUMP},  # same dump pasted twice
    {"role": "assistant", "content": "Q1 demo revenue was 12 units, Q2 was 15."},
    {"role": "user", "content": "What changed between the quarters?"},
    {"role": "user", "content": "Also keep the summary under five sentences.",
     "pin": True},
    {"role": "user", "content": "And what is the outlook for Q3?"},
]

fd, path = tempfile.mkstemp(suffix=".jsonl")
with os.fdopen(fd, "w") as fh:
    for m in TRANSCRIPT:
        fh.write(json.dumps(m) + "\n")

messages = cb.load_transcript(path)
budget = cb.Budget(max_tokens=1500, keep_last=2)
result = cb.plan(messages, budget)

print("== analyze ==")
print(cb.report(messages, result))

print("\n== trim: messages that survive ==")
for i, m in enumerate(messages):
    mark = "KEEP " if result["keep"][i] else "DROP "
    print("%s #%d %-9s ~%-5d tokens (%s)"
          % (mark, i, m["role"], m["tokens"], result["reasons"][i]))

print("\nThe plan keeps the system prompt, the pinned style instruction,")
print("the latest questions, and one copy of the big tool dump. The older")
print("duplicate is dropped twice over: it no longer fits and its content")
print("is already kept once.")
os.unlink(path)
