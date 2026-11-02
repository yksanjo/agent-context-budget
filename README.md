# agent-context-budget

Audit and trim an agent's conversation transcript against a context-window
budget, in dependency-free Python.

Agents that run for more than a few turns silently approach their context
limit: tool results pile up, the same content gets pasted twice, and the
oldest messages fall out of the window whether or not they mattered.
`agent-context-budget` reads a session transcript, estimates where the
window is going, plans which messages survive a fixed budget, and flags the
usual waste. No model calls and no network, so it is safe to run in CI or
inside another agent's loop.

## How it works

Input is a JSONL transcript, one message per line:

```json
{"role": "system", "content": "You are ..."}
{"role": "user", "content": "Summarize this filing", "pin": true}
{"role": "tool", "content": "...tool output..."}
```

Roles: `system`, `user`, `assistant`, `tool`. `pin` is optional.

The planner keeps, in priority order:

1. **Pinned messages** and **system messages** (disable with
   `--no-keep-system`), plus the **last `--keep-last` messages**. These are
   mandatory; if they alone exceed the budget, that is an error, not a trim.
2. Everything else, newest first, while room remains. Identical content is
   never kept twice; the newer copy wins.

Lint findings over the whole transcript: `oversized-message` (a single
message over 25% of the budget), `duplicate-content` (the same block pasted
more than once), and `mandatory-over-budget` (the mandatory set cannot
fit).

## Usage

```bash
# Audit a session against an 8k budget
python3 context_budget.py analyze session.jsonl --max-tokens 8000

# Machine-readable report
python3 context_budget.py analyze session.jsonl --max-tokens 8000 --json

# Emit the trimmed transcript (what you would actually send to the model)
python3 context_budget.py trim session.jsonl --max-tokens 8000 --keep-last 4 > slim.jsonl
```

Exit codes: `0` fits, `1` does not fit (mandatory set over budget), `2`
usage or input error.

Library:

```python
import context_budget as cb
msgs = cb.load_transcript("session.jsonl")
result = cb.plan(msgs, cb.Budget(max_tokens=8000, keep_last=4))
print(cb.report(msgs, result))
slim = cb.trimmed(msgs, result)
```

`python3 demo.py` runs a synthetic session: a big tool dump pasted twice, a
pinned style instruction, and a budget that forces a trim.

## Limits

- **Token counts are estimates.** The counter is `ceil(chars / 4)`, a
  planning heuristic. Real tokenizer counts vary by model and can differ by
  20% or more, especially for code and non-English text. Leave headroom in
  `--max-tokens`; do not set it to the model's exact window size.
- **No semantic judgment.** The planner knows recency, pins, roles, size,
  and duplicates. It does not know that message 12 held the key fact. Pin
  what matters; treat the trim as a mechanical default, not a summarizer.
- **Duplicate detection is exact-match only.** Near-duplicates (same dump,
  one line changed) are not caught.
- **Estimates are per message.** Overhead for roles, wrappers, and chat
  template tokens is not counted.
- The JSONL schema is the one documented above. Adapters for real agent
  frameworks belong in your own code.

## Tests

```bash
python3 -m unittest test_context_budget -v
```

29 tests cover token estimation, transcript validation, mandatory-set
rules, newest-first filling, pin and system handling, duplicate detection,
trim ordering, and the CLI exit codes.

## License

MIT
