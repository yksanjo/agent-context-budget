import json
import os
import subprocess
import sys
import tempfile
import unittest

import context_budget as cb


def msg(role, content, pin=False):
    return {"role": role, "content": content, "pin": pin,
            "tokens": cb.estimate_tokens(content),
            "digest": __import__("hashlib").sha256(content.encode()).hexdigest()[:16]}


def write_tmp(lines):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")
    return path


class EstimateTests(unittest.TestCase):
    def test_empty_is_one(self):
        self.assertEqual(cb.estimate_tokens(""), 1)

    def test_scales_with_length(self):
        self.assertEqual(cb.estimate_tokens("a" * 400), 100)

    def test_unicode_counts_characters(self):
        self.assertEqual(cb.estimate_tokens("é" * 40), 10)


class LoadTests(unittest.TestCase):
    def test_roundtrip(self):
        path = write_tmp([{"role": "system", "content": "you are x"},
                          {"role": "user", "content": "hi", "pin": True}])
        msgs = cb.load_transcript(path)
        os.unlink(path)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertTrue(msgs[1]["pin"])
        self.assertFalse(msgs[0]["pin"])

    def test_blank_lines_skipped(self):
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "w") as fh:
            fh.write('{"role":"user","content":"a"}\n\n{"role":"user","content":"b"}\n')
        self.assertEqual(len(cb.load_transcript(path)), 2)
        os.unlink(path)

    def test_invalid_json_rejected(self):
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "w") as fh:
            fh.write('{"role":"user"}\n')
        with self.assertRaises(cb.TranscriptError):
            cb.load_transcript(path)
        os.unlink(path)

    def test_unknown_role_rejected(self):
        path = write_tmp([{"role": "human", "content": "x"}])
        with self.assertRaises(cb.TranscriptError):
            cb.load_transcript(path)
        os.unlink(path)

    def test_non_string_content_rejected(self):
        path = write_tmp([{"role": "user", "content": ["x"]}])
        with self.assertRaises(cb.TranscriptError):
            cb.load_transcript(path)
        os.unlink(path)

    def test_non_object_rejected(self):
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "w") as fh:
            fh.write('["user","hi"]\n')
        with self.assertRaises(cb.TranscriptError):
            cb.load_transcript(path)
        os.unlink(path)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.sys = msg("system", "s" * 40)            # 10 tokens
        self.old = msg("user", "o" * 4000)            # 1000 tokens
        self.mid = msg("assistant", "m" * 400)        # 100 tokens
        self.new_u = msg("user", "n" * 40)            # 10 tokens
        self.new_a = msg("assistant", "n" * 40)       # 10 tokens

    def test_everything_fits(self):
        msgs = [self.sys, self.mid, self.new_u]
        res = cb.plan(msgs, cb.Budget(2000, keep_last=1))
        self.assertTrue(all(res["keep"]))
        self.assertTrue(res["fits"])
        self.assertEqual(res["dropped_messages"], 0)

    def test_old_dropped_when_tight(self):
        msgs = [self.sys, self.old, self.new_u, self.new_a]
        res = cb.plan(msgs, cb.Budget(500, keep_last=2))
        self.assertFalse(res["keep"][1])
        self.assertEqual(res["reasons"][1], "dropped")
        self.assertTrue(res["fits"])

    def test_keep_last_counts_non_system_only(self):
        msgs = [self.sys, self.old, self.new_u, self.new_a]
        res = cb.plan(msgs, cb.Budget(500, keep_last=2))
        # system + last two non-system are mandatory
        self.assertTrue(res["keep"][0])
        self.assertTrue(res["keep"][2])
        self.assertTrue(res["keep"][3])

    def test_pinned_survives(self):
        pinned = msg("user", "p" * 400, pin=True)
        msgs = [self.sys, pinned, self.old, self.new_u, self.new_a]
        res = cb.plan(msgs, cb.Budget(500, keep_last=2))
        self.assertTrue(res["keep"][1])
        self.assertEqual(res["reasons"][1], "pinned")

    def test_mandatory_overrun_is_error(self):
        big_pin = msg("user", "p" * 4000, pin=True)
        msgs = [big_pin, self.new_u]
        res = cb.plan(msgs, cb.Budget(500, keep_last=1))
        self.assertFalse(res["fits"])
        rules = [f["rule"] for f in res["findings"] if f["severity"] == "error"]
        self.assertIn("mandatory-over-budget", rules)

    def test_oversized_message_warns(self):
        msgs = [self.sys, self.old, self.new_u]
        res = cb.plan(msgs, cb.Budget(2000, keep_last=1))
        self.assertIn("oversized-message",
                      [f["rule"] for f in res["findings"]])

    def test_duplicate_content_warns(self):
        dup = msg("tool", "d" * 400)
        msgs = [self.sys, msg("tool", "d" * 400), dup, self.new_u]
        res = cb.plan(msgs, cb.Budget(4000, keep_last=1))
        self.assertIn("duplicate-content",
                      [f["rule"] for f in res["findings"]])

    def test_newest_first_fill_order(self):
        m1 = msg("user", "a" * 400)      # 100
        m2 = msg("assistant", "b" * 400) # 100
        m3 = msg("user", "c" * 400)      # 100
        msgs = [m1, m2, m3, self.new_u]
        res = cb.plan(msgs, cb.Budget(150, keep_last=1))
        # mandatory: new_u (10). Room 140: m3 (100) fits, m2 and m1 dropped
        self.assertEqual(res["keep"], [False, False, True, True])
        # with more room, m2 also survives - newest fills first
        res2 = cb.plan(msgs, cb.Budget(260, keep_last=1))
        self.assertEqual(res2["keep"], [False, True, True, True])

    def test_duplicate_kept_only_once(self):
        dump = "d" * 400
        msgs = [self.sys, msg("tool", dump), msg("tool", dump), self.new_u]
        res = cb.plan(msgs, cb.Budget(500, keep_last=1))
        kept = [i for i, k in enumerate(res["keep"]) if k]
        self.assertEqual(kept, [0, 2, 3])  # newer copy wins
        self.assertEqual(res["reasons"][1], "duplicate")

    def test_no_keep_system(self):
        big_sys = msg("system", "s" * 4000)  # 1000 tokens
        msgs = [big_sys, self.new_u]
        kept_when_forced = cb.plan(msgs, cb.Budget(30, keep_last=1))
        self.assertTrue(kept_when_forced["keep"][0])  # mandatory, triggers overrun
        dropped_when_not = cb.plan(msgs, cb.Budget(30, keep_last=1,
                                                   keep_system=False))
        self.assertFalse(dropped_when_not["keep"][0])
        self.assertTrue(dropped_when_not["fits"])

    def test_empty_transcript(self):
        res = cb.plan([], cb.Budget(1000))
        self.assertTrue(res["fits"])
        self.assertEqual(res["total_tokens"], 0)

    def test_budget_validation(self):
        with self.assertRaises(ValueError):
            cb.Budget(0)
        with self.assertRaises(ValueError):
            cb.Budget(100, keep_last=-1)

    def test_trimmed_preserves_order(self):
        msgs = [self.sys, self.old, self.mid, self.new_u, self.new_a]
        res = cb.plan(msgs, cb.Budget(120, keep_last=2))
        kept = cb.trimmed(msgs, res)
        self.assertEqual([m["role"] for m in kept],
                         ["system", "user", "assistant"])
        # order preserved even when a middle message survives
        res2 = cb.plan(msgs, cb.Budget(200, keep_last=2))
        self.assertEqual([m["role"] for m in cb.trimmed(msgs, res2)],
                         ["system", "assistant", "user", "assistant"])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.path = write_tmp([
            {"role": "system", "content": "you are a test agent"},
            {"role": "user", "content": "x" * 4000},
            {"role": "assistant", "content": "working on it"},
            {"role": "user", "content": "latest question"},
        ])

    def tearDown(self):
        os.unlink(self.path)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "context_budget.py", *args],
                              capture_output=True, text=True)

    def test_analyze_fits_exit_0(self):
        r = self.run_cli("analyze", self.path, "--max-tokens", "8000")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("FITS", r.stdout)

    def test_analyze_tight_still_fits_when_mandatory_ok(self):
        r = self.run_cli("analyze", self.path, "--max-tokens", "200")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("drop 1", r.stdout)

    def test_analyze_json(self):
        r = self.run_cli("analyze", self.path, "--max-tokens", "8000", "--json")
        data = json.loads(r.stdout)
        self.assertEqual(data["total_messages"], 4)
        self.assertTrue(data["fits"])

    def test_mandatory_overrun_exit_1(self):
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"role": "user", "content": "x" * 8000,
                                 "pin": True}) + "\n")
        r = self.run_cli("analyze", path, "--max-tokens", "100")
        os.unlink(path)
        self.assertEqual(r.returncode, 1)
        self.assertIn("mandatory-over-budget", r.stdout)

    def test_trim_outputs_kept_jsonl(self):
        r = self.run_cli("trim", self.path, "--max-tokens", "200")
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        roles = [json.loads(l)["role"] for l in lines]
        self.assertEqual(roles, ["system", "assistant", "user"])
        self.assertIn("kept 3 of 4", r.stderr)

    def test_missing_file_exit_2(self):
        r = self.run_cli("analyze", "/nonexistent.jsonl", "--max-tokens", "100")
        self.assertEqual(r.returncode, 2)

    def test_bad_transcript_exit_2(self):
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "w") as fh:
            fh.write("not json\n")
        r = self.run_cli("analyze", path, "--max-tokens", "100")
        os.unlink(path)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
