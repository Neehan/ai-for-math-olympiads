"""Alternate-sketch routing, reference isolation, and pre-firewall loading."""

import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.checkpoint_namespace import namespace
from src import storage
from src.config import load_config
from src.constants import CONFIG_PATH, data_source_urls
from src.prefetch import prefetch
from src.prompts import audit_prompt, state_audit_prompt, task_prompt
from src.run import hint_for


class AltHintTests(unittest.TestCase):
    def setUp(self):
        self.urls = data_source_urls("alt-hint")
        self.solutions = [
            {"problem_id": pid, "statement": f"Prove {pid}.", "reference_solutions": [
                {"route_id": "hard_hint", "solution": f"PRIVATE alternate proof {pid}."}
            ]}
            for pid in ("aobench-example", "PB-Advanced-example")
        ]
        self.hints = [
            {"problem_id": r["problem_id"], "domain": domain, "hint": "Use the alternate invariant."}
            for r, domain in zip(self.solutions, ("algebra", "combinatorics"))
        ]
        self.outlines = [
            {"problem_id": r["problem_id"], "steps": [{"step": s} for s in (
                "Construct the alternate invariant.", "Apply the alternate bound.", "Handle equality."
            )]}
            for r in self.solutions
        ]
        self.sources = {
            self.urls["solutions"]: self.solutions,
            self.urls["hints"]: self.hints,
            self.urls["outlines"]: self.outlines,
        }

    def fetch(self, env, url):
        return copy.deepcopy(self.sources[url])

    def download(self, url, **kwargs):
        return io.BytesIO("\n".join(json.dumps(r) for r in self.sources[url]).encode())

    def test_same_hint_protocol_and_no_proof_or_outline_in_generation_prompt(self):
        config = load_config(CONFIG_PATH)
        original, alternate = config.arms["hint"], config.arms["alt-hint"]
        self.assertEqual((original.hint, original.mode, original.budget_units, original.seeds),
                         (alternate.hint, alternate.mode, alternate.budget_units, alternate.seeds))
        with patch.object(storage, "_fetch_jsonl", side_effect=self.fetch):
            problems = storage.load_problems("alt-hint")
        self.assertEqual(len(problems), 2)
        for problem in problems:
            prompt = task_prompt(problem, hint_for(problem, alternate), "/tmp/scratch", 200_000)
            self.assertEqual(
                prompt,
                task_prompt(problem, hint_for(problem, original), "/tmp/scratch", 200_000),
            )
            self.assertIn("Use the alternate invariant.", prompt)
            self.assertNotIn("PRIVATE", prompt)
            self.assertNotIn("Apply the alternate bound.", prompt)
            self.assertIsNone(problem.hint_h1)

    def test_both_audits_use_alternate_proof_and_state_uses_alternate_outline(self):
        with patch.object(storage, "_fetch_jsonl", side_effect=self.fetch):
            problem = storage.load_problems("alt-hint")[0]
            correctness = storage.load_audit_references("alt-hint")
            states = storage.load_state_audit_references("alt-hint")
        proof = correctness[problem.problem_id][1]
        self.assertEqual(correctness, states)
        self.assertIn("PRIVATE alternate proof", proof)
        self.assertIn(proof, audit_prompt(problem, reference_solution=proof, solution_text="Candidate proof."))
        state = state_audit_prompt(problem, outline=problem.hint_h3, reference_solution=proof, solution_text="Candidate proof.")
        self.assertIn(proof, state)
        self.assertIn("Apply the alternate bound.", state)

    def test_missing_duplicate_and_mismatched_rows_fail_closed(self):
        for fault in ("missing", "duplicate", "extra"):
            with self.subTest(fault=fault):
                rows = copy.deepcopy(self.hints)
                if fault == "missing":
                    rows.pop()
                elif fault == "duplicate":
                    rows.append(rows[0])
                else:
                    rows.append(dict(rows[0], problem_id="unexpected"))
                with patch.object(storage, "_fetch_jsonl", side_effect=lambda env, url: rows if url == self.urls["hints"] else self.fetch(env, url)):
                    with self.assertRaises(ValueError):
                        storage.load_problems("alt-hint")

    def test_empty_or_oversize_hint_and_wrong_outline_rejected(self):
        for hint in ("", "word " * 26):
            with self.subTest(hint=hint):
                self.hints[0]["hint"] = hint
                with patch.object(storage, "_fetch_jsonl", side_effect=self.fetch):
                    with self.assertRaises(ValueError):
                        storage.load_problems("alt-hint")
        self.hints[0]["hint"] = "Use the alternate invariant."
        self.outlines[0]["steps"].pop()
        with patch.object(storage, "_fetch_jsonl", side_effect=self.fetch):
            with self.assertRaises(ValueError):
                storage.load_problems("alt-hint")

    def test_prefetch_and_consume_all_stages_without_generation_proof_leak(self):
        for stage in ("run", "audit", "state-audit"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                with patch("src.prefetch.urllib.request.urlopen", side_effect=self.download) as get:
                    prefetch(stage, "alt-hint", directory)
                self.assertEqual(get.call_count, 3)  # only the three alt files
                self.assertNotIn("PRIVATE", (directory / "problems.jsonl").read_text())
                self.assertEqual((directory / "solutions.jsonl").exists(), stage != "run")
                env = {f"{name.upper()}_FILE": str(directory / f"{name}.jsonl")
                       for name in ("problems", "hints", "outlines", "solutions")}
                with patch.dict(os.environ, env), patch("src.storage.urllib.request.urlopen", side_effect=AssertionError("Unexpected network call")):
                    self.assertEqual(len(storage.load_problems("alt-hint")), 2)
                    if stage == "audit":
                        self.assertEqual(len(storage.load_audit_references("alt-hint")), 2)
                    elif stage == "state-audit":
                        self.assertEqual(len(storage.load_state_audit_references("alt-hint")), 2)
                self.assertEqual(list(directory.iterdir()), [])

    def test_alt_urls_ignore_dataset_but_regular_urls_do_not(self):
        for dataset, regular in (("math-contests-2026", "hard_hints"), ("imobench", "imobench_hints")):
            env = dict(os.environ, HARNESS_DATASET=dataset)
            output = subprocess.check_output([sys.executable, "-c", "import json; from src.constants import data_source_urls; print(json.dumps([data_source_urls('alt-hint'),data_source_urls('hint')]))"], env=env, text=True)
            alternate, original = json.loads(output)
            self.assertEqual(alternate, self.urls)
            self.assertTrue(original["hints"].endswith(f"/{regular}.jsonl"))

    def test_regular_prefetch_keeps_original_sources(self):
        urls = data_source_urls("hint")
        problems = [dict(r, domain="algebra") for r in self.solutions]
        sources = {
            urls["problems"]: problems, urls["hints"]: self.hints,
            urls["outlines"]: self.outlines, urls["solutions"]: self.solutions,
        }
        def download(url, **kwargs):
            return io.BytesIO("\n".join(json.dumps(r) for r in sources[url]).encode())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with patch("src.prefetch.urllib.request.urlopen", side_effect=download) as get:
                prefetch("audit", "hint", path)
            self.assertEqual(get.call_count, 4)
            rows = [json.loads(x) for x in (path / "problems.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["domain"], "algebra")
            self.assertNotIn("reference_solutions", rows[0])
            self.assertIn("PRIVATE", (path / "solutions.jsonl").read_text())

    def test_host_results_routing_ignores_dataset_only_for_alt_hint(self):
        root = CONFIG_PATH.parent
        prefix = (root / "run.sh").read_text().split("IMAGE=olympiad-harness", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "config.json").write_text(CONFIG_PATH.read_text())
            script = directory / "routing.sh"
            script.write_text(prefix + '\nprintf "%s|%s\\n" "$DATASET_NAME" "$RESULTS_DIR_NAME"\n')
            for stage in ("run", "audit", "state-audit"):
                for flag in (["--dataset", "imobench"], ["--dataset=math-contests-2026"]):
                    output = subprocess.check_output(["bash", str(script), stage, "--arm=alt-hint", *flag], text=True)
                    self.assertEqual(output.strip(), "math-contests-2026|results")
            output = subprocess.check_output(["bash", str(script), "run", "--arm", "hint", "--dataset", "imobench"], text=True)
            self.assertEqual(output.strip(), "imobench|results-imobench")

    def test_adding_alt_arm_preserves_all_existing_checkpoint_namespaces(self):
        current = CONFIG_PATH.read_bytes()
        previous = b"".join(line for line in current.splitlines(keepends=True) if b'"alt-hint"' not in line)
        read_bytes = Path.read_bytes
        settings = CONFIG_PATH.parent / "agent_settings.json"
        for arm in load_config(CONFIG_PATH).arms:
            if arm == "alt-hint":
                continue
            with self.subTest(arm=arm):
                args = ["run", "test-solver", "test-judge", arm]
                now = namespace(args, settings)
                with patch.object(Path, "read_bytes", lambda p: previous if p == Path("config.json") else read_bytes(p)):
                    self.assertEqual(namespace(args, settings), now)

    def test_alt_audit_automatically_enables_state_annotation(self):
        script = (CONFIG_PATH.parent / "run.sh").read_text()
        dispatch = "RUN_STATE_AUDIT=0" + script.split("RUN_STATE_AUDIT=0", 1)[1].split("# State annotation runs", 1)[0]
        for arm, expected in (("alt-hint", "1"), ("hint", "0"), ("baseline-parallel", "1")):
            result = subprocess.check_output(
                ["bash", "-c", 'ARM_NAME="$1"\n' + dispatch + '\nprintf "%s" "$RUN_STATE_AUDIT"', "test", arm],
                text=True,
            )
            self.assertEqual(result, expected)

    def test_missing_hf_file_does_not_fall_back_to_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.prefetch.urllib.request.urlopen", side_effect=OSError("not uploaded")) as fetch:
                with self.assertRaisesRegex(OSError, "not uploaded"):
                    prefetch("run", "alt-hint", Path(tmp))
            self.assertEqual(fetch.call_count, 1)
            self.assertIn("hard_alt_solutions.jsonl", fetch.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
