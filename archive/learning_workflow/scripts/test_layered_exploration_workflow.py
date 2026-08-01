#!/usr/bin/env python3
"""Offline tests for the standalone layered exploration workflow."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layered_exploration_core as core
import layered_exploration_orchestrator as orchestrator


class ProviderParserTests(unittest.TestCase):
    def test_provider_schemas_require_every_declared_object_field(self) -> None:
        def inspect(schema):
            schema_types = schema.get("type")
            if schema_types == "object" or (
                isinstance(schema_types, list) and "object" in schema_types
            ):
                properties = schema.get("properties", {})
                self.assertFalse(schema.get("additionalProperties", True))
                self.assertEqual(set(properties), set(schema.get("required", [])))
                for child in properties.values():
                    inspect(child)
            if schema_types == "array" or (
                isinstance(schema_types, list) and "array" in schema_types
            ):
                inspect(schema.get("items", {}))

        for schema in core.SCHEMAS.values():
            inspect(schema)

    def test_parse_deepseek_structured_stream(self) -> None:
        expected = {
            "action": "complete",
            "candidates": [],
            "gaps": [],
            "next_queries": [],
        }
        stdout = "\n".join(
            [
                json.dumps({"type": "system", "session_id": "deepseek-session"}),
                json.dumps(
                    {
                        "type": "result",
                        "session_id": "deepseek-session",
                        "structured_output": expected,
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    }
                ),
            ]
        )
        data, session_id, telemetry, events = core.parse_deepseek_stream(stdout)
        self.assertEqual(expected, data)
        self.assertEqual("deepseek-session", session_id)
        self.assertEqual(2, len(events))
        self.assertIn("usage", telemetry)

    def test_parse_codex_stream_and_tool_rejection_signal(self) -> None:
        message = {
            "action": "complete",
            "candidates": [],
            "gaps": [],
            "next_queries": [],
        }
        clean_stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "codex-thread"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item-1",
                            "type": "agent_message",
                            "text": json.dumps(message),
                        },
                    }
                ),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4}}),
            ]
        )
        data, session_id, _telemetry, tool_events = core.parse_codex_stream(clean_stdout)
        self.assertEqual(message, data)
        self.assertEqual("codex-thread", session_id)
        self.assertEqual([], tool_events)

        tool_stdout = clean_stdout.replace(
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4}}),
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {"type": "command_execution", "command": "pwd"},
                        }
                    ),
                    json.dumps(
                        {"type": "turn.completed", "usage": {"input_tokens": 4}}
                    ),
                ]
            ),
        )
        _data, _session, _telemetry, tool_events = core.parse_codex_stream(tool_stdout)
        self.assertEqual(1, len(tool_events))

    def test_codex_command_has_root_approval_policy_and_read_only_sandbox(self) -> None:
        with tempfile.TemporaryDirectory(dir=core.VAULT_ROOT) as temporary:
            work_dir = Path(temporary)
            provider = core.CodexCLIProvider(work_dir)
            output = "\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {
                                "type": "agent_message",
                                "text": json.dumps(
                                    {
                                        "action": "complete",
                                        "candidates": [],
                                        "gaps": [],
                                        "next_queries": [],
                                    }
                                ),
                            },
                        }
                    ),
                    json.dumps({"type": "turn.completed", "usage": {}}),
                ]
            )
            captured: dict[str, object] = {}

            def fake_run_process(command, **kwargs):
                captured["command"] = list(command)
                captured["kwargs"] = kwargs
                return 0, output, "", 0.01

            with mock.patch.object(core, "run_process", fake_run_process):
                provider.call(
                    role="discovery",
                    prompt="synthetic",
                    schema_name="discovery",
                    conversation_key="test",
                )
            command = captured["command"]
            self.assertEqual("codex", command[0])
            self.assertEqual(
                ["--ask-for-approval", "never"],
                command[1:3],
            )
            self.assertLess(command.index("--sandbox"), command.index("exec"))
            self.assertIn("read-only", command)
            self.assertIn("--output-schema", command)

    def test_deepseek_command_uses_claude_cli_no_tools_and_schema(self) -> None:
        with tempfile.TemporaryDirectory(dir=core.VAULT_ROOT) as temporary:
            provider = core.DeepSeekCLIProvider(
                Path(temporary),
                model="deepseek-test-model",
            )
            payload = {
                "action": "complete",
                "candidates": [],
                "gaps": [],
                "next_queries": [],
            }
            output = json.dumps(
                {
                    "type": "result",
                    "session_id": "deepseek-thread",
                    "structured_output": payload,
                }
            )
            captured: dict[str, object] = {}

            def fake_run_process(command, **kwargs):
                captured["command"] = list(command)
                captured["kwargs"] = kwargs
                return 0, output, "", 0.01

            with mock.patch.object(core, "run_process", fake_run_process):
                response = provider.call(
                    role="discovery",
                    prompt="synthetic",
                    schema_name="discovery",
                    conversation_key="deepseek-command",
                )
            command = captured["command"]
            self.assertEqual("claude", command[0])
            self.assertIn("deepseek-test-model", command)
            self.assertIn("--json-schema", command)
            tools_index = command.index("--tools")
            self.assertEqual("", command[tools_index + 1])
            self.assertEqual("deepseek-thread", response.session_id)

    def test_codex_first_turn_can_resume_once_for_structure_repair(self) -> None:
        with tempfile.TemporaryDirectory(dir=core.VAULT_ROOT) as temporary:
            provider = core.CodexCLIProvider(Path(temporary))
            invalid = "\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "repair-thread"}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {
                                "type": "agent_message",
                                "text": json.dumps({"unexpected": True}),
                            },
                        }
                    ),
                    json.dumps({"type": "turn.completed", "usage": {}}),
                ]
            )
            valid = "\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "repair-thread"}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {
                                "type": "agent_message",
                                "text": json.dumps(
                                    {
                                        "action": "complete",
                                        "candidates": [],
                                        "gaps": [],
                                        "next_queries": [],
                                    }
                                ),
                            },
                        }
                    ),
                    json.dumps({"type": "turn.completed", "usage": {}}),
                ]
            )
            commands: list[list[str]] = []
            outputs = iter((invalid, valid))

            def fake_run_process(command, **_kwargs):
                commands.append(list(command))
                return 0, next(outputs), "", 0.01

            with mock.patch.object(core, "run_process", fake_run_process):
                response = provider.call(
                    role="discovery",
                    prompt="synthetic",
                    schema_name="discovery",
                    conversation_key="repair",
                )
            self.assertEqual("complete", response.data["action"])
            self.assertEqual(2, len(commands))
            self.assertIn("resume", commands[1])
            self.assertIn("repair-thread", commands[1])


class EvidenceTests(unittest.TestCase):
    def test_quote_and_line_coordinates_are_authoritative(self) -> None:
        with tempfile.TemporaryDirectory(dir=core.VAULT_ROOT) as temporary:
            path = Path(temporary) / "evidence.md"
            path.write_text("first\nexact supporting sentence\nthird\n", encoding="utf-8")
            candidate = {
                "statement": "The sentence is supported.",
                "claim_type": "method",
                "layer": "L1",
                "entity_name": "Example",
                "source_path": str(path),
                "line_start": 2,
                "line_end": 2,
                "quote": "exact supporting sentence",
                "evidence_mode": "direct",
                "scope": "test",
                "confidence": "high",
            }
            claim, errors = core.candidate_to_claim(candidate)
            self.assertFalse(errors)
            self.assertIsNotNone(claim)
            invalid = {**candidate, "quote": "invented quote"}
            rejected, errors = core.candidate_to_claim(invalid)
            self.assertIsNone(rejected)
            self.assertIn("quote does not match", errors[0])

    def test_direction_entries_require_an_explicit_connected_subgraph(self) -> None:
        edge = {
            "from_entry_id": "E-L2",
            "to_entry_id": "E-L4",
        }
        self.assertTrue(
            orchestrator._selected_entries_are_connected(
                {"E-L2", "E-L4"},
                [edge],
            )
        )
        self.assertFalse(
            orchestrator._selected_entries_are_connected(
                {"E-L2", "E-L3", "E-L4"},
                [edge],
            )
        )
        self.assertTrue(
            orchestrator._selected_entries_are_connected({"E-L2"}, [])
        )


class OfflineWorkflowTests(unittest.TestCase):
    def test_fixture_provider_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory(dir=core.VAULT_ROOT) as temporary:
            base = Path(temporary)
            knowledge_root = base / "kb"
            knowledge_root.mkdir()
            source = knowledge_root / "runtime.md"
            source.write_text(
                "# Runtime evidence\n"
                "The baseline runs requests sequentially on GPU and reports latency.\n"
                "The runtime scheduler can overlap independent preprocessing with GPU execution.\n"
                "The implementation is available as the RuntimeX module and profiler.\n",
                encoding="utf-8",
            )
            source_path = source.relative_to(core.VAULT_ROOT).as_posix()

            candidates = [
                {
                    "statement": "The current GPU runtime baseline executes requests sequentially.",
                    "claim_type": "baseline",
                    "layer": "L2",
                    "entity_name": "Sequential GPU runtime",
                    "source_path": source_path,
                    "line_start": 2,
                    "line_end": 2,
                    "quote": "The baseline runs requests sequentially on GPU and reports latency.",
                    "evidence_mode": "direct",
                    "scope": "GPU request execution",
                    "confidence": "high",
                },
                {
                    "statement": "Independent preprocessing can overlap GPU execution.",
                    "claim_type": "scenario",
                    "layer": "L2",
                    "entity_name": "Overlap scheduler",
                    "source_path": source_path,
                    "line_start": 3,
                    "line_end": 3,
                    "quote": "The runtime scheduler can overlap independent preprocessing with GPU execution.",
                    "evidence_mode": "direct",
                    "scope": "independent preprocessing and GPU execution",
                    "confidence": "high",
                },
                {
                    "statement": "RuntimeX provides a reusable module and profiler.",
                    "claim_type": "implementation",
                    "layer": "L2",
                    "entity_name": "RuntimeX",
                    "source_path": source_path,
                    "line_start": 4,
                    "line_end": 4,
                    "quote": "The implementation is available as the RuntimeX module and profiler.",
                    "evidence_mode": "direct",
                    "scope": "runtime implementation and measurement",
                    "confidence": "high",
                },
            ]
            claims = []
            for candidate in candidates:
                claim, errors = core.candidate_to_claim(candidate)
                self.assertFalse(errors)
                claims.append(claim)
            baseline_claim_id = claims[0]["claim_id"]
            opportunity_claim_id = claims[1]["claim_id"]
            implementation_claim_id = claims[2]["claim_id"]

            anchor_id = core.anchor_id_for(
                workload="GPU request serving",
                phase="request execution",
                regime="independent preprocessing available",
                backend="single GPU with CPU preprocessing",
                bottleneck="serialized preprocessing and GPU execution",
                target_metrics=["end-to-end latency", "GPU utilization"],
                primary_baseline_name="Sequential GPU runtime",
            )
            baseline_id = core.baseline_id_for(
                anchor_id=anchor_id,
                role="current_practice",
                name="Sequential GPU runtime",
            )
            baseline_entity_id = core.entity_id_for(
                name="Sequential GPU runtime",
                entity_type="system",
            )
            opportunity_entity_id = core.entity_id_for(
                name="Overlap scheduler",
                entity_type="method",
            )
            implementation_entity_id = core.entity_id_for(
                name="RuntimeX",
                entity_type="code",
            )
            baseline_entry_id = core.entry_id_for(
                anchor_id=anchor_id,
                layer="L2",
                role="baseline_behavior",
                entity_id=baseline_entity_id,
                claim="Requests are serialized in the current GPU runtime.",
            )
            opportunity_entry_id = core.entry_id_for(
                anchor_id=anchor_id,
                layer="L2",
                role="opportunity",
                entity_id=opportunity_entity_id,
                claim="Overlap independent preprocessing with GPU execution.",
            )
            implementation_entry_id = core.entry_id_for(
                anchor_id=anchor_id,
                layer="L2",
                role="implementation",
                entity_id=implementation_entity_id,
                claim="RuntimeX supplies a module and profiler.",
            )
            direction_id = core.direction_id_for(
                anchor_id=anchor_id,
                title="Overlap preprocessing with GPU execution",
                selected_entry_ids=[opportunity_entry_id],
                hypothesis=(
                    "Overlapping independent preprocessing with GPU execution reduces "
                    "end-to-end latency relative to sequential execution."
                ),
            )
            null_review = {
                key: None for key in core.REVIEW_SCHEMA["properties"]
            }

            fixture = [
                {
                    "schema_name": "discovery",
                    "data": {
                        "action": "complete",
                        "candidates": candidates,
                        "gaps": [],
                        "next_queries": [],
                    },
                },
                {
                    "schema_name": "curation",
                    "data": {
                        "action": "complete",
                        "anchors": [
                            {
                                "anchor_key": "runtime-overlap",
                                "workload": "GPU request serving",
                                "phase": "request execution",
                                "regime": "independent preprocessing available",
                                "backend": "single GPU with CPU preprocessing",
                                "bottleneck": "serialized preprocessing and GPU execution",
                                "primary_baseline_key": "sequential",
                                "target_metrics": [
                                    "end-to-end latency",
                                    "GPU utilization",
                                ],
                                "evidence_refs": [
                                    baseline_claim_id,
                                    opportunity_claim_id,
                                ],
                                "status": "active",
                                "baselines": [
                                    {
                                        "baseline_key": "sequential",
                                        "name": "Sequential GPU runtime",
                                        "role": "current_practice",
                                        "description": (
                                            "Requests and preprocessing execute sequentially."
                                        ),
                                        "evidence_refs": [baseline_claim_id],
                                    }
                                ],
                                "entries": [
                                    {
                                        "entry_key": "baseline-entry",
                                        "entity_name": "Sequential GPU runtime",
                                        "entity_type": "system",
                                        "layer": "L2",
                                        "role": "baseline_behavior",
                                        "claim": (
                                            "Requests are serialized in the current GPU runtime."
                                        ),
                                        "modifiable_object": "request execution order",
                                        "applicable_baseline_keys": ["sequential"],
                                        "preconditions": [],
                                        "expected_effect": "reference latency",
                                        "evidence_refs": [baseline_claim_id],
                                        "confidence": "high",
                                        "status": "accepted",
                                    },
                                    {
                                        "entry_key": "opportunity-entry",
                                        "entity_name": "Overlap scheduler",
                                        "entity_type": "method",
                                        "layer": "L2",
                                        "role": "opportunity",
                                        "claim": (
                                            "Overlap independent preprocessing with GPU execution."
                                        ),
                                        "modifiable_object": "runtime scheduling order",
                                        "applicable_baseline_keys": ["sequential"],
                                        "preconditions": [
                                            "preprocessing and GPU execution are independent"
                                        ],
                                        "expected_effect": "lower end-to-end latency",
                                        "evidence_refs": [opportunity_claim_id],
                                        "confidence": "high",
                                        "status": "accepted",
                                    },
                                    {
                                        "entry_key": "implementation-entry",
                                        "entity_name": "RuntimeX",
                                        "entity_type": "code",
                                        "layer": "L2",
                                        "role": "implementation",
                                        "claim": "RuntimeX supplies a module and profiler.",
                                        "modifiable_object": "RuntimeX scheduling module",
                                        "applicable_baseline_keys": ["sequential"],
                                        "preconditions": [],
                                        "expected_effect": "reduce implementation effort",
                                        "evidence_refs": [implementation_claim_id],
                                        "confidence": "high",
                                        "status": "accepted",
                                    },
                                    {
                                        "entry_key": "imprecise-entry",
                                        "entity_name": "Unverified optimizer",
                                        "entity_type": "method",
                                        "layer": "L2",
                                        "role": "method",
                                        "claim": "An unverified optimizer might help.",
                                        "modifiable_object": "unknown scheduler field",
                                        "applicable_baseline_keys": ["sequential"],
                                        "preconditions": [],
                                        "expected_effect": "unknown",
                                        "evidence_refs": ["C-NOT-A-REAL-CLAIM"],
                                        "confidence": "low",
                                        "status": "candidate",
                                    },
                                ],
                                "edges": [],
                                "gaps": [
                                    "strong and tool-evaluation baselines remain missing"
                                ],
                            }
                        ],
                        "evidence_requests": [],
                        "dispositions": [
                            {
                                "claim_id": baseline_claim_id,
                                "status": "integrated",
                                "reason": "defines current practice",
                            },
                            {
                                "claim_id": opportunity_claim_id,
                                "status": "integrated",
                                "reason": "defines an explorable overlap opportunity",
                            },
                            {
                                "claim_id": implementation_claim_id,
                                "status": "integrated",
                                "reason": "defines reusable implementation",
                            },
                        ],
                        "unresolved_gaps": [
                            "strong and tool-evaluation baselines remain missing"
                        ],
                    },
                },
                {
                    "schema_name": "direction",
                    "data": {
                        "action": "complete",
                        "directions": [
                            {
                                "title": "Overlap preprocessing with GPU execution",
                                "selected_entry_ids": [
                                    opportunity_entry_id,
                                ],
                                "selected_edge_ids": [],
                                "baseline_ids": [baseline_id],
                                "hypothesis": (
                                    "Overlapping independent preprocessing with GPU "
                                    "execution reduces end-to-end latency relative to "
                                    "sequential execution."
                                ),
                                "expected_effects": [
                                    "lower end-to-end latency",
                                    "higher GPU utilization",
                                ],
                                "preconditions": [
                                    "preprocessing and GPU execution are independent"
                                ],
                                "ablation_plan": [
                                    "sequential baseline",
                                    "overlap scheduler",
                                ],
                                "evidence_refs": [
                                    baseline_claim_id,
                                    opportunity_claim_id,
                                ],
                                "gaps": [],
                                "kind": "experiment",
                            }
                        ],
                        "evidence_requests": [],
                        "unresolved_gaps": [],
                    },
                },
                {
                    "schema_name": "judge",
                    "data": {
                        "action": "ask",
                        "dimension": "scenario_opportunity",
                        "question": (
                            "Does the evidence establish independent work and a fair "
                            "sequential baseline?"
                        ),
                        "evidence_request": {
                            "target": "",
                            "missing_claim": "",
                            "query": "",
                            "source_scope": "",
                            "decision_impact": "",
                        },
                        "review": dict(null_review),
                    },
                },
                {
                    "schema_name": "answer",
                    "data": {
                        "answer": (
                            "The evidence directly states sequential baseline execution "
                            "and that independent preprocessing can overlap GPU execution."
                        ),
                        "evidence_refs": [
                            baseline_claim_id,
                            opportunity_claim_id,
                        ],
                        "direct_facts": [
                            "baseline execution is sequential",
                            "preprocessing can overlap GPU execution",
                        ],
                        "inferences": [
                            "the overlap may reduce latency if overhead is smaller than hidden work"
                        ],
                        "gaps": ["overlap overhead is not quantified"],
                    },
                },
                {
                    "schema_name": "judge",
                    "data": {
                        "action": "complete",
                        "dimension": "complete",
                        "question": "",
                        "evidence_request": {
                            "target": "",
                            "missing_claim": "",
                            "query": "",
                            "source_scope": "",
                            "decision_impact": "",
                        },
                        "review": {
                            "exploration_value": "high",
                            "implementation_reuse": "middle",
                            "method_reference": "middle",
                            "baseline_quality": "middle",
                            "cross_layer_validity": "not_applicable",
                            "experiment_readiness": "ready",
                            "decision": "experiment_candidate",
                            "reasons": [
                                "The scenario and modifiable runtime object are explicit.",
                                "A current-practice baseline is retained.",
                            ],
                            "falsifiable_hypothesis": (
                                "Overlap lowers end-to-end latency when hidden preprocessing "
                                "time exceeds scheduling overhead."
                            ),
                            "implementation_plan": [
                                "add an overlap mode to the RuntimeX scheduling module",
                                "record CPU/GPU timeline and request latency",
                            ],
                            "baseline_ablation_matrix": [
                                "sequential current practice",
                                "overlap enabled",
                            ],
                            "metrics": [
                                "end-to-end latency",
                                "GPU utilization",
                                "scheduling overhead",
                            ],
                            "failure_conditions": [
                                "preprocessing depends on current GPU output",
                                "scheduling overhead exceeds hidden work",
                            ],
                            "gaps": ["a strong optimized baseline is still missing"],
                            "evidence_refs": [
                                baseline_claim_id,
                                opportunity_claim_id,
                            ],
                            "entry_refs": [
                                opportunity_entry_id,
                            ],
                            "edge_refs": [],
                            "alternative_entry_refs": [
                                baseline_entry_id,
                                implementation_entry_id,
                            ],
                            "alternative_edge_refs": [],
                            "baseline_refs": [baseline_id],
                        },
                    },
                },
            ]
            additional_review_steps = []
            for dimension, question in (
                (
                    "baseline_fairness",
                    "Is the sequential current-practice baseline comparable?",
                ),
                (
                    "entry_validity",
                    "Is the selected runtime opportunity grounded and atomic?",
                ),
                (
                    "cross_layer_validity",
                    "Does this single-layer Direction correctly avoid claiming cross-layer synergy?",
                ),
                (
                    "implementation_reuse",
                    "Is there a concrete reusable implementation path?",
                ),
                (
                    "experiment_measurement",
                    "Can the hypothesis be falsified with the named metrics and ablations?",
                ),
            ):
                additional_review_steps.extend(
                    [
                        {
                            "schema_name": "judge",
                            "data": {
                                "action": "ask",
                                "dimension": dimension,
                                "question": question,
                                "evidence_request": {
                                    "target": "",
                                    "missing_claim": "",
                                    "query": "",
                                    "source_scope": "",
                                    "decision_impact": "",
                                },
                                "review": dict(null_review),
                            },
                        },
                        {
                            "schema_name": "answer",
                            "data": {
                                "answer": (
                                    "The bundle retains the sequential baseline, the "
                                    "atomic overlap opportunity, and the RuntimeX "
                                    "implementation evidence; quantitative overhead "
                                    "remains a stated gap."
                                ),
                                "evidence_refs": [
                                    baseline_claim_id,
                                    opportunity_claim_id,
                                    implementation_claim_id,
                                ],
                                "direct_facts": [
                                    "the baseline is sequential",
                                    "the overlap opportunity is explicit",
                                    "RuntimeX provides a module and profiler",
                                ],
                                "inferences": [
                                    "the proposed ablation can isolate scheduling benefit"
                                ],
                                "gaps": ["overlap overhead is not quantified"],
                            },
                        },
                    ]
                )
            fixture[-1:-1] = additional_review_steps
            fixture_path = base / "fixture.json"
            fixture_path.write_text(
                json.dumps(fixture, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            run_dir = base / "run"
            code = orchestrator.main(
                [
                    "init",
                    "--topic",
                    "Runtime overlap acceleration",
                    "--work-dir",
                    str(run_dir),
                    "--provider",
                    "fixture",
                    "--fixture-file",
                    str(fixture_path),
                    "--knowledge-root",
                    str(knowledge_root),
                    "--layers",
                    "L2",
                    "--axes",
                    "exploration",
                    "--discovery-rounds",
                    "1",
                    "--discovery-workers",
                    "1",
                    "--max-task-attempts",
                    "1",
                    "--curator-rounds-per-batch",
                    "1",
                    "--direction-rounds-per-anchor",
                    "1",
                    "--review-rounds-per-direction",
                    "8",
                ]
            )
            self.assertEqual(0, code)
            result = orchestrator.Workflow(run_dir).run_all(resume=False)
            self.assertEqual([], result["errors"])
            self.assertTrue((run_dir / "final.md").is_file())
            final_text = (run_dir / "final.md").read_text(encoding="utf-8")
            self.assertIn(direction_id, final_text)
            self.assertIn(baseline_id, final_text)
            self.assertIn("experiment_candidate", final_text)
            rejected_mutations = (
                run_dir / "curation" / "rejected_mutations.jsonl"
            ).read_text(encoding="utf-8")
            self.assertIn("imprecise-entry", rejected_mutations)
            self.assertIn("curation_entry", rejected_mutations)
            review = json.loads(
                (run_dir / "reviews" / f"{direction_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("complete", review["status"])
            self.assertEqual(
                "experiment_candidate", review["review"]["decision"]
            )
            resumed = orchestrator.Workflow(run_dir).run_all(resume=True)
            self.assertEqual([], resumed["errors"])
            resumed_state = json.loads(
                (run_dir / "state.json").read_text(encoding="utf-8")
            )
            self.assertIn("rendered", resumed_state["completed_phases"])


if __name__ == "__main__":
    unittest.main()
