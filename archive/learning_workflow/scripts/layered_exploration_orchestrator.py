#!/usr/bin/env python3
"""Archived standalone evidence-preserving L1-L6 exploration workflow.

The legacy TypeScript schedulers are provenance only. This program owns the new
task graph, provider calls, evidence validation, canonical objects, review
loops, checkpoints, deterministic validation, and rendering.

It is retained for historical audit and is not part of the current Simple
Semantic Loop runtime.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from layered_exploration_core import (
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_KNOWLEDGE_ROOTS,
    LAYERS,
    PROTOCOL_VERSION,
    REVIEW_DIMENSIONS,
    REVIEW_SCHEMA,
    SKILL_ROOT,
    VAULT_ROOT,
    VALUE_AXES,
    LocalMarkdownRetriever,
    Provider,
    ProviderError,
    ProviderResponse,
    anchor_id_for,
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    baseline_id_for,
    build_seed_query,
    candidate_to_claim,
    canonical_json,
    direction_id_for,
    edge_id_for,
    entity_id_for,
    entry_id_for,
    format_snippets,
    install_signal_cleanup,
    load_reference,
    markdown_cell,
    normalize_key,
    normalize_text,
    provider_from_config,
    read_json,
    read_jsonl,
    resolve_source_path,
    sha256_text,
    stable_id,
    utc_now,
    validate_references,
    validate_schema,
)


TERMINAL_TASK_STATUSES = {"done", "failed_terminal", "skipped"}
DISCOVERY_TASK_STATUSES = {
    "pending", "running", "done", "failed_retriable", "failed_terminal", "skipped",
}
PHASES = (
    "initialized",
    "discovery",
    "curation",
    "direction_build",
    "direction_review",
    "validated",
    "rendered",
)
ROLE_PROVIDER_KEYS = {
    "discovery": "discovery_provider",
    "curator": "curator_provider",
    "direction": "direction_provider",
    "judge": "review_provider",
    "evidence": "review_provider",
}
BASELINE_ROLES = (
    "current_practice",
    "strong",
    "tool_evaluation",
    "reusable_implementation",
)


def _json_excerpt(value: Any, max_chars: int = 140_000) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        return (
            f"[lossless packet; {len(text)} characters exceeds the nominal "
            f"{max_chars}-character planning threshold]\n"
            + text
        )
    return text


def _csv_values(value: str, allowed: Iterable[str], label: str) -> list[str]:
    selected = [item.strip() for item in value.split(",") if item.strip()]
    allowed_set = set(allowed)
    invalid = [item for item in selected if item not in allowed_set]
    if invalid:
        raise ValueError(f"invalid {label}: {', '.join(invalid)}")
    if not selected:
        raise ValueError(f"{label} cannot be empty")
    return selected


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({normalize_text(str(item)) for item in value if normalize_text(str(item))})


def _merge_strings(left: Sequence[str], right: Sequence[str]) -> list[str]:
    return sorted({*left, *right})


def _status_choice(left: str, right: str, order: Sequence[str]) -> str:
    rank = {status: index for index, status in enumerate(order)}
    return min((left, right), key=lambda status: rank.get(status, len(order)))


def _selected_entries_are_connected(
    entry_ids: Iterable[str],
    edges: Iterable[Mapping[str, Any]],
) -> bool:
    selected = set(str(item) for item in entry_ids)
    if len(selected) <= 1:
        return True
    adjacency = {entry_id: set() for entry_id in selected}
    for edge in edges:
        from_id = str(edge.get("from_entry_id", ""))
        to_id = str(edge.get("to_entry_id", ""))
        if from_id in selected and to_id in selected:
            adjacency[from_id].add(to_id)
            adjacency[to_id].add(from_id)
    visited: set[str] = set()
    pending = [next(iter(selected))]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency[current] - visited)
    return visited == selected


def _review_semantic_errors(
    review: Mapping[str, Any],
    direction: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not review.get("reasons"):
        errors.append("complete review has no reasons")
    if not review.get("evidence_refs"):
        errors.append("complete review has no evidence refs")
    if review.get("decision") == "experiment_candidate":
        if review.get("exploration_value") == "low":
            errors.append(
                "experiment_candidate cannot have low exploration_value"
            )
        if review.get("baseline_quality") not in ("high", "middle"):
            errors.append(
                "experiment_candidate requires high/middle baseline_quality"
            )
        if review.get("experiment_readiness") != "ready":
            errors.append("experiment_candidate requires experiment_readiness=ready")
        if not direction.get("baseline_ids"):
            errors.append("experiment_candidate Direction has no baseline_ids")
        if not normalize_text(str(review.get("falsifiable_hypothesis", ""))):
            errors.append("experiment_candidate has no falsifiable hypothesis")
        for field in (
            "implementation_plan",
            "baseline_ablation_matrix",
            "metrics",
            "failure_conditions",
        ):
            if not review.get(field):
                errors.append(f"experiment_candidate has empty {field}")
    return errors


def _command_probe(command: Sequence[str], timeout: int = 15) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            cwd=str(VAULT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "detail": str(exc)}
    output = normalize_text((result.stdout or "") + " " + (result.stderr or ""))
    return {
        "ok": result.returncode == 0,
        "return_code": result.returncode,
        "detail": output[:500],
    }


def doctor_report() -> dict[str, Any]:
    claude_path = shutil.which("claude")
    codex_path = shutil.which("codex")
    rg_path = shutil.which("rg")
    claude_version = _command_probe(["claude", "--version"]) if claude_path else {"ok": False}
    codex_version = _command_probe(["codex", "--version"]) if codex_path else {"ok": False}
    codex_login = _command_probe(["codex", "login", "status"]) if codex_path else {"ok": False}
    try:
        openai_version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        openai_version = ""
    skill_files = [
        SKILL_ROOT / "SKILL.md",
        SKILL_ROOT / "references" / "domain-model.md",
        SKILL_ROOT / "references" / "orchestration-protocol.md",
    ]
    deepseek_env = {
        "ANTHROPIC_BASE_URL": bool(os.environ.get("ANTHROPIC_BASE_URL")),
        "ANTHROPIC_AUTH_TOKEN": bool(os.environ.get("ANTHROPIC_AUTH_TOKEN")),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "local": {
            "python": platform.python_version(),
            "vault_root": str(VAULT_ROOT),
            "vault_writable": os.access(VAULT_ROOT, os.W_OK),
            "rg": {"path": rg_path or "", "ok": bool(rg_path)},
            "skill": {
                "path": str(SKILL_ROOT),
                "ok": all(path.is_file() for path in skill_files),
                "missing": [str(path) for path in skill_files if not path.is_file()],
            },
        },
        "providers": {
            "deepseek-cli": {
                "binary": claude_path or "",
                "version": claude_version,
                "credential_variables_present": deepseek_env,
                "ready": bool(
                    claude_path
                    and claude_version.get("ok")
                    and all(deepseek_env.values())
                ),
                "live_call_verified": False,
                "default_model": DEFAULT_DEEPSEEK_MODEL,
            },
            "codex-cli": {
                "binary": codex_path or "",
                "version": codex_version,
                "login": codex_login,
                "ready": bool(
                    codex_path
                    and codex_version.get("ok")
                    and codex_login.get("ok")
                ),
                "live_call_verified": False,
            },
            "direct-openai-api-informational": {
                "python_sdk_version": openai_version,
                "OPENAI_API_KEY_present": bool(os.environ.get("OPENAI_API_KEY")),
                "ready": bool(openai_version and os.environ.get("OPENAI_API_KEY")),
                "used_by_this_workflow": False,
            },
            "fixture": {"ready": True, "tests_only": True},
        },
    }


def print_doctor(report: Mapping[str, Any]) -> None:
    print(f"workflow protocol: v{report['protocol_version']}")
    local = report["local"]
    print(
        "local: "
        f"python={local['python']} rg={'ready' if local['rg']['ok'] else 'missing'} "
        f"skill={'ready' if local['skill']['ok'] else 'missing'} "
        f"vault_write={'yes' if local['vault_writable'] else 'no'}"
    )
    for name in ("deepseek-cli", "codex-cli", "direct-openai-api-informational", "fixture"):
        provider = report["providers"][name]
        print(f"{name}: {'locally ready' if provider['ready'] else 'not ready'}")
    print("Credential values were not read or printed.")


def _initial_catalog() -> dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, "entities": []}


def _initial_anchors() -> dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, "anchors": []}


def _initial_directions() -> dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, "directions": []}


def canonical_evidence_refs(work_dir: Path) -> set[str]:
    refs: set[str] = set()
    catalog = read_json(work_dir / "catalog" / "entities.json", _initial_catalog())
    for entity in catalog.get("entities", []):
        refs.update(str(item) for item in entity.get("evidence_refs", []))
    anchors_doc = read_json(work_dir / "anchors" / "anchors.json", _initial_anchors())
    for anchor in anchors_doc.get("anchors", []):
        refs.update(str(item) for item in anchor.get("evidence_refs", []))
        for field in ("baselines", "entries", "edges"):
            for item in anchor.get(field, []):
                refs.update(str(ref) for ref in item.get("evidence_refs", []))
    directions_doc = read_json(
        work_dir / "directions" / "directions.json",
        _initial_directions(),
    )
    for direction in directions_doc.get("directions", []):
        refs.update(str(item) for item in direction.get("evidence_refs", []))
    for path in sorted((work_dir / "reviews").glob("*.json")):
        artifact = read_json(path, {})
        review = artifact.get("review", {}) if isinstance(artifact, dict) else {}
        refs.update(str(item) for item in review.get("evidence_refs", []))
        for item in artifact.get("qa", []) if isinstance(artifact, dict) else []:
            refs.update(str(ref) for ref in item.get("accepted_claim_ids", []))
            answer = item.get("answer", {})
            if isinstance(answer, dict):
                refs.update(str(ref) for ref in answer.get("evidence_refs", []))
    return refs


def reconcile_dispositions(work_dir: Path) -> None:
    claims = read_jsonl(work_dir / "evidence" / "claims.jsonl")
    claim_ids = {str(item["claim_id"]) for item in claims}
    used_refs = canonical_evidence_refs(work_dir)
    path = work_dir / "curation" / "dispositions.json"
    document = read_json(path, {"dispositions": []})
    index = {
        str(item["claim_id"]): item
        for item in document.get("dispositions", [])
        if str(item.get("claim_id", "")) in claim_ids
    }
    for claim_id in sorted(claim_ids):
        previous = index.get(claim_id, {})
        if claim_id in used_refs:
            index[claim_id] = {
                "claim_id": claim_id,
                "status": "integrated",
                "reason": (
                    previous.get("reason")
                    if previous.get("status") == "integrated"
                    else "deterministic canonical artifact evidence reference"
                ),
                "updated_at": utc_now(),
            }
        elif previous.get("status") == "integrated":
            index[claim_id] = {
                "claim_id": claim_id,
                "status": "needs_evidence",
                "reason": (
                    "model marked integrated, but no admitted canonical artifact "
                    "references this claim"
                ),
                "updated_at": utc_now(),
            }
        elif not previous:
            index[claim_id] = {
                "claim_id": claim_id,
                "status": "needs_evidence",
                "reason": "no admitted canonical artifact references this claim",
                "updated_at": utc_now(),
            }
    atomic_write_json(
        path,
        {
            "dispositions": sorted(
                index.values(), key=lambda item: item["claim_id"]
            )
        },
    )


def initialize_run(args: argparse.Namespace) -> Path:
    work_dir = Path(args.work_dir).resolve()
    if not normalize_text(args.topic):
        raise ValueError("topic cannot be empty")
    if work_dir == VAULT_ROOT.resolve():
        raise ValueError("work-dir must be a dedicated subdirectory, not the vault root")
    if work_dir.exists() and any(work_dir.iterdir()):
        raise ValueError(f"work-dir is not empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    layers = _csv_values(args.layers, LAYERS.keys(), "layers")
    axes = _csv_values(args.axes, VALUE_AXES, "axes")
    knowledge_roots = list(args.knowledge_root or DEFAULT_KNOWLEDGE_ROOTS)
    for root in knowledge_roots:
        candidate = Path(root)
        if not candidate.is_absolute():
            candidate = VAULT_ROOT / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(VAULT_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"knowledge root must be inside vault: {root}") from exc
        if not resolved.is_dir():
            raise ValueError(f"knowledge root does not exist or is not a directory: {root}")

    positive_fields = (
        "provider_timeout_seconds",
        "discovery_rounds",
        "discovery_workers",
        "snippets_per_round",
        "snippet_context_lines",
        "max_task_attempts",
        "curation_batch_size",
        "curator_rounds_per_batch",
        "direction_rounds_per_anchor",
        "review_rounds_per_direction",
    )
    for field in positive_fields:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    nonnegative_fields = (
        "max_budget_usd_per_call",
        "max_curator_evidence_requests",
        "max_direction_evidence_requests",
        "max_review_evidence_requests",
    )
    for field in nonnegative_fields:
        if float(getattr(args, field)) < 0:
            raise ValueError(f"--{field.replace('_', '-')} cannot be negative")

    base_provider = args.provider
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "topic": normalize_text(args.topic),
        "constraints": normalize_text(args.constraints or ""),
        "work_dir": str(work_dir),
        "knowledge_roots": knowledge_roots,
        "layers": layers,
        "axes": axes,
        "discovery_provider": args.discovery_provider or base_provider,
        "curator_provider": args.curator_provider or base_provider,
        "direction_provider": args.direction_provider or args.curator_provider or base_provider,
        "review_provider": args.review_provider or base_provider,
        "deepseek_model": args.deepseek_model,
        "codex_model": args.codex_model or "",
        "fixture_file": str(Path(args.fixture_file).resolve()) if args.fixture_file else "",
        "provider_timeout_seconds": args.provider_timeout_seconds,
        "max_budget_usd_per_call": args.max_budget_usd_per_call,
        "discovery_rounds": args.discovery_rounds,
        "discovery_workers": args.discovery_workers,
        "snippets_per_round": args.snippets_per_round,
        "snippet_context_lines": args.snippet_context_lines,
        "max_task_attempts": args.max_task_attempts,
        "curation_batch_size": args.curation_batch_size,
        "curator_rounds_per_batch": args.curator_rounds_per_batch,
        "max_curator_evidence_requests": args.max_curator_evidence_requests,
        "direction_rounds_per_anchor": args.direction_rounds_per_anchor,
        "max_direction_evidence_requests": args.max_direction_evidence_requests,
        "review_rounds_per_direction": args.review_rounds_per_direction,
        "max_review_evidence_requests": args.max_review_evidence_requests,
        "review_session_mode": args.review_session_mode,
    }
    config["synthetic_run"] = any(
        config[key] == "fixture"
        for key in (
            "discovery_provider", "curator_provider",
            "direction_provider", "review_provider",
        )
    )
    if any(
        config[key] == "fixture"
        for key in (
            "discovery_provider", "curator_provider",
            "direction_provider", "review_provider",
        )
    ) and not config["fixture_file"]:
        raise ValueError("fixture provider requires --fixture-file")
    if any(
        config[key] == "deepseek-cli"
        for key in (
            "discovery_provider", "curator_provider",
            "direction_provider", "review_provider",
        )
    ) and float(config["max_budget_usd_per_call"]) <= 0:
        raise ValueError(
            "--max-budget-usd-per-call must be positive when DeepSeek is selected"
        )
    if config["fixture_file"] and not Path(config["fixture_file"]).is_file():
        raise ValueError(f"fixture file does not exist: {config['fixture_file']}")

    run_id = stable_id("RUN", config["topic"], config["created_at"], str(work_dir))
    tasks: dict[str, dict[str, Any]] = {}
    for layer in layers:
        for axis in axes:
            task_id = f"DISC-{layer}-{axis}"
            tasks[task_id] = {
                "task_id": task_id,
                "layer": layer,
                "axis": axis,
                "status": "pending",
                "attempts": 0,
                "session_id": "",
                "session_provider": "",
                "session_model": "",
                "artifact": f"tasks/discovery/{task_id}.json",
                "accepted_claim_ids": [],
                "gaps": [],
                "last_error": "",
                "updated_at": config["created_at"],
            }
    state = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "topic": config["topic"],
        "synthetic_run": config["synthetic_run"],
        "phase": "initialized",
        "status": "initialized",
        "created_at": config["created_at"],
        "updated_at": config["created_at"],
        "completed_phases": [],
        "tasks": tasks,
        "sessions": {
            "curators": {},
            "directions": {},
            "judge": {},
            "evidence": {},
        },
        "counters": {
            "targeted_evidence_requests": 0,
        },
        "gaps": [],
    }
    for relative in (
        "tasks/discovery",
        "retrieval",
        "evidence",
        "catalog",
        "anchors",
        "directions",
        "curation",
        "reviews",
        "logs/model_calls",
        "provider_sandbox",
    ):
        (work_dir / relative).mkdir(parents=True, exist_ok=True)
    atomic_write_json(work_dir / "config.json", config)
    atomic_write_json(work_dir / "state.json", state)
    atomic_write_text(work_dir / "events.jsonl", "")
    atomic_write_text(work_dir / "evidence" / "claims.jsonl", "")
    atomic_write_text(work_dir / "evidence" / "rejected_claims.jsonl", "")
    atomic_write_text(work_dir / "curation" / "rejected_mutations.jsonl", "")
    atomic_write_json(work_dir / "curation" / "dispositions.json", {"dispositions": []})
    atomic_write_json(work_dir / "catalog" / "entities.json", _initial_catalog())
    atomic_write_json(work_dir / "anchors" / "anchors.json", _initial_anchors())
    atomic_write_json(work_dir / "directions" / "directions.json", _initial_directions())
    append_jsonl(
        work_dir / "events.jsonl",
        {
            "timestamp": utc_now(),
            "phase": "initialized",
            "event": "run_initialized",
            "run_id": run_id,
            "config_hash": sha256_text(canonical_json(config)),
        },
    )
    return work_dir


class Workflow:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir.resolve()
        install_signal_cleanup()
        self.config_path = self.work_dir / "config.json"
        self.state_path = self.work_dir / "state.json"
        self.events_path = self.work_dir / "events.jsonl"
        self.config = read_json(self.config_path)
        self.state = read_json(self.state_path)
        if not isinstance(self.config, dict) or not isinstance(self.state, dict):
            raise ValueError(f"not an initialized run directory: {self.work_dir}")
        if self.config.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported config protocol {self.config.get('protocol_version')}; "
                f"expected {PROTOCOL_VERSION}"
            )
        if self.state.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported state protocol {self.state.get('protocol_version')}; "
                f"expected {PROTOCOL_VERSION}"
            )
        self.state_lock = threading.RLock()
        self.claim_lock = threading.RLock()
        self.provider_lock = threading.Lock()
        self.target_lock = threading.Lock()
        self.providers: dict[str, Provider] = {}
        self.retriever = LocalMarkdownRetriever(
            VAULT_ROOT,
            [str(item) for item in self.config.get("knowledge_roots", DEFAULT_KNOWLEDGE_ROOTS)],
        )
        self.claim_ids = {
            str(row.get("claim_id"))
            for row in read_jsonl(self.work_dir / "evidence" / "claims.jsonl")
            if row.get("claim_id")
        }
        self._target_counter = int(
            self.state.get("counters", {}).get("targeted_evidence_requests", 0)
        )

    def event(self, event: str, **fields: Any) -> None:
        append_jsonl(
            self.events_path,
            {
                "timestamp": utc_now(),
                "phase": self.state.get("phase", ""),
                "event": event,
                **fields,
            },
        )

    def checkpoint(self) -> None:
        with self.state_lock:
            self.state["updated_at"] = utc_now()
            atomic_write_json(self.state_path, self.state)

    def add_gap(self, gap: str) -> None:
        normalized = normalize_text(gap)
        with self.state_lock:
            gaps = set(str(item) for item in self.state.get("gaps", []))
            gaps.add(normalized)
            self.state["gaps"] = sorted(gaps)
            self.checkpoint()
        self.event("workflow_gap_recorded", gap=normalized)

    def set_phase(self, phase: str, status: str = "running") -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase}")
        with self.state_lock:
            self.state["phase"] = phase
            self.state["status"] = status
            self.checkpoint()
        self.event("phase_changed", target_phase=phase, status=status)

    def mark_phase_completed(self, phase: str) -> None:
        with self.state_lock:
            completed = set(str(item) for item in self.state.get("completed_phases", []))
            completed.add(phase)
            self.state["completed_phases"] = [
                item for item in PHASES if item in completed
            ]
            self.checkpoint()
        self.event("phase_completed", completed_phase=phase)

    def provider(self, role: str) -> Provider:
        key = ROLE_PROVIDER_KEYS[role]
        provider_name = str(self.config[key])
        with self.provider_lock:
            provider = self.providers.get(provider_name)
            if provider is None:
                provider = provider_from_config(provider_name, self.work_dir, self.config)
                self.providers[provider_name] = provider
            return provider

    def resume_session_id(
        self,
        *,
        role: str,
        record: Any,
        context: str,
    ) -> str:
        if not isinstance(record, dict) or not record.get("session_id"):
            return ""
        provider = self.provider(role)
        expected_provider = provider.name
        expected_model = str(getattr(provider, "model", "") or "")
        stored_provider = str(record.get("provider", ""))
        stored_model = str(record.get("model", ""))
        if stored_provider != expected_provider or (
            expected_model
            and stored_model not in (expected_model, "<config-default>")
        ):
            self.event(
                "session_not_resumed",
                role=role,
                context=context,
                stored_provider=stored_provider,
                configured_provider=expected_provider,
                stored_model=stored_model,
                configured_model=expected_model,
                reason="provider/model mismatch",
            )
            return ""
        return str(record["session_id"])

    @staticmethod
    def session_record(response: ProviderResponse) -> dict[str, str]:
        return {
            "provider": response.provider,
            "model": response.model,
            "session_id": response.session_id,
        }

    def call(
        self,
        *,
        role: str,
        prompt: str,
        schema_name: str,
        conversation_key: str,
        session_id: str = "",
    ) -> ProviderResponse:
        provider = self.provider(role)
        response = provider.call(
            role=role,
            prompt=prompt,
            schema_name=schema_name,
            conversation_key=conversation_key,
            session_id=session_id,
        )
        self.event(
            "provider_call_completed",
            role=role,
            provider=response.provider,
            model=response.model,
            conversation_key=conversation_key,
            session_id=response.session_id,
            raw_log_path=response.raw_log_path,
            telemetry=response.telemetry,
        )
        return response

    def reset_for_resume(self) -> None:
        invalidate_from = ""
        with self.state_lock:
            for task in self.state.get("tasks", {}).values():
                status = task.get("status")
                artifact = self.work_dir / str(task.get("artifact", ""))
                if status == "running" or status == "failed_retriable":
                    task["status"] = "pending"
                    task["last_error"] = "reset during resume"
                    invalidate_from = "discovery"
                elif status == "done" and not artifact.is_file():
                    task["status"] = "pending"
                    task["last_error"] = "done artifact missing during resume"
                    invalidate_from = "discovery"
                if task.get("status") not in DISCOVERY_TASK_STATUSES:
                    task["status"] = "pending"
                    task["last_error"] = "invalid task status reset during resume"
                    invalidate_from = "discovery"
                task["updated_at"] = utc_now()

            completed = set(
                str(item) for item in self.state.get("completed_phases", [])
            )
            if "curation" in completed and not all(
                path.is_file()
                for path in (
                    self.work_dir / "catalog" / "entities.json",
                    self.work_dir / "anchors" / "anchors.json",
                    self.work_dir / "curation" / "dispositions.json",
                )
            ):
                invalidate_from = invalidate_from or "curation"
            if "direction_build" in completed and not (
                self.work_dir / "directions" / "directions.json"
            ).is_file():
                invalidate_from = invalidate_from or "direction_build"
            if "direction_review" in completed:
                directions_doc = read_json(
                    self.work_dir / "directions" / "directions.json",
                    _initial_directions(),
                )
                review_incomplete = False
                for direction in directions_doc.get("directions", []):
                    artifact = read_json(
                        self.work_dir
                        / "reviews"
                        / f"{direction['direction_id']}.json"
                    )
                    if not isinstance(artifact, dict) or artifact.get("status") != "complete":
                        review_incomplete = True
                        break
                if review_incomplete:
                    invalidate_from = invalidate_from or "direction_review"

            if invalidate_from:
                index = PHASES.index(invalidate_from)
                completed = {
                    phase for phase in completed if PHASES.index(phase) < index
                }
            self.state["completed_phases"] = [
                phase for phase in PHASES if phase in completed
            ]
            self.checkpoint()
        self.event("resume_state_reconciled", invalidated_from=invalidate_from)

    def _update_task(self, task_id: str, **changes: Any) -> None:
        with self.state_lock:
            task = self.state["tasks"][task_id]
            task.update(changes)
            task["updated_at"] = utc_now()
            self.checkpoint()
        self.event(
            "discovery_task_transition",
            task_id=task_id,
            status=changes.get("status", task.get("status")),
            attempt=task.get("attempts", 0),
            error_summary=changes.get("last_error", ""),
        )

    def _append_claim(
        self,
        candidate: Mapping[str, Any],
        *,
        origin: Mapping[str, Any],
        allowed_snippets: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[str | None, list[str]]:
        packet_errors: list[str] = []
        if allowed_snippets is not None:
            try:
                candidate_path = resolve_source_path(
                    str(candidate.get("source_path", "")),
                    VAULT_ROOT,
                )
                candidate_start = int(candidate.get("line_start", 0))
                candidate_end = int(candidate.get("line_end", 0))
                permitted = False
                for snippet in allowed_snippets:
                    snippet_path = resolve_source_path(
                        str(snippet.get("source_path", "")),
                        VAULT_ROOT,
                    )
                    if (
                        snippet_path == candidate_path
                        and candidate_start >= int(snippet.get("line_start", 0))
                        and candidate_end <= int(snippet.get("line_end", 0))
                    ):
                        permitted = True
                        break
                if not permitted:
                    packet_errors.append(
                        "citation is outside the evidence snippets supplied to this worker"
                    )
            except (TypeError, ValueError):
                packet_errors.append(
                    "citation path/line coordinates cannot be matched to the evidence packet"
                )
        if packet_errors:
            rejection = {
                "rejection_id": stable_id(
                    "RC",
                    origin,
                    candidate,
                    packet_errors,
                    utc_now(),
                ),
                "origin": dict(origin),
                "candidate": dict(candidate),
                "errors": packet_errors,
                "rejected_at": utc_now(),
            }
            append_jsonl(
                self.work_dir / "evidence" / "rejected_claims.jsonl",
                rejection,
            )
            return None, packet_errors
        claim, errors = candidate_to_claim(candidate, vault_root=VAULT_ROOT)
        if claim is None:
            rejection = {
                "rejection_id": stable_id(
                    "RC",
                    origin,
                    candidate,
                    errors,
                    utc_now(),
                ),
                "origin": dict(origin),
                "candidate": dict(candidate),
                "errors": errors,
                "rejected_at": utc_now(),
            }
            append_jsonl(self.work_dir / "evidence" / "rejected_claims.jsonl", rejection)
            return None, errors
        claim_id = str(claim["claim_id"])
        with self.claim_lock:
            if claim_id not in self.claim_ids:
                append_jsonl(self.work_dir / "evidence" / "claims.jsonl", claim)
                self.claim_ids.add(claim_id)
                self.event("evidence_claim_accepted", claim_id=claim_id, origin=dict(origin))
        return claim_id, []

    def _discovery_prompt(
        self,
        *,
        task: Mapping[str, Any],
        query: str,
        snippets_text: str,
        existing_claims: Sequence[Mapping[str, Any]],
        round_number: int,
    ) -> str:
        layer = str(task["layer"])
        axis = str(task["axis"])
        return (
            "# Role contract\n"
            + load_reference("discovery-worker.md")
            + "\n\n# Current task\n"
            f"topic: {self.config['topic']}\n"
            f"constraints: {self.config.get('constraints', '')}\n"
            f"layer: {layer} — {LAYERS[layer]['name']}\n"
            f"layer scope: {LAYERS[layer]['scope']}\n"
            f"value axis: {axis}\n"
            f"round: {round_number}/{self.config['discovery_rounds']}\n"
            f"retrieval query: {query}\n\n"
            "# Already accepted for this task\n"
            + _json_excerpt(existing_claims, 30_000)
            + "\n\n# Line-numbered evidence packet\n"
            + snippets_text
            + "\n\n# Output instruction\n"
            "Return exactly the schema object. Every candidate quote must occur inside "
            "the declared source path and line range above. Preserve baseline candidates "
            "regardless of novelty. Do not construct Anchors or Directions."
        )

    def _run_discovery_attempt(self, task_id: str) -> None:
        task = dict(self.state["tasks"][task_id])
        layer = str(task["layer"])
        axis = str(task["axis"])
        seed_query = build_seed_query(
            str(self.config["topic"]),
            layer,
            axis,
            str(self.config.get("constraints", "")),
        )
        query_queue = [seed_query]
        seen_queries: set[str] = set()
        used_snippets: set[str] = set()
        accepted_ids = set(str(item) for item in task.get("accepted_claim_ids", []))
        gaps = list(task.get("gaps", []))
        rounds: list[dict[str, Any]] = []
        allowed_snippets: list[dict[str, Any]] = []
        session_id = str(task.get("session_id", ""))
        session_id = self.resume_session_id(
            role="discovery",
            record={
                "provider": task.get("session_provider", ""),
                "model": task.get("session_model", ""),
                "session_id": session_id,
            },
            context=task_id,
        )

        for round_number in range(1, int(self.config["discovery_rounds"]) + 1):
            if not query_queue:
                gaps.append("discovery query queue is empty")
                break
            query = query_queue.pop(0)
            query_key = normalize_key(query)
            if not query_key or query_key in seen_queries:
                gaps.append(f"no-progress retrieval query at round {round_number}: {query}")
                break
            seen_queries.add(query_key)
            snippets = self.retriever.search(
                query,
                limit=int(self.config["snippets_per_round"]),
                context_lines=int(self.config["snippet_context_lines"]),
                exclude_snippets=used_snippets,
            )
            for snippet in snippets:
                used_snippets.add(snippet.snippet_id)
                allowed_snippets.append(snippet.as_dict())
            retrieval_path = self.work_dir / "retrieval" / f"{task_id}_r{round_number:02d}.json"
            atomic_write_json(
                retrieval_path,
                {
                    "task_id": task_id,
                    "round": round_number,
                    "query": query,
                    "snippets": [snippet.as_dict() for snippet in snippets],
                },
            )
            if not snippets:
                gaps.append(f"no local evidence found for query: {query}")
                break

            all_claims = read_jsonl(self.work_dir / "evidence" / "claims.jsonl")
            existing_claims = [
                claim for claim in all_claims if str(claim.get("claim_id")) in accepted_ids
            ]
            response = self.call(
                role="discovery",
                prompt=self._discovery_prompt(
                    task=task,
                    query=query,
                    snippets_text=format_snippets(snippets),
                    existing_claims=existing_claims,
                    round_number=round_number,
                ),
                schema_name="discovery",
                conversation_key=f"{task_id}:r{round_number}",
                session_id=session_id,
            )
            session_id = response.session_id
            valid_this_round: list[str] = []
            invalid_this_round: list[dict[str, Any]] = []
            for candidate_index, candidate in enumerate(response.data["candidates"]):
                claim_id, errors = self._append_claim(
                    candidate,
                    origin={
                        "kind": "discovery",
                        "task_id": task_id,
                        "round": round_number,
                        "candidate_index": candidate_index,
                    },
                    allowed_snippets=allowed_snippets,
                )
                if claim_id:
                    accepted_ids.add(claim_id)
                    valid_this_round.append(claim_id)
                else:
                    invalid_this_round.append(
                        {"candidate_index": candidate_index, "errors": errors}
                    )
            gaps.extend(_string_list(response.data.get("gaps")))
            rounds.append(
                {
                    "round": round_number,
                    "query": query,
                    "retrieval_artifact": str(retrieval_path.relative_to(self.work_dir)),
                    "accepted_claim_ids": valid_this_round,
                    "invalid_candidates": invalid_this_round,
                    "provider": response.provider,
                    "model": response.model,
                    "session_id": response.session_id,
                    "action": response.data["action"],
                    "next_queries": response.data["next_queries"],
                }
            )
            self._update_task(
                task_id,
                status="running",
                session_id=session_id,
                session_provider=response.provider,
                session_model=response.model,
                accepted_claim_ids=sorted(accepted_ids),
                gaps=sorted(set(gaps)),
            )
            if response.data["action"] == "complete":
                break
            added_queries = 0
            for candidate_query in response.data.get("next_queries", []):
                candidate_key = normalize_key(candidate_query)
                queued_keys = {normalize_key(item) for item in query_queue}
                if candidate_key and candidate_key not in seen_queries and candidate_key not in queued_keys:
                    query_queue.append(candidate_query)
                    added_queries += 1
            if not added_queries:
                gaps.append("worker requested continuation without a novel targeted query")
                break
        if query_queue:
            gaps.append(
                f"{len(query_queue)} targeted discovery query/queries remained "
                "unexecuted at the round budget"
            )

        artifact = {
            "task_id": task_id,
            "layer": layer,
            "axis": axis,
            "status": "done",
            "accepted_claim_ids": sorted(accepted_ids),
            "gaps": sorted(set(gaps)),
            "rounds": rounds,
            "unexecuted_queries": query_queue,
            "completed_at": utc_now(),
        }
        artifact_path = self.work_dir / str(task["artifact"])
        atomic_write_json(artifact_path, artifact)
        self._update_task(
            task_id,
            status="done",
            session_id=session_id,
            session_provider=(
                rounds[-1]["provider"] if rounds else task.get("session_provider", "")
            ),
            session_model=(
                rounds[-1]["model"] if rounds else task.get("session_model", "")
            ),
            accepted_claim_ids=sorted(accepted_ids),
            gaps=sorted(set(gaps)),
            last_error="",
        )

    def _run_discovery_task(self, task_id: str) -> None:
        maximum = int(self.config["max_task_attempts"])
        while int(self.state["tasks"][task_id].get("attempts", 0)) < maximum:
            attempts = int(self.state["tasks"][task_id].get("attempts", 0)) + 1
            self._update_task(task_id, status="running", attempts=attempts, last_error="")
            try:
                self._run_discovery_attempt(task_id)
                return
            except Exception as exc:  # noqa: BLE001 - isolate independent task failures
                status = "failed_terminal" if attempts >= maximum else "failed_retriable"
                error = f"{type(exc).__name__}: {exc}"
                self._update_task(task_id, status=status, last_error=error)
                self.event(
                    "discovery_task_failed",
                    task_id=task_id,
                    attempt=attempts,
                    status=status,
                    error_summary=error,
                    traceback=traceback.format_exc(limit=8),
                )
                if status == "failed_terminal":
                    return

    def run_discovery(self) -> None:
        self.set_phase("discovery")
        task_ids = [
            task_id
            for task_id, task in sorted(self.state["tasks"].items())
            if task.get("status") not in TERMINAL_TASK_STATUSES
        ]
        workers = max(1, int(self.config["discovery_workers"]))
        if workers == 1:
            for task_id in task_ids:
                self._run_discovery_task(task_id)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._run_discovery_task, task_id): task_id
                    for task_id in task_ids
                }
                for future in concurrent.futures.as_completed(futures):
                    task_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:  # defensive: task normally contains failures
                        self.event(
                            "discovery_worker_crashed",
                            task_id=task_id,
                            error_summary=f"{type(exc).__name__}: {exc}",
                        )
        statuses = [
            str(task.get("status")) for task in self.state["tasks"].values()
        ]
        incomplete = [
            status for status in statuses if status not in TERMINAL_TASK_STATUSES
        ]
        terminal_failures = statuses.count("failed_terminal")
        with self.state_lock:
            self.state["status"] = "partial" if incomplete or terminal_failures else "running"
            self.checkpoint()
        self.event(
            "discovery_completed",
            total=len(statuses),
            terminal_failures=terminal_failures,
            incomplete=len(incomplete),
            claim_count=len(self.claim_ids),
        )
        self.mark_phase_completed("discovery")

    def _next_target_id(self, scope: str) -> str:
        with self.target_lock:
            self._target_counter += 1
            counter = self._target_counter
            with self.state_lock:
                self.state.setdefault("counters", {})[
                    "targeted_evidence_requests"
                ] = self._target_counter
                self.checkpoint()
        return f"TARGET-{counter:04d}-{normalize_key(scope)[:30]}"

    def targeted_discovery(
        self,
        request: Mapping[str, Any],
        *,
        scope: str,
    ) -> list[str]:
        target_id = self._next_target_id(scope)
        query = normalize_text(
            str(request.get("query") or request.get("missing_claim") or request.get("target"))
        )
        if not query:
            self.event("targeted_evidence_skipped", target_id=target_id, reason="empty query")
            return []
        snippets = self.retriever.search(
            query,
            limit=int(self.config["snippets_per_round"]),
            context_lines=int(self.config["snippet_context_lines"]),
        )
        retrieval_path = self.work_dir / "retrieval" / f"{target_id}.json"
        atomic_write_json(
            retrieval_path,
            {
                "target_id": target_id,
                "scope": scope,
                "request": dict(request),
                "query": query,
                "snippets": [snippet.as_dict() for snippet in snippets],
            },
        )
        if not snippets:
            self.add_gap(
                f"targeted evidence {target_id} found no local support for: {query}"
            )
            self.event(
                "targeted_evidence_completed",
                target_id=target_id,
                accepted_claim_ids=[],
                gap="no matching local evidence",
            )
            return []
        prompt = (
            "# Role contract\n"
            + load_reference("discovery-worker.md")
            + "\n\n# Targeted evidence request\n"
            + _json_excerpt(dict(request), 12_000)
            + f"\nworkflow scope: {scope}\n"
            f"topic: {self.config['topic']}\n\n"
            "# Evidence packet\n"
            + format_snippets(snippets)
            + "\n\nReturn atomic candidates only for the named missing proposition. "
            "Use action=complete; next_queries may name a remaining exact gap."
        )
        response = self.call(
            role="discovery",
            prompt=prompt,
            schema_name="discovery",
            conversation_key=target_id,
        )
        accepted: list[str] = []
        for index, candidate in enumerate(response.data["candidates"]):
            claim_id, _errors = self._append_claim(
                candidate,
                origin={
                    "kind": "targeted",
                    "target_id": target_id,
                    "scope": scope,
                    "candidate_index": index,
                },
                allowed_snippets=[snippet.as_dict() for snippet in snippets],
            )
            if claim_id:
                accepted.append(claim_id)
        self.event(
            "targeted_evidence_completed",
            target_id=target_id,
            accepted_claim_ids=sorted(set(accepted)),
            gaps=response.data.get("gaps", []),
        )
        for gap in response.data.get("gaps", []):
            self.add_gap(f"targeted evidence {target_id}: {gap}")
        return sorted(set(accepted))

    def _curator_prompt(
        self,
        *,
        batch_index: int,
        batch_count: int,
        claims: Sequence[Mapping[str, Any]],
        new_claims: Sequence[Mapping[str, Any]],
        action_context: str,
    ) -> str:
        catalog = read_json(self.work_dir / "catalog" / "entities.json", _initial_catalog())
        anchors = read_json(self.work_dir / "anchors" / "anchors.json", _initial_anchors())
        return (
            "# Domain and value rules\n"
            + load_reference("domain-model.md")
            + "\n\n"
            + load_reference("layer-value-rubric.md")
            + "\n\n# Curator role\n"
            + load_reference("curator-loop.md")
            + "\n\n# Run context\n"
            f"topic: {self.config['topic']}\n"
            f"constraints: {self.config.get('constraints', '')}\n"
            f"claim batch: {batch_index}/{batch_count}\n"
            f"loop context: {action_context}\n\n"
            "# Existing canonical entity catalog\n"
            + _json_excerpt(catalog, 35_000)
            + "\n\n# Existing canonical Anchors\n"
            + _json_excerpt(anchors, 70_000)
            + "\n\n# Claims in this batch\n"
            + _json_excerpt(list(claims), 65_000)
            + "\n\n# Newly retrieved targeted claims\n"
            + _json_excerpt(list(new_claims), 25_000)
            + "\n\n# Mutation protocol\n"
            "Use temporary anchor_key/baseline_key/entry_key values only inside this "
            "response. Re-propose the complete baseline set and all entries needed by "
            "each proposed edge. Every retained object must cite existing claim IDs. "
            "Keep evidence-grounded baselines even with low exploration novelty. "
            "Disposition every supplied batch claim when possible. Return one schema object."
        )

    def _record_mutation_rejection(
        self,
        *,
        phase: str,
        proposal: Mapping[str, Any],
        errors: Sequence[str],
    ) -> None:
        append_jsonl(
            self.work_dir / "curation" / "rejected_mutations.jsonl",
            {
                "rejection_id": stable_id("RM", phase, proposal, list(errors), utc_now()),
                "phase": phase,
                "proposal": dict(proposal),
                "errors": list(errors),
                "rejected_at": utc_now(),
            },
        )

    def _merge_entity(
        self,
        index: dict[str, dict[str, Any]],
        entity: dict[str, Any],
    ) -> None:
        previous = index.get(entity["entity_id"])
        if previous is None:
            index[entity["entity_id"]] = entity
            return
        aliases = list(entity.get("aliases", []))
        if entity.get("name") != previous.get("name"):
            aliases.append(str(entity.get("name", "")))
        previous["aliases"] = _merge_strings(previous.get("aliases", []), aliases)
        previous["evidence_refs"] = _merge_strings(
            previous.get("evidence_refs", []), entity.get("evidence_refs", [])
        )
        previous["updated_at"] = utc_now()

    def _merge_anchor(
        self,
        index: dict[str, dict[str, Any]],
        anchor: dict[str, Any],
    ) -> None:
        previous = index.get(anchor["anchor_id"])
        if previous is None:
            index[anchor["anchor_id"]] = anchor
            return
        previous["evidence_refs"] = _merge_strings(
            previous.get("evidence_refs", []), anchor.get("evidence_refs", [])
        )
        previous["gaps"] = _merge_strings(previous.get("gaps", []), anchor.get("gaps", []))
        previous["status"] = _status_choice(
            str(previous.get("status", "candidate")),
            str(anchor.get("status", "candidate")),
            ("active", "candidate", "needs_evidence", "rejected"),
        )
        for field, identifier in (
            ("baselines", "baseline_id"),
            ("entries", "entry_id"),
            ("edges", "edge_id"),
        ):
            object_index = {
                str(item[identifier]): item for item in previous.get(field, [])
            }
            for addition in anchor.get(field, []):
                existing = object_index.get(str(addition[identifier]))
                if existing is None:
                    object_index[str(addition[identifier])] = addition
                    continue
                for list_field in (
                    "evidence_refs", "preconditions", "applicable_baseline_ids",
                ):
                    if list_field in existing or list_field in addition:
                        existing[list_field] = _merge_strings(
                            existing.get(list_field, []), addition.get(list_field, [])
                        )
                if "status" in existing and "status" in addition:
                    existing["status"] = _status_choice(
                        str(existing["status"]),
                        str(addition["status"]),
                        ("accepted", "candidate", "needs_evidence", "rejected"),
                    )
                existing["updated_at"] = utc_now()
            previous[field] = sorted(object_index.values(), key=lambda item: item[identifier])
        previous["updated_at"] = utc_now()

    def apply_curation(self, data: Mapping[str, Any]) -> dict[str, Any]:
        claims = read_jsonl(self.work_dir / "evidence" / "claims.jsonl")
        valid_claim_ids = {str(claim["claim_id"]) for claim in claims}
        catalog = read_json(self.work_dir / "catalog" / "entities.json", _initial_catalog())
        anchors_doc = read_json(self.work_dir / "anchors" / "anchors.json", _initial_anchors())
        entity_index = {
            str(entity["entity_id"]): entity for entity in catalog.get("entities", [])
        }
        anchor_index = {
            str(anchor["anchor_id"]): anchor for anchor in anchors_doc.get("anchors", [])
        }
        accepted_anchors = 0
        rejected_anchors = 0
        rejected_objects = 0

        for proposal in data.get("anchors", []):
            anchor_errors: list[str] = []
            admission_gaps: list[str] = []
            anchor_refs = _string_list(proposal.get("evidence_refs"))
            anchor_errors.extend(
                validate_references(anchor_refs, valid_claim_ids, "anchor evidence")
            )
            if not anchor_refs:
                anchor_errors.append("anchor has no evidence refs")
            baseline_proposals = list(proposal.get("baselines", []))
            baseline_keys = [str(item["baseline_key"]) for item in baseline_proposals]
            if len(set(baseline_keys)) != len(baseline_keys):
                anchor_errors.append("duplicate baseline_key inside Anchor proposal")
            primary_key = str(proposal.get("primary_baseline_key", ""))
            baseline_by_key = {
                str(item["baseline_key"]): item for item in baseline_proposals
            }
            primary = baseline_by_key.get(primary_key)
            if primary is None:
                anchor_errors.append(
                    "primary_baseline_key does not resolve inside proposed baselines"
                )
            else:
                primary_refs = _string_list(primary.get("evidence_refs"))
                anchor_errors.extend(
                    validate_references(
                        primary_refs,
                        valid_claim_ids,
                        f"primary baseline {primary_key} evidence",
                    )
                )
                if not primary_refs:
                    anchor_errors.append(
                        f"primary baseline {primary_key} has no evidence refs"
                    )
            if anchor_errors:
                rejected_anchors += 1
                self._record_mutation_rejection(
                    phase="curation_anchor",
                    proposal=proposal,
                    errors=anchor_errors,
                )
                continue

            assert primary is not None
            anchor_id = anchor_id_for(
                workload=str(proposal["workload"]),
                phase=str(proposal["phase"]),
                regime=str(proposal["regime"]),
                backend=str(proposal["backend"]),
                bottleneck=str(proposal["bottleneck"]),
                target_metrics=_string_list(proposal.get("target_metrics")),
                primary_baseline_name=str(primary["name"]),
            )
            baseline_key_to_id: dict[str, str] = {}
            baselines: list[dict[str, Any]] = []
            for baseline in baseline_proposals:
                baseline_key = str(baseline["baseline_key"])
                refs = _string_list(baseline.get("evidence_refs"))
                baseline_errors = validate_references(
                    refs,
                    valid_claim_ids,
                    f"baseline {baseline_key} evidence",
                )
                if not refs:
                    baseline_errors.append(
                        f"baseline {baseline_key} has no evidence refs"
                    )
                if baseline_errors:
                    rejected_objects += 1
                    admission_gaps.append(
                        f"baseline proposal {baseline_key} was quarantined"
                    )
                    self._record_mutation_rejection(
                        phase="curation_baseline",
                        proposal=baseline,
                        errors=baseline_errors,
                    )
                    continue
                baseline_id = baseline_id_for(
                    anchor_id=anchor_id,
                    role=str(baseline["role"]),
                    name=str(baseline["name"]),
                )
                baseline_key_to_id[baseline_key] = baseline_id
                baselines.append(
                    {
                        "baseline_id": baseline_id,
                        "anchor_id": anchor_id,
                        "name": normalize_text(str(baseline["name"])),
                        "role": baseline["role"],
                        "description": normalize_text(str(baseline["description"])),
                        "evidence_refs": refs,
                        "created_at": utc_now(),
                        "updated_at": utc_now(),
                    }
                )
            if primary_key not in baseline_key_to_id:
                rejected_anchors += 1
                self._record_mutation_rejection(
                    phase="curation_anchor",
                    proposal=proposal,
                    errors=["validated primary baseline was not admitted"],
                )
                continue

            entry_proposals = list(proposal.get("entries", []))
            entry_key_counts: dict[str, int] = {}
            for entry in entry_proposals:
                entry_key = str(entry["entry_key"])
                entry_key_counts[entry_key] = entry_key_counts.get(entry_key, 0) + 1
            entries: list[dict[str, Any]] = []
            entry_key_to_id: dict[str, str] = {}
            staged_entities: list[dict[str, Any]] = []
            for entry in entry_proposals:
                entry_key = str(entry["entry_key"])
                entry_errors: list[str] = []
                if entry_key_counts[entry_key] > 1:
                    entry_errors.append(
                        f"duplicate entry_key {entry_key} inside Anchor proposal"
                    )
                refs = _string_list(entry.get("evidence_refs"))
                entry_errors.extend(
                    validate_references(
                        refs,
                        valid_claim_ids,
                        f"entry {entry_key} evidence",
                    )
                )
                if not refs:
                    entry_errors.append(f"entry {entry_key} has no evidence refs")
                applicable: list[str] = []
                for baseline_key in entry.get("applicable_baseline_keys", []):
                    baseline_id = baseline_key_to_id.get(str(baseline_key))
                    if not baseline_id:
                        entry_errors.append(
                            f"entry {entry_key} references unknown/quarantined "
                            f"baseline_key {baseline_key}"
                        )
                    else:
                        applicable.append(baseline_id)
                if entry_errors:
                    rejected_objects += 1
                    admission_gaps.append(
                        f"entry proposal {entry_key} was quarantined"
                    )
                    self._record_mutation_rejection(
                        phase="curation_entry",
                        proposal=entry,
                        errors=entry_errors,
                    )
                    continue
                entity_id = entity_id_for(
                    name=str(entry["entity_name"]),
                    entity_type=str(entry["entity_type"]),
                )
                staged_entities.append(
                    {
                        "entity_id": entity_id,
                        "name": normalize_text(str(entry["entity_name"])),
                        "entity_type": entry["entity_type"],
                        "aliases": [],
                        "evidence_refs": refs,
                        "created_at": utc_now(),
                        "updated_at": utc_now(),
                    }
                )
                entry_id = entry_id_for(
                    anchor_id=anchor_id,
                    layer=str(entry["layer"]),
                    role=str(entry["role"]),
                    entity_id=entity_id,
                    claim=str(entry["claim"]),
                )
                entry_key_to_id[entry_key] = entry_id
                entries.append(
                    {
                        "entry_id": entry_id,
                        "entity_id": entity_id,
                        "anchor_id": anchor_id,
                        "layer": entry["layer"],
                        "role": entry["role"],
                        "claim": normalize_text(str(entry["claim"])),
                        "modifiable_object": normalize_text(str(entry["modifiable_object"])),
                        "applicable_baseline_ids": sorted(set(applicable)),
                        "preconditions": _string_list(entry["preconditions"]),
                        "expected_effect": normalize_text(str(entry["expected_effect"])),
                        "evidence_refs": refs,
                        "confidence": entry["confidence"],
                        "status": entry["status"],
                        "created_at": utc_now(),
                        "updated_at": utc_now(),
                    }
                )

            edges: list[dict[str, Any]] = []
            for edge in proposal.get("edges", []):
                edge_errors: list[str] = []
                from_id = entry_key_to_id.get(str(edge["from_entry_key"]))
                to_id = entry_key_to_id.get(str(edge["to_entry_key"]))
                if not from_id or not to_id:
                    edge_errors.append(
                        "edge endpoints must resolve to entries re-proposed in the same Anchor"
                    )
                refs = _string_list(edge.get("evidence_refs"))
                edge_errors.extend(
                    validate_references(
                        refs,
                        valid_claim_ids,
                        f"edge {edge.get('from_entry_key')}->{edge.get('to_entry_key')} evidence",
                    )
                )
                if not refs:
                    edge_errors.append(
                        f"edge {edge.get('from_entry_key')}->{edge.get('to_entry_key')} "
                        "has no evidence refs"
                    )
                if edge_errors:
                    rejected_objects += 1
                    admission_gaps.append(
                        "edge proposal "
                        f"{edge.get('from_entry_key')}->{edge.get('to_entry_key')} "
                        "was quarantined"
                    )
                    self._record_mutation_rejection(
                        phase="curation_edge",
                        proposal=edge,
                        errors=edge_errors,
                    )
                    continue
                assert from_id is not None and to_id is not None
                edge_id = edge_id_for(
                    anchor_id=anchor_id,
                    from_entry_id=from_id,
                    to_entry_id=to_id,
                    relation=str(edge["relation"]),
                    interface=str(edge["interface"]),
                )
                edges.append(
                    {
                        "edge_id": edge_id,
                        "anchor_id": anchor_id,
                        "from_entry_id": from_id,
                        "to_entry_id": to_id,
                        "relation": edge["relation"],
                        "interface": normalize_text(str(edge["interface"])),
                        "compatibility": edge["compatibility"],
                        "condition": normalize_text(str(edge["condition"])),
                        "evidence_refs": refs,
                        "confidence": edge["confidence"],
                        "created_at": utc_now(),
                        "updated_at": utc_now(),
                    }
                )

            for entity in staged_entities:
                self._merge_entity(entity_index, entity)
            anchor = {
                "anchor_id": anchor_id,
                "workload": normalize_text(str(proposal["workload"])),
                "phase": normalize_text(str(proposal["phase"])),
                "regime": normalize_text(str(proposal["regime"])),
                "backend": normalize_text(str(proposal["backend"])),
                "bottleneck": normalize_text(str(proposal["bottleneck"])),
                "primary_baseline_id": baseline_key_to_id[primary_key],
                "target_metrics": _string_list(proposal["target_metrics"]),
                "evidence_refs": anchor_refs,
                "status": proposal["status"],
                "baselines": sorted(baselines, key=lambda item: item["baseline_id"]),
                "entries": sorted(entries, key=lambda item: item["entry_id"]),
                "edges": sorted(edges, key=lambda item: item["edge_id"]),
                "gaps": _merge_strings(
                    _string_list(proposal.get("gaps")),
                    admission_gaps,
                ),
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            self._merge_anchor(anchor_index, anchor)
            accepted_anchors += 1

        dispositions_doc = read_json(
            self.work_dir / "curation" / "dispositions.json",
            {"dispositions": []},
        )
        disposition_index = {
            str(item["claim_id"]): item
            for item in dispositions_doc.get("dispositions", [])
        }
        for disposition in data.get("dispositions", []):
            claim_id = str(disposition["claim_id"])
            if claim_id not in valid_claim_ids:
                self._record_mutation_rejection(
                    phase="curation_disposition",
                    proposal=disposition,
                    errors=[f"unknown claim_id {claim_id}"],
                )
                continue
            disposition_index[claim_id] = {
                **dict(disposition),
                "updated_at": utc_now(),
            }
        atomic_write_json(
            self.work_dir / "catalog" / "entities.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "entities": sorted(entity_index.values(), key=lambda item: item["entity_id"]),
            },
        )
        atomic_write_json(
            self.work_dir / "anchors" / "anchors.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "anchors": sorted(anchor_index.values(), key=lambda item: item["anchor_id"]),
            },
        )
        atomic_write_json(
            self.work_dir / "curation" / "dispositions.json",
            {
                "dispositions": sorted(
                    disposition_index.values(), key=lambda item: item["claim_id"]
                )
            },
        )
        result = {
            "accepted_anchors": accepted_anchors,
            "rejected_anchors": rejected_anchors,
            "rejected_objects": rejected_objects,
            "entity_count": len(entity_index),
            "anchor_count": len(anchor_index),
        }
        self.event("curation_mutations_applied", **result)
        return result

    def run_curation(self) -> None:
        self.set_phase("curation")
        claims = read_jsonl(self.work_dir / "evidence" / "claims.jsonl")
        if not claims:
            gap = "curation skipped: discovery produced no validated claims"
            self.add_gap(gap)
            self.event("curation_completed", claim_count=0, gap=gap)
            self.mark_phase_completed("curation")
            return
        batch_size = max(1, int(self.config["curation_batch_size"]))
        batches = [
            claims[index:index + batch_size]
            for index in range(0, len(claims), batch_size)
        ]
        request_budget = int(self.config["max_curator_evidence_requests"])
        requests_used = 0
        seen_evidence_queries: set[str] = set()

        for batch_index, batch in enumerate(batches, 1):
            batch_key = f"batch-{batch_index:04d}"
            session_id = self.resume_session_id(
                role="curator",
                record=(
                    self.state.get("sessions", {})
                    .get("curators", {})
                    .get(batch_key, {})
                ),
                context=f"curator:{batch_key}",
            )
            new_claims: list[dict[str, Any]] = []
            action_context = "initial batch integration"
            seen_actions: set[str] = set()
            batch_terminal = False
            for round_number in range(
                1, int(self.config["curator_rounds_per_batch"]) + 1
            ):
                response = self.call(
                    role="curator",
                    prompt=self._curator_prompt(
                        batch_index=batch_index,
                        batch_count=len(batches),
                        claims=batch,
                        new_claims=new_claims,
                        action_context=action_context,
                    ),
                    schema_name="curation",
                    conversation_key=f"curator:b{batch_index}:r{round_number}",
                    session_id=session_id,
                )
                session_id = response.session_id
                with self.state_lock:
                    self.state["sessions"].setdefault("curators", {})[batch_key] = (
                        self.session_record(response)
                    )
                    self.checkpoint()
                for gap in response.data.get("unresolved_gaps", []):
                    self.add_gap(
                        f"curation batch {batch_index}: {gap}"
                    )
                mutation_result = self.apply_curation(response.data)
                signature = sha256_text(canonical_json(response.data))
                if signature in seen_actions:
                    self.event(
                        "curator_loop_stopped",
                        batch_index=batch_index,
                        reason="repeated identical action",
                    )
                    break
                seen_actions.add(signature)
                action = response.data["action"]
                if action in ("integrate", "complete"):
                    if action == "complete" or mutation_result["accepted_anchors"] > 0:
                        batch_terminal = True
                        break
                    action_context = "previous integration contained no valid mutation; repair semantics"
                    continue
                if action == "request_evidence":
                    requests = list(response.data.get("evidence_requests", []))
                    if not requests or requests_used >= request_budget:
                        action_context = "evidence request budget exhausted; complete with explicit gaps"
                        continue
                    accepted_ids: list[str] = []
                    for request in requests:
                        if requests_used >= request_budget:
                            break
                        query_key = normalize_key(str(request.get("query", "")))
                        if not query_key or query_key in seen_evidence_queries:
                            self.event(
                                "evidence_request_rejected",
                                scope=f"curation-batch-{batch_index}",
                                query=request.get("query", ""),
                                reason="empty or repeated normalized query",
                            )
                            continue
                        seen_evidence_queries.add(query_key)
                        requests_used += 1
                        accepted_ids.extend(
                            self.targeted_discovery(
                                request,
                                scope=f"curation-batch-{batch_index}",
                            )
                        )
                    current_claims = {
                        str(item["claim_id"]): item
                        for item in read_jsonl(
                            self.work_dir / "evidence" / "claims.jsonl"
                        )
                    }
                    new_claims = [
                        current_claims[claim_id]
                        for claim_id in sorted(set(accepted_ids))
                        if claim_id in current_claims
                    ]
                    action_context = (
                        "targeted evidence returned; integrate it or preserve the exact gap"
                    )
            if not batch_terminal:
                self.add_gap(
                    f"curation batch {batch_index}/{len(batches)} exhausted or "
                    "stalled before an accepted integrate/complete action"
                )

        all_claims = read_jsonl(self.work_dir / "evidence" / "claims.jsonl")
        dispositions_path = self.work_dir / "curation" / "dispositions.json"
        dispositions_doc = read_json(dispositions_path, {"dispositions": []})
        disposition_index = {
            str(item["claim_id"]): item
            for item in dispositions_doc.get("dispositions", [])
        }
        accepted_refs: set[str] = set()
        canonical_anchors = read_json(
            self.work_dir / "anchors" / "anchors.json",
            _initial_anchors(),
        )
        for anchor in canonical_anchors.get("anchors", []):
            accepted_refs.update(str(item) for item in anchor.get("evidence_refs", []))
            for field in ("baselines", "entries", "edges"):
                for item in anchor.get(field, []):
                    accepted_refs.update(
                        str(ref) for ref in item.get("evidence_refs", [])
                    )
        for claim in all_claims:
            claim_id = str(claim["claim_id"])
            if claim_id in accepted_refs:
                previous = disposition_index.get(claim_id, {})
                disposition_index[claim_id] = {
                    "claim_id": claim_id,
                    "status": "integrated",
                    "reason": (
                        previous.get("reason")
                        if previous.get("status") == "integrated"
                        else "deterministic canonical-object evidence reference"
                    ),
                    "updated_at": utc_now(),
                }
            elif (
                claim_id in disposition_index
                and disposition_index[claim_id].get("status") == "integrated"
            ):
                disposition_index[claim_id] = {
                    "claim_id": claim_id,
                    "status": "needs_evidence",
                    "reason": (
                        "curator marked integrated, but no admitted canonical object "
                        "references this claim"
                    ),
                    "updated_at": utc_now(),
                }
            elif claim_id not in disposition_index:
                disposition_index[claim_id] = {
                    "claim_id": claim_id,
                    "status": "needs_evidence",
                    "reason": "curator returned no final disposition within the configured budget",
                    "updated_at": utc_now(),
                }
        atomic_write_json(
            dispositions_path,
            {
                "dispositions": sorted(
                    disposition_index.values(), key=lambda item: item["claim_id"]
                )
            },
        )
        reconcile_dispositions(self.work_dir)
        anchors = read_json(self.work_dir / "anchors" / "anchors.json", _initial_anchors())
        self.event(
            "curation_completed",
            claim_count=len(all_claims),
            anchor_count=len(anchors.get("anchors", [])),
            evidence_requests=requests_used,
        )
        self.mark_phase_completed("curation")

    def _anchor_claims(
        self,
        anchor: Mapping[str, Any],
        extra_claim_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        refs = set(str(item) for item in anchor.get("evidence_refs", []))
        refs.update(str(item) for item in extra_claim_ids)
        for field in ("baselines", "entries", "edges"):
            for item in anchor.get(field, []):
                refs.update(str(ref) for ref in item.get("evidence_refs", []))
        all_claims = read_jsonl(self.work_dir / "evidence" / "claims.jsonl")
        return [claim for claim in all_claims if str(claim["claim_id"]) in refs]

    def _direction_prompt(
        self,
        *,
        anchor: Mapping[str, Any],
        targeted_claim_ids: Sequence[str],
        action_context: str,
    ) -> str:
        return (
            "# Domain rules\n"
            + load_reference("domain-model.md")
            + "\n\n# Direction construction rules\n"
            + load_reference("curator-loop.md")
            + "\n\n# Value rubric\n"
            + load_reference("layer-value-rubric.md")
            + "\n\n# Exact Anchor map\n"
            + _json_excerpt(anchor, 100_000)
            + "\n\n# Evidence claims\n"
            + _json_excerpt(
                self._anchor_claims(anchor, targeted_claim_ids),
                75_000,
            )
            + "\n\n# Loop context\n"
            + action_context
            + "\n\nReturn one or more separate compatible subgraphs. Use canonical "
            "entry_id, edge_id, and baseline_id values exactly. Do not force L1-L6 "
            "coverage. Keep baseline/implementation/method references as separate "
            "Direction kinds when they are useful but not experiments."
        )

    def apply_directions(
        self,
        *,
        anchor: Mapping[str, Any],
        data: Mapping[str, Any],
    ) -> dict[str, int]:
        directions_doc = read_json(
            self.work_dir / "directions" / "directions.json",
            _initial_directions(),
        )
        direction_index = {
            str(item["direction_id"]): item
            for item in directions_doc.get("directions", [])
        }
        valid_claim_ids = {
            str(item["claim_id"])
            for item in read_jsonl(self.work_dir / "evidence" / "claims.jsonl")
        }
        baseline_index = {
            str(item["baseline_id"]): item for item in anchor.get("baselines", [])
        }
        entry_index = {
            str(item["entry_id"]): item for item in anchor.get("entries", [])
        }
        edge_index = {
            str(item["edge_id"]): item for item in anchor.get("edges", [])
        }
        accepted = 0
        rejected = 0
        for proposal in data.get("directions", []):
            errors: list[str] = []
            entry_ids = sorted(set(str(item) for item in proposal["selected_entry_ids"]))
            edge_ids = sorted(set(str(item) for item in proposal["selected_edge_ids"]))
            baseline_ids = sorted(set(str(item) for item in proposal["baseline_ids"]))
            evidence_refs = _string_list(proposal["evidence_refs"])
            errors.extend(validate_references(entry_ids, set(entry_index), "direction entry"))
            errors.extend(validate_references(edge_ids, set(edge_index), "direction edge"))
            errors.extend(
                validate_references(baseline_ids, set(baseline_index), "direction baseline")
            )
            errors.extend(
                validate_references(evidence_refs, valid_claim_ids, "direction evidence")
            )
            if not evidence_refs:
                errors.append("direction has no evidence refs")
            for entry_id in entry_ids:
                entry = entry_index.get(entry_id)
                if entry and entry.get("status") == "rejected":
                    errors.append(f"direction selects rejected entry {entry_id}")
            kind = str(proposal["kind"])
            if kind == "experiment" and not entry_ids:
                errors.append("experiment Direction has no selected entries")
            if kind == "experiment" and not baseline_ids:
                errors.append("experiment Direction has no baseline")
            if not entry_ids and not baseline_ids:
                errors.append("Direction has neither entries nor baselines")
            for edge_id in edge_ids:
                edge = edge_index.get(edge_id)
                if not edge:
                    continue
                endpoints = {str(edge["from_entry_id"]), str(edge["to_entry_id"])}
                if not endpoints.issubset(set(entry_ids)):
                    errors.append(
                        f"edge {edge_id} endpoints are not both selected"
                    )
                if (
                    str(edge.get("compatibility")) in ("incompatible", "unknown")
                    or str(edge.get("relation")) in ("conflicts_with", "substitutes")
                ):
                    errors.append(
                        f"edge {edge_id} is incompatible/unknown/substitutive "
                        "and cannot form synergy"
                    )
                if (
                    edge.get("compatibility") == "conditional"
                    and not normalize_text(str(edge.get("condition", "")))
                ):
                    errors.append(
                        f"conditional edge {edge_id} has no explicit condition"
                    )
            selected_edge_objects = [
                edge_index[edge_id] for edge_id in edge_ids if edge_id in edge_index
            ]
            if not _selected_entries_are_connected(
                entry_ids,
                selected_edge_objects,
            ):
                errors.append(
                    "selected entries do not form one connected entry-level subgraph"
                )
            if errors:
                rejected += 1
                self._record_mutation_rejection(
                    phase="direction_build", proposal=proposal, errors=errors
                )
                continue
            direction_id = direction_id_for(
                anchor_id=str(anchor["anchor_id"]),
                title=str(proposal["title"]),
                selected_entry_ids=entry_ids,
                hypothesis=str(proposal["hypothesis"]),
            )
            direction = {
                "direction_id": direction_id,
                "anchor_id": anchor["anchor_id"],
                "title": normalize_text(str(proposal["title"])),
                "selected_entry_ids": entry_ids,
                "selected_edge_ids": edge_ids,
                "baseline_ids": baseline_ids,
                "hypothesis": normalize_text(str(proposal["hypothesis"])),
                "expected_effects": _string_list(proposal["expected_effects"]),
                "preconditions": _string_list(proposal["preconditions"]),
                "ablation_plan": _string_list(proposal["ablation_plan"]),
                "evidence_refs": evidence_refs,
                "gaps": _string_list(proposal["gaps"]),
                "kind": kind,
                "status": "needs_evidence" if proposal["gaps"] else "candidate",
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            previous = direction_index.get(direction_id)
            if previous is None:
                direction_index[direction_id] = direction
            else:
                for field in (
                    "selected_entry_ids", "selected_edge_ids", "baseline_ids",
                    "expected_effects", "preconditions", "ablation_plan",
                    "evidence_refs", "gaps",
                ):
                    previous[field] = _merge_strings(
                        previous.get(field, []), direction.get(field, [])
                    )
                previous["status"] = _status_choice(
                    str(previous.get("status", "candidate")),
                    str(direction["status"]),
                    ("reviewed", "candidate", "needs_evidence", "rejected"),
                )
                previous["updated_at"] = utc_now()
            accepted += 1
        atomic_write_json(
            self.work_dir / "directions" / "directions.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "directions": sorted(
                    direction_index.values(), key=lambda item: item["direction_id"]
                ),
            },
        )
        self.event(
            "direction_mutations_applied",
            anchor_id=anchor["anchor_id"],
            accepted=accepted,
            rejected=rejected,
        )
        return {"accepted": accepted, "rejected": rejected}

    def run_direction_build(self) -> None:
        self.set_phase("direction_build")
        anchors_doc = read_json(
            self.work_dir / "anchors" / "anchors.json", _initial_anchors()
        )
        anchors = [
            anchor
            for anchor in anchors_doc.get("anchors", [])
            if anchor.get("status") != "rejected"
            and (anchor.get("entries") or anchor.get("baselines"))
        ]
        request_budget = int(self.config["max_direction_evidence_requests"])
        requests_used = 0
        for anchor in anchors:
            anchor_id = str(anchor["anchor_id"])
            session_id = self.resume_session_id(
                role="direction",
                record=(
                    self.state.get("sessions", {})
                    .get("directions", {})
                    .get(anchor_id, {})
                ),
                context=f"direction:{anchor_id}",
            )
            targeted_claim_ids: list[str] = []
            seen_evidence_queries: set[str] = set()
            action_context = "construct initial compatible subgraphs"
            seen_signatures: set[str] = set()
            anchor_completed = False
            for round_number in range(
                1, int(self.config["direction_rounds_per_anchor"]) + 1
            ):
                response = self.call(
                    role="direction",
                    prompt=self._direction_prompt(
                        anchor=anchor,
                        targeted_claim_ids=targeted_claim_ids,
                        action_context=action_context,
                    ),
                    schema_name="direction",
                    conversation_key=f"direction:{anchor_id}:r{round_number}",
                    session_id=session_id,
                )
                session_id = response.session_id
                with self.state_lock:
                    self.state["sessions"]["directions"][anchor_id] = (
                        self.session_record(response)
                    )
                    self.checkpoint()
                for gap in response.data.get("unresolved_gaps", []):
                    self.add_gap(f"Direction build {anchor_id}: {gap}")
                mutation = self.apply_directions(anchor=anchor, data=response.data)
                signature = sha256_text(canonical_json(response.data))
                if signature in seen_signatures:
                    self.event(
                        "direction_loop_stopped",
                        anchor_id=anchor_id,
                        reason="repeated identical action",
                    )
                    break
                seen_signatures.add(signature)
                action = response.data["action"]
                if action == "complete":
                    anchor_completed = True
                    break
                if action == "integrate":
                    if mutation["accepted"]:
                        action_context = (
                            "previous subgraphs were integrated; complete or add only "
                            "materially distinct alternatives"
                        )
                    else:
                        action_context = (
                            "previous proposals failed deterministic validation; "
                            "return corrected proposals or explicit gaps"
                        )
                    continue
                requests = response.data.get("evidence_requests", [])
                if not requests or requests_used >= request_budget:
                    action_context = "evidence budget exhausted; complete with exact gaps"
                    continue
                for request in requests:
                    if requests_used >= request_budget:
                        break
                    query_key = normalize_key(str(request.get("query", "")))
                    if not query_key or query_key in seen_evidence_queries:
                        self.event(
                            "evidence_request_rejected",
                            scope=f"direction-{anchor_id}",
                            query=request.get("query", ""),
                            reason="empty or repeated normalized query",
                        )
                        continue
                    seen_evidence_queries.add(query_key)
                    requests_used += 1
                    targeted_claim_ids.extend(
                        self.targeted_discovery(
                            request,
                            scope=f"direction-{anchor_id}",
                        )
                    )
                targeted_claim_ids = sorted(set(targeted_claim_ids))
                action_context = "targeted claims added; rebuild or preserve the gap"
            if not anchor_completed:
                self.add_gap(
                    f"Direction build for Anchor {anchor_id} exhausted or stalled "
                    "before a complete action"
                )
        directions_doc = read_json(
            self.work_dir / "directions" / "directions.json",
            _initial_directions(),
        )
        reconcile_dispositions(self.work_dir)
        self.event(
            "direction_build_completed",
            anchor_count=len(anchors),
            direction_count=len(directions_doc.get("directions", [])),
            evidence_requests=requests_used,
        )
        self.mark_phase_completed("direction_build")

    def _experiment_bundle(
        self,
        direction: Mapping[str, Any],
        *,
        extra_claim_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        anchors_doc = read_json(
            self.work_dir / "anchors" / "anchors.json", _initial_anchors()
        )
        anchor = next(
            item
            for item in anchors_doc.get("anchors", [])
            if item["anchor_id"] == direction["anchor_id"]
        )
        selected_entries = [
            item
            for item in anchor.get("entries", [])
            if item["entry_id"] in set(direction.get("selected_entry_ids", []))
        ]
        selected_edges = [
            item
            for item in anchor.get("edges", [])
            if item["edge_id"] in set(direction.get("selected_edge_ids", []))
        ]
        unselected_entries = [
            item
            for item in anchor.get("entries", [])
            if item["entry_id"] not in set(direction.get("selected_entry_ids", []))
        ]
        unselected_edges = [
            item
            for item in anchor.get("edges", [])
            if item["edge_id"] not in set(direction.get("selected_edge_ids", []))
        ]
        baselines = [
            item
            for item in anchor.get("baselines", [])
            if item["baseline_id"] in set(direction.get("baseline_ids", []))
        ]
        claim_refs = set(str(item) for item in direction.get("evidence_refs", []))
        claim_refs.update(str(item) for item in extra_claim_ids)
        for item in [
            *selected_entries,
            *selected_edges,
            *unselected_entries,
            *unselected_edges,
            *baselines,
        ]:
            claim_refs.update(str(ref) for ref in item.get("evidence_refs", []))
        claims = [
            item
            for item in read_jsonl(self.work_dir / "evidence" / "claims.jsonl")
            if str(item["claim_id"]) in claim_refs
        ]
        return {
            "topic": self.config["topic"],
            "anchor": {
                key: anchor[key]
                for key in (
                    "anchor_id", "workload", "phase", "regime", "backend",
                    "bottleneck", "primary_baseline_id", "target_metrics", "gaps",
                )
            },
            "direction": dict(direction),
            "baselines": baselines,
            "selected_entries": selected_entries,
            "selected_edges": selected_edges,
            "unselected_entries": unselected_entries,
            "unselected_edges": unselected_edges,
            "evidence_claims": claims,
        }

    def _judge_prompt(
        self,
        *,
        bundle: Mapping[str, Any],
        qa: Sequence[Mapping[str, Any]],
        round_number: int,
        loop_context: str,
    ) -> str:
        covered_dimensions = {
            str(item.get("dimension"))
            for item in qa
            if item.get("action") == "ask" and isinstance(item.get("answer"), dict)
        }
        pending_dimensions = [
            dimension
            for dimension in REVIEW_DIMENSIONS
            if dimension not in covered_dimensions
        ]
        return (
            "# Expert review protocol\n"
            + load_reference("direction-review-loop.md")
            + "\n\n# Value and technical rubric\n"
            + load_reference("layer-value-rubric.md")
            + "\n\n# Normalized Experiment Bundle\n"
            + _json_excerpt(bundle, 120_000)
            + "\n\n# Accumulated question/answer ledger\n"
            + _json_excerpt(list(qa), 65_000)
            + f"\n\n# Round\n{round_number}/{self.config['review_rounds_per_direction']}\n"
            f"covered dimensions: {', '.join(sorted(covered_dimensions)) or 'none'}\n"
            f"pending mandatory dimensions: {', '.join(pending_dimensions) or 'none'}\n"
            f"loop context: {loop_context}\n\n"
            "You cannot retrieve. Choose exactly one action. For ask, fill dimension "
            "and question, use empty strings in evidence_request and set every review "
            "field to null. For request_evidence, fill all evidence_request fields and "
            "set every review field to null. For complete, "
            "use dimension=complete and populate every ExpertReview field; use IDs "
            "from the bundle. The outer gate rejects complete while any mandatory "
            "dimension remains pending."
        )

    def _evidence_answer_prompt(
        self,
        *,
        bundle: Mapping[str, Any],
        question: str,
        dimension: str,
        round_number: int,
    ) -> str:
        return (
            "# Evidence role protocol\n"
            + load_reference("direction-review-loop.md")
            + "\n\n# Experiment Bundle and allowlisted evidence\n"
            + _json_excerpt(bundle, 130_000)
            + f"\n\n# Current question\nround={round_number}\n"
            f"dimension={dimension}\nquestion={question}\n\n"
            "Answer only from evidence_claims and canonical objects above. Cite claim "
            "IDs, separate direct facts from inference, and preserve unknowns."
        )

    def _validate_complete_review(
        self,
        *,
        review: Mapping[str, Any],
        bundle: Mapping[str, Any],
    ) -> list[str]:
        errors = validate_schema(review, REVIEW_SCHEMA)
        valid_claims = {
            str(item["claim_id"]) for item in bundle.get("evidence_claims", [])
        }
        valid_entries = {
            str(item["entry_id"]) for item in bundle.get("selected_entries", [])
        }
        valid_edges = {
            str(item["edge_id"]) for item in bundle.get("selected_edges", [])
        }
        valid_baselines = {
            str(item["baseline_id"]) for item in bundle.get("baselines", [])
        }
        valid_alternative_entries = {
            str(item["entry_id"]) for item in bundle.get("unselected_entries", [])
        }
        valid_alternative_edges = {
            str(item["edge_id"]) for item in bundle.get("unselected_edges", [])
        }
        if not errors:
            errors.extend(
                validate_references(
                    review.get("evidence_refs", []), valid_claims, "review evidence"
                )
            )
            errors.extend(
                validate_references(
                    review.get("entry_refs", []), valid_entries, "review entry"
                )
            )
            errors.extend(
                validate_references(
                    review.get("edge_refs", []), valid_edges, "review edge"
                )
            )
            errors.extend(
                validate_references(
                    review.get("baseline_refs", []), valid_baselines, "review baseline"
                )
            )
            errors.extend(
                validate_references(
                    review.get("alternative_entry_refs", []),
                    valid_alternative_entries,
                    "review alternative entry",
                )
            )
            errors.extend(
                validate_references(
                    review.get("alternative_edge_refs", []),
                    valid_alternative_edges,
                    "review alternative edge",
                )
            )
            if (
                review.get("decision") == "experiment_candidate"
                and not review.get("baseline_refs")
            ):
                errors.append("experiment_candidate review has no baseline_refs")
            errors.extend(
                _review_semantic_errors(
                    review,
                    bundle.get("direction", {}),
                )
            )
        return errors

    def _write_review_artifact(
        self,
        *,
        direction: Mapping[str, Any],
        status: str,
        qa: Sequence[Mapping[str, Any]],
        rounds: int,
        review: Mapping[str, Any] | None = None,
        pending_reason: str = "",
        provider: str = "",
        model: str = "",
    ) -> None:
        direction_id = str(direction["direction_id"])
        artifact = {
            "protocol_version": PROTOCOL_VERSION,
            "review_id": stable_id("R", direction_id, status, review or {}, pending_reason),
            "direction_id": direction_id,
            "status": status,
            "rounds": rounds,
            "provider": provider,
            "model": model,
            "qa": list(qa),
            "review": dict(review or {}),
            "pending_reason": pending_reason,
            "created_at": utc_now(),
        }
        atomic_write_json(self.work_dir / "reviews" / f"{direction_id}.json", artifact)
        atomic_write_text(
            self.work_dir / "reviews" / f"{direction_id}.md",
            render_review_markdown(direction, artifact),
        )
        self.event(
            "direction_review_written",
            direction_id=direction_id,
            review_status=status,
            decision=(review or {}).get("decision", ""),
            rounds=rounds,
        )

    def _update_direction_review_status(self, direction_id: str, decision: str) -> None:
        path = self.work_dir / "directions" / "directions.json"
        document = read_json(path, _initial_directions())
        for direction in document.get("directions", []):
            if direction["direction_id"] != direction_id:
                continue
            direction["status"] = {
                "experiment_candidate": "reviewed",
                "needs_evidence": "needs_evidence",
                "baseline_reference": "reviewed",
                "rejected": "rejected",
            }[decision]
            direction["updated_at"] = utc_now()
        atomic_write_json(path, document)

    def review_direction(self, direction: Mapping[str, Any]) -> None:
        direction_id = str(direction["direction_id"])
        existing_artifact = read_json(
            self.work_dir / "reviews" / f"{direction_id}.json",
            {},
        )
        qa: list[dict[str, Any]] = (
            list(existing_artifact.get("qa", []))
            if isinstance(existing_artifact, dict)
            else []
        )
        extra_claim_ids = sorted(
            {
                str(claim_id)
                for item in qa
                for claim_id in item.get("accepted_claim_ids", [])
            }
        )
        resume_review_sessions = (
            self.config.get("review_session_mode", "stateless") == "resume"
        )
        judge_session = (
            self.resume_session_id(
                role="judge",
                record=(
                    self.state.get("sessions", {})
                    .get("judge", {})
                    .get(direction_id, {})
                ),
                context=f"judge:{direction_id}",
            )
            if resume_review_sessions
            else ""
        )
        evidence_session = (
            self.resume_session_id(
                role="evidence",
                record=(
                    self.state.get("sessions", {})
                    .get("evidence", {})
                    .get(direction_id, {})
                ),
                context=f"evidence:{direction_id}",
            )
            if resume_review_sessions
            else ""
        )
        requests_used = sum(
            1 for item in qa if item.get("action") == "request_evidence"
        )
        seen_evidence_queries = {
            normalize_key(str(item.get("request", {}).get("query", "")))
            for item in qa
            if item.get("action") == "request_evidence"
        }
        seen_evidence_queries.discard("")
        max_requests = int(self.config["max_review_evidence_requests"])
        last_provider = ""
        last_model = ""
        loop_context = "start with the highest-priority unresolved dimension"
        completed_rounds = max(
            (int(item.get("round", 0)) for item in qa),
            default=0,
        )
        try:
            for round_number in range(
                completed_rounds + 1,
                int(self.config["review_rounds_per_direction"]) + 1,
            ):
                bundle = self._experiment_bundle(
                    direction, extra_claim_ids=extra_claim_ids
                )
                response = self.call(
                    role="judge",
                    prompt=self._judge_prompt(
                        bundle=bundle,
                        qa=qa,
                        round_number=round_number,
                        loop_context=loop_context,
                    ),
                    schema_name="judge",
                    conversation_key=f"judge:{direction_id}:r{round_number}",
                    session_id=judge_session,
                )
                judge_session = (
                    response.session_id if resume_review_sessions else ""
                )
                last_provider = response.provider
                last_model = response.model
                with self.state_lock:
                    self.state["sessions"]["judge"][direction_id] = (
                        self.session_record(response)
                    )
                    self.checkpoint()
                action = response.data["action"]
                if action == "complete":
                    covered_dimensions = {
                        str(item.get("dimension"))
                        for item in qa
                        if item.get("action") == "ask"
                        and isinstance(item.get("answer"), dict)
                    }
                    missing_dimensions = [
                        dimension
                        for dimension in REVIEW_DIMENSIONS
                        if dimension not in covered_dimensions
                    ]
                    if missing_dimensions:
                        loop_context = (
                            "completion gate rejected the previous complete action; "
                            "ask one question for a pending dimension: "
                            + ", ".join(missing_dimensions)
                        )
                        self.event(
                            "review_completion_rejected",
                            direction_id=direction_id,
                            round=round_number,
                            missing_dimensions=missing_dimensions,
                        )
                        continue
                    review = response.data.get("review", {})
                    errors = self._validate_complete_review(
                        review=review,
                        bundle=bundle,
                    )
                    if errors:
                        raise ProviderError(
                            "complete review failed semantic validation: "
                            + "; ".join(errors[:20])
                        )
                    self._write_review_artifact(
                        direction=direction,
                        status="complete",
                        qa=qa,
                        rounds=round_number,
                        review=review,
                        provider=last_provider,
                        model=last_model,
                    )
                    self._update_direction_review_status(
                        direction_id, str(review["decision"])
                    )
                    return
                if action == "request_evidence":
                    if response.data.get("dimension") == "complete":
                        raise ProviderError(
                            "request_evidence action must name a review dimension"
                        )
                    request = response.data.get("evidence_request", {})
                    query_key = normalize_key(str(request.get("query", "")))
                    if (
                        requests_used >= max_requests
                        or not query_key
                        or query_key in seen_evidence_queries
                    ):
                        self.event(
                            "evidence_request_rejected",
                            scope=f"review-{direction_id}",
                            query=request.get("query", ""),
                            reason="budget exhausted, empty, or repeated normalized query",
                        )
                        loop_context = (
                            "targeted retrieval unavailable or exhausted; resolve with "
                            "needs_evidence and an explicit gap"
                        )
                    else:
                        seen_evidence_queries.add(query_key)
                        requests_used += 1
                        new_ids = self.targeted_discovery(
                            request,
                            scope=f"review-{direction_id}",
                        )
                        extra_claim_ids = sorted({*extra_claim_ids, *new_ids})
                        qa.append(
                            {
                                "round": round_number,
                                "action": "request_evidence",
                                "dimension": response.data.get("dimension"),
                                "request": request,
                                "accepted_claim_ids": new_ids,
                            }
                        )
                        loop_context = (
                            "targeted evidence was added; reassess only the affected dimension"
                        )
                    continue
                dimension = normalize_text(str(response.data.get("dimension", "")))
                question = normalize_text(str(response.data.get("question", "")))
                if not dimension or dimension == "complete" or not question:
                    raise ProviderError("judge ask action omitted dimension or question")
                previous_questions = {
                    normalize_key(str(item.get("question", ""))) for item in qa
                }
                if normalize_key(question) in previous_questions:
                    raise ProviderError("judge repeated an already answered question")
                evidence_response = self.call(
                    role="evidence",
                    prompt=self._evidence_answer_prompt(
                        bundle=bundle,
                        question=question,
                        dimension=dimension,
                        round_number=round_number,
                    ),
                    schema_name="answer",
                    conversation_key=f"evidence:{direction_id}:r{round_number}",
                    session_id=evidence_session,
                )
                evidence_session = (
                    evidence_response.session_id if resume_review_sessions else ""
                )
                with self.state_lock:
                    self.state["sessions"]["evidence"][direction_id] = (
                        self.session_record(evidence_response)
                    )
                    self.checkpoint()
                answer = evidence_response.data
                valid_claim_ids = {
                    str(item["claim_id"])
                    for item in bundle.get("evidence_claims", [])
                }
                errors = validate_references(
                    answer.get("evidence_refs", []),
                    valid_claim_ids,
                    "evidence answer",
                )
                if errors:
                    raise ProviderError("; ".join(errors))
                qa.append(
                    {
                        "round": round_number,
                        "action": "ask",
                        "dimension": dimension,
                        "question": question,
                        "answer": answer,
                    }
                )
                loop_context = "continue to the next decision-relevant unresolved dimension"
        except Exception as exc:  # preserve partial expert work for audit/resume
            self._write_review_artifact(
                direction=direction,
                status="pending",
                qa=qa,
                rounds=len(qa),
                pending_reason=f"{type(exc).__name__}: {exc}",
                provider=last_provider,
                model=last_model,
            )
            self.event(
                "direction_review_failed",
                direction_id=direction_id,
                error_summary=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=8),
            )
            return
        self._write_review_artifact(
            direction=direction,
            status="pending",
            qa=qa,
            rounds=int(self.config["review_rounds_per_direction"]),
            pending_reason="review round budget exhausted before a valid complete action",
            provider=last_provider,
            model=last_model,
        )

    def run_reviews(self) -> None:
        self.set_phase("direction_review")
        directions_doc = read_json(
            self.work_dir / "directions" / "directions.json",
            _initial_directions(),
        )
        for direction in directions_doc.get("directions", []):
            artifact = self.work_dir / "reviews" / f"{direction['direction_id']}.json"
            existing = read_json(artifact)
            if isinstance(existing, dict) and existing.get("status") == "complete":
                continue
            self.review_direction(direction)
        completed = 0
        pending = 0
        for path in sorted((self.work_dir / "reviews").glob("*.json")):
            artifact = read_json(path, {})
            if artifact.get("status") == "complete":
                completed += 1
            else:
                pending += 1
        reconcile_dispositions(self.work_dir)
        self.event(
            "direction_review_completed",
            completed=completed,
            pending=pending,
        )
        self.mark_phase_completed("direction_review")

    def run_all(self, *, resume: bool, stop_after: str = "") -> dict[str, Any]:
        if resume:
            self.reset_for_resume()
        elif self.state.get("phase") != "initialized":
            raise ValueError(
                "existing run has progressed; pass --resume to reconcile and continue"
            )
        phase_methods = (
            ("discovery", self.run_discovery),
            ("curation", self.run_curation),
            ("direction_build", self.run_direction_build),
            ("direction_review", self.run_reviews),
        )
        completed = set(
            str(item) for item in self.state.get("completed_phases", [])
        )
        for phase, method in phase_methods:
            if phase in completed:
                self.event("phase_skipped_on_resume", skipped_phase=phase)
                continue
            method()
            completed.add(phase)
            if stop_after == phase:
                return {"stopped_after": phase}
        validation = validate_run(self.work_dir, write=True)
        if validation["errors"]:
            with self.state_lock:
                self.state["status"] = "invalid"
                self.checkpoint()
            self.event(
                "validation_failed",
                error_count=len(validation["errors"]),
                warning_count=len(validation["warnings"]),
            )
            return validation
        self.set_phase("validated", "validated")
        self.mark_phase_completed("validated")
        render_run(self.work_dir)
        final_validation = validate_run(self.work_dir, write=True)
        if final_validation["errors"]:
            with self.state_lock:
                self.state["status"] = "invalid"
                self.checkpoint()
            return final_validation
        self.set_phase(
            "rendered",
            "complete" if not final_validation["warnings"] else "complete_with_warnings",
        )
        self.mark_phase_completed("rendered")
        self.event(
            "workflow_completed",
            warning_count=len(final_validation["warnings"]),
            final_artifact="final.md",
        )
        return final_validation


def _claim_validation_errors(
    claim: Mapping[str, Any],
    *,
    vault_root: Path = VAULT_ROOT,
) -> list[str]:
    candidate = {
        key: claim.get(key)
        for key in (
            "statement", "claim_type", "layer", "entity_name", "source_path",
            "line_start", "line_end", "quote", "evidence_mode", "scope", "confidence",
        )
    }
    rebuilt, errors = candidate_to_claim(candidate, vault_root=vault_root)
    if rebuilt is not None and rebuilt["claim_id"] != claim.get("claim_id"):
        errors.append(
            f"claim stable ID mismatch: stored={claim.get('claim_id')} "
            f"expected={rebuilt['claim_id']}"
        )
    return errors


def validate_run(work_dir: Path, *, write: bool = False) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    config = read_json(work_dir / "config.json")
    state = read_json(work_dir / "state.json")
    if not isinstance(config, dict):
        errors.append("config.json is missing or not an object")
        config = {}
    if not isinstance(state, dict):
        errors.append("state.json is missing or not an object")
        state = {}
    if config.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("config protocol_version mismatch")
    if config.get("synthetic_run"):
        warnings.append("run uses synthetic fixture provider output")
    if state.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("state protocol_version mismatch")

    tasks = state.get("tasks", {}) if isinstance(state.get("tasks", {}), dict) else {}
    for task_id, task in tasks.items():
        status = task.get("status")
        if status not in DISCOVERY_TASK_STATUSES:
            errors.append(f"task {task_id} has invalid status {status!r}")
        if status == "done":
            artifact = work_dir / str(task.get("artifact", ""))
            if not artifact.is_file():
                errors.append(f"task {task_id} is done but artifact is missing")
        if status not in TERMINAL_TASK_STATUSES:
            warnings.append(f"task {task_id} is not terminal: {status}")
        if status == "failed_terminal":
            warnings.append(f"task {task_id} failed terminally: {task.get('last_error', '')}")
    counts["tasks"] = len(tasks)

    try:
        claims = read_jsonl(work_dir / "evidence" / "claims.jsonl")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"claims ledger cannot parse: {exc}")
        claims = []
    claim_index: dict[str, dict[str, Any]] = {}
    for row_number, claim in enumerate(claims, 1):
        claim_id = str(claim.get("claim_id", ""))
        if not claim_id:
            errors.append(f"claim row {row_number} has no claim_id")
            continue
        previous = claim_index.get(claim_id)
        if previous is not None and canonical_json(previous) != canonical_json(claim):
            errors.append(f"duplicate claim ID with conflicting content: {claim_id}")
        claim_index[claim_id] = claim
        for error in _claim_validation_errors(claim):
            errors.append(f"claim {claim_id}: {error}")
    valid_claim_ids = set(claim_index)
    counts["claims"] = len(claim_index)

    catalog = read_json(work_dir / "catalog" / "entities.json", _initial_catalog())
    if not isinstance(catalog, dict) or not isinstance(catalog.get("entities"), list):
        errors.append("catalog/entities.json has invalid shape")
        catalog = _initial_catalog()
    entity_index: dict[str, dict[str, Any]] = {}
    for entity in catalog["entities"]:
        entity_id = str(entity.get("entity_id", ""))
        if not entity_id:
            errors.append("entity missing entity_id")
            continue
        if entity_id in entity_index and canonical_json(entity_index[entity_id]) != canonical_json(entity):
            errors.append(f"duplicate entity ID with conflicting content: {entity_id}")
        expected = entity_id_for(
            name=str(entity.get("name", "")),
            entity_type=str(entity.get("entity_type", "")),
        )
        if expected != entity_id:
            errors.append(f"entity {entity_id} stable ID mismatch; expected {expected}")
        if entity.get("entity_type") not in (
            "method", "system", "code", "tool", "hardware", "dataset", "metric",
        ):
            errors.append(f"entity {entity_id} has invalid entity_type")
        errors.extend(
            validate_references(
                entity.get("evidence_refs", []),
                valid_claim_ids,
                f"entity {entity_id} evidence",
            )
        )
        entity_index[entity_id] = entity
    counts["entities"] = len(entity_index)

    anchors_doc = read_json(work_dir / "anchors" / "anchors.json", _initial_anchors())
    if not isinstance(anchors_doc, dict) or not isinstance(anchors_doc.get("anchors"), list):
        errors.append("anchors/anchors.json has invalid shape")
        anchors_doc = _initial_anchors()
    anchor_index: dict[str, dict[str, Any]] = {}
    baseline_index: dict[str, dict[str, Any]] = {}
    entry_index: dict[str, dict[str, Any]] = {}
    edge_index: dict[str, dict[str, Any]] = {}
    for anchor in anchors_doc["anchors"]:
        anchor_id = str(anchor.get("anchor_id", ""))
        if not anchor_id:
            errors.append("anchor missing anchor_id")
            continue
        if anchor.get("status") not in (
            "candidate", "active", "needs_evidence", "rejected",
        ):
            errors.append(f"Anchor {anchor_id} has invalid status")
        for field in ("workload", "phase", "regime", "backend", "bottleneck"):
            if not normalize_text(str(anchor.get(field, ""))):
                errors.append(f"Anchor {anchor_id} has empty {field}")
        if anchor_id in anchor_index and canonical_json(anchor_index[anchor_id]) != canonical_json(anchor):
            errors.append(f"duplicate Anchor ID with conflicting content: {anchor_id}")
        baselines = anchor.get("baselines", [])
        primary = next(
            (
                item
                for item in baselines
                if item.get("baseline_id") == anchor.get("primary_baseline_id")
            ),
            None,
        )
        if primary is None:
            errors.append(f"Anchor {anchor_id} primary baseline does not resolve")
            primary_name = ""
        else:
            primary_name = str(primary.get("name", ""))
        expected_anchor_id = anchor_id_for(
            workload=str(anchor.get("workload", "")),
            phase=str(anchor.get("phase", "")),
            regime=str(anchor.get("regime", "")),
            backend=str(anchor.get("backend", "")),
            bottleneck=str(anchor.get("bottleneck", "")),
            target_metrics=anchor.get("target_metrics", []),
            primary_baseline_name=primary_name,
        )
        if expected_anchor_id != anchor_id:
            errors.append(
                f"Anchor {anchor_id} stable ID mismatch; expected {expected_anchor_id}"
            )
        errors.extend(
            validate_references(
                anchor.get("evidence_refs", []),
                valid_claim_ids,
                f"Anchor {anchor_id} evidence",
            )
        )
        roles = {str(item.get("role")) for item in baselines}
        for role in BASELINE_ROLES:
            if role not in roles:
                warnings.append(f"Anchor {anchor_id} missing baseline role: {role}")
        local_baseline_ids: set[str] = set()
        for baseline in baselines:
            baseline_id = str(baseline.get("baseline_id", ""))
            expected = baseline_id_for(
                anchor_id=anchor_id,
                role=str(baseline.get("role", "")),
                name=str(baseline.get("name", "")),
            )
            if expected != baseline_id:
                errors.append(
                    f"baseline {baseline_id} stable ID mismatch; expected {expected}"
                )
            if baseline.get("anchor_id") != anchor_id:
                errors.append(f"baseline {baseline_id} crosses Anchor")
            if baseline.get("role") not in BASELINE_ROLES:
                errors.append(f"baseline {baseline_id} has invalid role")
            if baseline_id in baseline_index and canonical_json(baseline_index[baseline_id]) != canonical_json(baseline):
                errors.append(f"duplicate baseline ID with conflicting content: {baseline_id}")
            errors.extend(
                validate_references(
                    baseline.get("evidence_refs", []),
                    valid_claim_ids,
                    f"baseline {baseline_id} evidence",
                )
            )
            baseline_index[baseline_id] = baseline
            local_baseline_ids.add(baseline_id)
        local_entry_ids: set[str] = set()
        for entry in anchor.get("entries", []):
            entry_id = str(entry.get("entry_id", ""))
            if entry.get("anchor_id") != anchor_id:
                errors.append(f"entry {entry_id} crosses Anchor")
            if entry.get("entity_id") not in entity_index:
                errors.append(f"entry {entry_id} references missing entity")
            if entry.get("layer") not in LAYERS:
                errors.append(f"entry {entry_id} has invalid layer")
            if entry.get("role") not in (
                "baseline_behavior", "opportunity", "method",
                "implementation", "constraint", "evaluation",
            ):
                errors.append(f"entry {entry_id} has invalid role")
            if entry.get("confidence") not in ("high", "middle", "low"):
                errors.append(f"entry {entry_id} has invalid confidence")
            if entry.get("status") not in (
                "candidate", "accepted", "needs_evidence", "rejected",
            ):
                errors.append(f"entry {entry_id} has invalid status")
            expected = entry_id_for(
                anchor_id=anchor_id,
                layer=str(entry.get("layer", "")),
                role=str(entry.get("role", "")),
                entity_id=str(entry.get("entity_id", "")),
                claim=str(entry.get("claim", "")),
            )
            if expected != entry_id:
                errors.append(f"entry {entry_id} stable ID mismatch; expected {expected}")
            errors.extend(
                validate_references(
                    entry.get("applicable_baseline_ids", []),
                    local_baseline_ids,
                    f"entry {entry_id} baseline",
                )
            )
            errors.extend(
                validate_references(
                    entry.get("evidence_refs", []),
                    valid_claim_ids,
                    f"entry {entry_id} evidence",
                )
            )
            if entry_id in entry_index and canonical_json(entry_index[entry_id]) != canonical_json(entry):
                errors.append(f"duplicate entry ID with conflicting content: {entry_id}")
            entry_index[entry_id] = entry
            local_entry_ids.add(entry_id)
        for layer in LAYERS:
            if not any(item.get("layer") == layer for item in anchor.get("entries", [])):
                warnings.append(f"Anchor {anchor_id} has no {layer} entries")
        for edge in anchor.get("edges", []):
            edge_id = str(edge.get("edge_id", ""))
            if edge.get("anchor_id") != anchor_id:
                errors.append(f"edge {edge_id} crosses Anchor")
            if edge.get("relation") not in (
                "depends_on", "enables", "controls", "consumes", "produces",
                "complements", "substitutes", "conflicts_with", "measures",
            ):
                errors.append(f"edge {edge_id} has invalid relation")
            if edge.get("compatibility") not in (
                "compatible", "conditional", "incompatible", "unknown",
            ):
                errors.append(f"edge {edge_id} has invalid compatibility")
            if edge.get("confidence") not in ("high", "middle", "low"):
                errors.append(f"edge {edge_id} has invalid confidence")
            from_id = str(edge.get("from_entry_id", ""))
            to_id = str(edge.get("to_entry_id", ""))
            if from_id not in local_entry_ids or to_id not in local_entry_ids:
                errors.append(f"edge {edge_id} has missing/cross-Anchor endpoint")
            expected = edge_id_for(
                anchor_id=anchor_id,
                from_entry_id=from_id,
                to_entry_id=to_id,
                relation=str(edge.get("relation", "")),
                interface=str(edge.get("interface", "")),
            )
            if expected != edge_id:
                errors.append(f"edge {edge_id} stable ID mismatch; expected {expected}")
            errors.extend(
                validate_references(
                    edge.get("evidence_refs", []),
                    valid_claim_ids,
                    f"edge {edge_id} evidence",
                )
            )
            if edge_id in edge_index and canonical_json(edge_index[edge_id]) != canonical_json(edge):
                errors.append(f"duplicate edge ID with conflicting content: {edge_id}")
            edge_index[edge_id] = edge
        anchor_index[anchor_id] = anchor
    counts["anchors"] = len(anchor_index)
    counts["baselines"] = len(baseline_index)
    counts["entries"] = len(entry_index)
    counts["edges"] = len(edge_index)
    if not anchor_index:
        warnings.append("no evidence-validated Anchor was produced")
    if not baseline_index:
        warnings.append("no evidence-validated baseline was produced")

    directions_doc = read_json(
        work_dir / "directions" / "directions.json", _initial_directions()
    )
    if not isinstance(directions_doc, dict) or not isinstance(
        directions_doc.get("directions"), list
    ):
        errors.append("directions/directions.json has invalid shape")
        directions_doc = _initial_directions()
    direction_index: dict[str, dict[str, Any]] = {}
    for direction in directions_doc["directions"]:
        direction_id = str(direction.get("direction_id", ""))
        anchor_id = str(direction.get("anchor_id", ""))
        anchor = anchor_index.get(anchor_id)
        if not anchor:
            errors.append(f"Direction {direction_id} references missing Anchor {anchor_id}")
            continue
        if direction.get("kind") not in (
            "experiment", "baseline_reference",
            "implementation_reference", "method_reference",
        ):
            errors.append(f"Direction {direction_id} has invalid kind")
        if direction.get("status") not in (
            "candidate", "needs_evidence", "reviewed", "rejected",
        ):
            errors.append(f"Direction {direction_id} has invalid status")
        expected = direction_id_for(
            anchor_id=anchor_id,
            title=str(direction.get("title", "")),
            selected_entry_ids=direction.get("selected_entry_ids", []),
            hypothesis=str(direction.get("hypothesis", "")),
        )
        if expected != direction_id:
            errors.append(
                f"Direction {direction_id} stable ID mismatch; expected {expected}"
            )
        local_entries = {
            str(item["entry_id"]): item for item in anchor.get("entries", [])
        }
        local_edges = {
            str(item["edge_id"]): item for item in anchor.get("edges", [])
        }
        local_baselines = {
            str(item["baseline_id"]): item for item in anchor.get("baselines", [])
        }
        selected_entries = set(str(item) for item in direction.get("selected_entry_ids", []))
        selected_edges = set(str(item) for item in direction.get("selected_edge_ids", []))
        errors.extend(
            validate_references(selected_entries, set(local_entries), f"Direction {direction_id} entry")
        )
        for entry_id in selected_entries:
            entry = local_entries.get(entry_id)
            if entry and entry.get("status") == "rejected":
                errors.append(
                    f"Direction {direction_id} selects rejected entry {entry_id}"
                )
        errors.extend(
            validate_references(selected_edges, set(local_edges), f"Direction {direction_id} edge")
        )
        errors.extend(
            validate_references(
                direction.get("baseline_ids", []),
                set(local_baselines),
                f"Direction {direction_id} baseline",
            )
        )
        errors.extend(
            validate_references(
                direction.get("evidence_refs", []),
                valid_claim_ids,
                f"Direction {direction_id} evidence",
            )
        )
        for edge_id in selected_edges:
            edge = local_edges.get(edge_id)
            if not edge:
                continue
            endpoints = {str(edge["from_entry_id"]), str(edge["to_entry_id"])}
            if not endpoints.issubset(selected_entries):
                errors.append(
                    f"Direction {direction_id} selects edge {edge_id} without both endpoints"
                )
            if (
                edge.get("compatibility") in ("incompatible", "unknown")
                or edge.get("relation") in ("conflicts_with", "substitutes")
            ):
                errors.append(
                    f"Direction {direction_id} selects incompatible/unknown/"
                    f"conflict edge {edge_id}"
                )
            if (
                edge.get("compatibility") == "conditional"
                and not normalize_text(str(edge.get("condition", "")))
            ):
                errors.append(
                    f"Direction {direction_id} selects conditional edge {edge_id} "
                    "without a condition"
                )
        selected_edge_objects = [
            local_edges[edge_id] for edge_id in selected_edges if edge_id in local_edges
        ]
        if not _selected_entries_are_connected(
            selected_entries,
            selected_edge_objects,
        ):
            errors.append(
                f"Direction {direction_id} selected entries are not one connected subgraph"
            )
        if direction.get("kind") == "experiment":
            if not direction.get("baseline_ids"):
                errors.append(f"experiment Direction {direction_id} has no baseline")
            if not normalize_text(str(direction.get("hypothesis", ""))):
                errors.append(f"experiment Direction {direction_id} has no hypothesis")
        if len(
            {
                local_entries[item]["layer"]
                for item in selected_entries
                if item in local_entries
            }
        ) <= 1:
            warnings.append(f"Direction {direction_id} is single-layer")
        if direction_id in direction_index and canonical_json(direction_index[direction_id]) != canonical_json(direction):
            errors.append(f"duplicate Direction ID with conflicting content: {direction_id}")
        direction_index[direction_id] = direction
    counts["directions"] = len(direction_index)
    if not direction_index:
        warnings.append("no Direction was produced; baseline registry may still be useful")
    for gap in state.get("gaps", []) if isinstance(state.get("gaps", []), list) else []:
        warnings.append(f"run gap: {gap}")

    complete_reviews = 0
    pending_reviews = 0
    for direction_id, direction in direction_index.items():
        path = work_dir / "reviews" / f"{direction_id}.json"
        artifact = read_json(path)
        if not isinstance(artifact, dict):
            warnings.append(f"Direction {direction_id} has no review artifact")
            continue
        if artifact.get("direction_id") != direction_id:
            errors.append(f"review {path.name} references wrong Direction")
        if artifact.get("status") not in ("complete", "pending"):
            errors.append(f"review {path.name} has invalid status")
        expected_review_id = stable_id(
            "R",
            direction_id,
            artifact.get("status"),
            artifact.get("review", {}),
            artifact.get("pending_reason", ""),
        )
        if artifact.get("review_id") != expected_review_id:
            errors.append(
                f"review {path.name} stable ID mismatch; expected {expected_review_id}"
            )
        if artifact.get("status") == "complete":
            complete_reviews += 1
            covered_dimensions = {
                str(item.get("dimension"))
                for item in artifact.get("qa", [])
                if item.get("action") == "ask"
                and isinstance(item.get("answer"), dict)
            }
            missing_dimensions = [
                dimension
                for dimension in REVIEW_DIMENSIONS
                if dimension not in covered_dimensions
            ]
            if missing_dimensions:
                errors.append(
                    f"review {direction_id} completed without dimensions: "
                    + ", ".join(missing_dimensions)
                )
            review = artifact.get("review")
            schema_errors = validate_schema(review, REVIEW_SCHEMA)
            errors.extend(
                f"review {direction_id}: {error}" for error in schema_errors
            )
            if isinstance(review, dict):
                errors.extend(
                    validate_references(
                        review.get("evidence_refs", []),
                        valid_claim_ids,
                        f"review {direction_id} evidence",
                    )
                )
                errors.extend(
                    validate_references(
                        review.get("entry_refs", []),
                        set(direction.get("selected_entry_ids", [])),
                        f"review {direction_id} entry",
                    )
                )
                errors.extend(
                    validate_references(
                        review.get("edge_refs", []),
                        set(direction.get("selected_edge_ids", [])),
                        f"review {direction_id} edge",
                    )
                )
                errors.extend(
                    validate_references(
                        review.get("baseline_refs", []),
                        set(direction.get("baseline_ids", [])),
                        f"review {direction_id} baseline",
                    )
                )
                anchor = anchor_index.get(str(direction.get("anchor_id")), {})
                alternative_entries = {
                    str(item["entry_id"])
                    for item in anchor.get("entries", [])
                    if item["entry_id"]
                    not in set(direction.get("selected_entry_ids", []))
                }
                alternative_edges = {
                    str(item["edge_id"])
                    for item in anchor.get("edges", [])
                    if item["edge_id"]
                    not in set(direction.get("selected_edge_ids", []))
                }
                errors.extend(
                    validate_references(
                        review.get("alternative_entry_refs", []),
                        alternative_entries,
                        f"review {direction_id} alternative entry",
                    )
                )
                errors.extend(
                    validate_references(
                        review.get("alternative_edge_refs", []),
                        alternative_edges,
                        f"review {direction_id} alternative edge",
                    )
                )
                if (
                    review.get("decision") == "experiment_candidate"
                    and not direction.get("baseline_ids")
                ):
                    errors.append(
                        f"reviewed experiment candidate {direction_id} has no baseline"
                    )
                errors.extend(
                    f"review {direction_id}: {error}"
                    for error in _review_semantic_errors(review, direction)
                )
                expected_status = {
                    "experiment_candidate": "reviewed",
                    "needs_evidence": "needs_evidence",
                    "baseline_reference": "reviewed",
                    "rejected": "rejected",
                }.get(str(review.get("decision", "")))
                if expected_status and direction.get("status") != expected_status:
                    errors.append(
                        f"Direction {direction_id} status {direction.get('status')} "
                        f"does not match review decision {review.get('decision')}"
                    )
        else:
            pending_reviews += 1
            warnings.append(
                f"Direction {direction_id} review pending: "
                f"{artifact.get('pending_reason', '')}"
            )
    counts["complete_reviews"] = complete_reviews
    counts["pending_reviews"] = pending_reviews

    dispositions_doc = read_json(
        work_dir / "curation" / "dispositions.json",
        {"dispositions": []},
    )
    if not isinstance(dispositions_doc, dict) or not isinstance(
        dispositions_doc.get("dispositions"), list
    ):
        errors.append("curation/dispositions.json has invalid shape")
        dispositions_doc = {"dispositions": []}
    disposition_index: dict[str, dict[str, Any]] = {}
    for disposition in dispositions_doc["dispositions"]:
        claim_id = str(disposition.get("claim_id", ""))
        if claim_id not in valid_claim_ids:
            errors.append(f"disposition references unknown claim {claim_id}")
            continue
        if disposition.get("status") not in (
            "integrated", "duplicate", "irrelevant_to_scope",
            "needs_evidence", "invalid",
        ):
            errors.append(f"claim {claim_id} has invalid disposition status")
        if claim_id in disposition_index and canonical_json(
            disposition_index[claim_id]
        ) != canonical_json(disposition):
            errors.append(f"claim {claim_id} has conflicting dispositions")
        disposition_index[claim_id] = disposition
    for claim_id in valid_claim_ids:
        if claim_id not in disposition_index:
            errors.append(f"claim {claim_id} has no curation disposition")
    used_refs = canonical_evidence_refs(work_dir)
    for claim_id in valid_claim_ids:
        status = disposition_index.get(claim_id, {}).get("status")
        if claim_id in used_refs and status != "integrated":
            errors.append(
                f"claim {claim_id} is used by a canonical artifact but not integrated"
            )
        if claim_id not in used_refs and status == "integrated":
            errors.append(
                f"claim {claim_id} is marked integrated but has no canonical reference"
            )
    counts["dispositions"] = len(disposition_index)

    final_path = work_dir / "final.md"
    if final_path.is_file():
        final_text = final_path.read_text(encoding="utf-8", errors="replace")
        for baseline_id in baseline_index:
            if baseline_id not in final_text:
                errors.append(f"final.md omits baseline {baseline_id}")
        for direction_id in direction_index:
            if direction_id not in final_text:
                errors.append(f"final.md omits Direction {direction_id}")

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "validated_at": utc_now(),
        "valid": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "counts": counts,
    }
    if write:
        atomic_write_json(work_dir / "validation.json", result)
    return result


def render_review_markdown(
    direction: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> str:
    lines = [
        f"# {direction.get('title', direction.get('direction_id', 'Direction'))}",
        "",
        f"- Direction: `{direction.get('direction_id', '')}`",
        f"- Anchor: `{direction.get('anchor_id', '')}`",
        f"- Review status: `{artifact.get('status', '')}`",
        f"- Provider/model: `{artifact.get('provider', '')}` / `{artifact.get('model', '')}`",
        "",
    ]
    if artifact.get("status") != "complete":
        lines.extend(
            [
                "## Pending reason",
                "",
                str(artifact.get("pending_reason", "")),
                "",
            ]
        )
    else:
        review = artifact.get("review", {})
        lines.extend(
            [
                "## Decision",
                "",
                f"`{review.get('decision', '')}`",
                "",
                "| Axis | Rating |",
                "|---|---|",
                f"| Exploration | {review.get('exploration_value', '')} |",
                f"| Implementation reuse | {review.get('implementation_reuse', '')} |",
                f"| Method reference | {review.get('method_reference', '')} |",
                f"| Baseline quality | {review.get('baseline_quality', '')} |",
                f"| Cross-layer validity | {review.get('cross_layer_validity', '')} |",
                f"| Experiment readiness | {review.get('experiment_readiness', '')} |",
                "",
                "## Reasons",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in review.get("reasons", []))
        lines.extend(
            [
                "",
                "## Falsifiable hypothesis",
                "",
                str(review.get("falsifiable_hypothesis", "")),
                "",
                "## Implementation plan",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in review.get("implementation_plan", []))
        lines.extend(["", "## Baseline and ablation matrix", ""])
        lines.extend(f"- {item}" for item in review.get("baseline_ablation_matrix", []))
        lines.extend(["", "## Metrics", ""])
        lines.extend(f"- {item}" for item in review.get("metrics", []))
        lines.extend(["", "## Failure conditions", ""])
        lines.extend(f"- {item}" for item in review.get("failure_conditions", []))
        lines.extend(["", "## Gaps", ""])
        lines.extend(f"- {item}" for item in review.get("gaps", []))
        lines.extend(
            [
                "",
                "## Traceability",
                "",
                f"- Evidence: {', '.join(review.get('evidence_refs', []))}",
                f"- Entries: {', '.join(review.get('entry_refs', []))}",
                f"- Edges: {', '.join(review.get('edge_refs', []))}",
                f"- Alternative entries: {', '.join(review.get('alternative_entry_refs', []))}",
                f"- Alternative edges: {', '.join(review.get('alternative_edge_refs', []))}",
                f"- Baselines: {', '.join(review.get('baseline_refs', []))}",
                "",
            ]
        )
    if artifact.get("qa"):
        lines.extend(["## Review Q&A", ""])
        for item in artifact["qa"]:
            lines.append(f"### Round {item.get('round', '')}: {item.get('dimension', item.get('action', ''))}")
            lines.append("")
            if item.get("question"):
                lines.append(f"Question: {item['question']}")
                lines.append("")
            answer = item.get("answer")
            if isinstance(answer, dict):
                lines.append(f"Answer: {answer.get('answer', '')}")
                lines.append("")
                lines.append(
                    f"Evidence: {', '.join(answer.get('evidence_refs', []))}"
                )
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _rating_rank(value: str) -> int:
    return {"high": 0, "middle": 1, "low": 2}.get(value, 3)


def render_run(work_dir: Path) -> Path:
    work_dir = work_dir.resolve()
    config = read_json(work_dir / "config.json", {})
    state = read_json(work_dir / "state.json", {})
    validation = validate_run(work_dir, write=False)
    claims = read_jsonl(work_dir / "evidence" / "claims.jsonl")
    claim_index = {str(item["claim_id"]): item for item in claims}
    catalog = read_json(work_dir / "catalog" / "entities.json", _initial_catalog())
    anchors_doc = read_json(work_dir / "anchors" / "anchors.json", _initial_anchors())
    directions_doc = read_json(
        work_dir / "directions" / "directions.json", _initial_directions()
    )
    anchors = anchors_doc.get("anchors", [])
    directions = directions_doc.get("directions", [])
    reviews: dict[str, dict[str, Any]] = {}
    for direction in directions:
        artifact = read_json(
            work_dir / "reviews" / f"{direction['direction_id']}.json", {}
        )
        reviews[str(direction["direction_id"])] = artifact if isinstance(artifact, dict) else {}
        if isinstance(artifact, dict) and artifact:
            atomic_write_text(
                work_dir / "reviews" / f"{direction['direction_id']}.md",
                render_review_markdown(direction, artifact),
            )

    decision_order = {
        "experiment_candidate": 0,
        "needs_evidence": 1,
        "baseline_reference": 2,
        "": 3,
        "rejected": 4,
    }

    def direction_sort_key(direction: Mapping[str, Any]) -> tuple[Any, ...]:
        artifact = reviews.get(str(direction["direction_id"]), {})
        review = artifact.get("review", {}) if artifact.get("status") == "complete" else {}
        decision = str(review.get("decision", ""))
        if decision:
            outcome_rank = decision_order.get(decision, 3)
        else:
            outcome_rank = {
                "experiment": 1,
                "baseline_reference": 2,
                "implementation_reference": 2,
                "method_reference": 3,
            }.get(str(direction.get("kind", "")), 3)
        return (
            outcome_rank,
            _rating_rank(str(review.get("exploration_value", ""))),
            _rating_rank(str(review.get("implementation_reuse", ""))),
            _rating_rank(str(review.get("method_reference", ""))),
            {
                "experiment": 0,
                "implementation_reference": 1,
                "baseline_reference": 1,
                "method_reference": 2,
            }.get(str(direction.get("kind", "")), 3),
            str(direction["direction_id"]),
        )

    lines: list[str] = [
        f"# 分层实验探索结果：{config.get('topic', '')}",
        "",
        "## 运行信息",
        "",
        f"- Run ID: `{state.get('run_id', '')}`",
        f"- 协议版本: `{PROTOCOL_VERSION}`",
        f"- 范围约束: {config.get('constraints') or '未额外限定'}",
        f"- 证据声明数: {len(claims)}",
        f"- Anchor 数: {len(anchors)}",
        f"- Direction 数: {len(directions)}",
        "- 价值顺序: 可探索场景/加速机会 > 实现复用 > 论文方法参考；Baseline 独立保留。",
        "",
        "## 结果索引",
        "",
        "| Decision | Value (E/I/M) | Direction | Anchor | Kind |",
        "|---|---|---|---|---|",
    ]
    if config.get("synthetic_run"):
        lines[9:9] = [
            "- **SYNTHETIC FIXTURE RUN：本结果仅用于离线测试，不得作为研究结论。**",
        ]
    for direction in sorted(directions, key=direction_sort_key):
        artifact = reviews.get(str(direction["direction_id"]), {})
        review = artifact.get("review", {}) if artifact.get("status") == "complete" else {}
        decision = review.get("decision", "pending_review")
        ratings = "/".join(
            str(review.get(field, "-"))
            for field in ("exploration_value", "implementation_reuse", "method_reference")
        )
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(item)
                for item in (
                    decision,
                    ratings,
                    f"{direction['title']} (`{direction['direction_id']}`)",
                    f"`{direction['anchor_id']}`",
                    direction["kind"],
                )
            )
            + " |"
        )
    if not directions:
        lines.append("| — | — | 尚无可验证 Direction | — | — |")

    lines.extend(
        [
            "",
            "## Baseline / Reference Registry",
            "",
            "| Baseline ID | Anchor | Role | Name | Description | Evidence |",
            "|---|---|---|---|---|---|",
        ]
    )
    baseline_rows = [
        baseline
        for anchor in anchors
        for baseline in anchor.get("baselines", [])
    ]
    for baseline in sorted(baseline_rows, key=lambda item: item["baseline_id"]):
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(item)
                for item in (
                    f"`{baseline['baseline_id']}`",
                    f"`{baseline['anchor_id']}`",
                    baseline["role"],
                    baseline["name"],
                    baseline["description"],
                    ", ".join(f"`{item}`" for item in baseline["evidence_refs"]),
                )
            )
            + " |"
        )
    if not baseline_rows:
        lines.append("| — | — | — | 尚无通过证据验证的 baseline | — | — |")

    lines.extend(
        [
            "",
            "### Implementation / Method / Evaluation References",
            "",
            "| Object ID | Type | Anchor | Description | Evidence |",
            "|---|---|---|---|---|",
        ]
    )
    reference_rows: list[tuple[str, str, str, str, list[str]]] = []
    for anchor in anchors:
        for entry in anchor.get("entries", []):
            if entry.get("role") in ("implementation", "method", "evaluation"):
                reference_rows.append(
                    (
                        str(entry["entry_id"]),
                        f"entry:{entry['role']}",
                        str(anchor["anchor_id"]),
                        str(entry["claim"]),
                        list(entry.get("evidence_refs", [])),
                    )
                )
    for direction in directions:
        artifact = reviews.get(str(direction["direction_id"]), {})
        review = artifact.get("review", {}) if artifact.get("status") == "complete" else {}
        if (
            direction.get("kind") != "experiment"
            or review.get("decision") == "baseline_reference"
        ):
            reference_rows.append(
                (
                    str(direction["direction_id"]),
                    f"direction:{direction['kind']}",
                    str(direction["anchor_id"]),
                    str(direction["title"]),
                    list(direction.get("evidence_refs", [])),
                )
            )
    for object_id, object_type, anchor_id, description, evidence_refs in sorted(
        reference_rows,
        key=lambda item: item[0],
    ):
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(item)
                for item in (
                    f"`{object_id}`",
                    object_type,
                    f"`{anchor_id}`",
                    description,
                    ", ".join(f"`{item}`" for item in evidence_refs),
                )
            )
            + " |"
        )
    if not reference_rows:
        lines.append("| — | — | — | 尚无独立实现/方法/测量 reference | — |")

    lines.extend(["", "## Global L1-L6 Catalog", ""])
    for layer, layer_info in LAYERS.items():
        lines.extend(
            [
                f"### {layer} {layer_info['name']}",
                "",
                "| Entry ID | Anchor | Role | Entity | Claim | Baseline | Evidence |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        layer_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for anchor in anchors:
            for entry in anchor.get("entries", []):
                if entry.get("layer") == layer:
                    layer_entries.append((anchor, entry))
        for anchor, entry in sorted(
            layer_entries, key=lambda pair: pair[1]["entry_id"]
        ):
            entity = next(
                (
                    item
                    for item in catalog.get("entities", [])
                    if item["entity_id"] == entry["entity_id"]
                ),
                {"name": entry["entity_id"]},
            )
            lines.append(
                "| "
                + " | ".join(
                    markdown_cell(item)
                    for item in (
                        f"`{entry['entry_id']}`",
                        f"`{anchor['anchor_id']}`",
                        entry["role"],
                        f"{entity['name']} (`{entry['entity_id']}`)",
                        entry["claim"],
                        ", ".join(f"`{item}`" for item in entry["applicable_baseline_ids"]),
                        ", ".join(f"`{item}`" for item in entry["evidence_refs"]),
                    )
                )
                + " |"
            )
        if not layer_entries:
            lines.append("| — | — | — | — | 当前无通过验证的条目 | — | — |")
        lines.append("")

    lines.extend(["## Anchor Maps", ""])
    for anchor in sorted(anchors, key=lambda item: item["anchor_id"]):
        lines.extend(
            [
                f"### {anchor['anchor_id']}",
                "",
                "| Context field | Value |",
                "|---|---|",
                f"| Workload | {markdown_cell(anchor['workload'])} |",
                f"| Phase | {markdown_cell(anchor['phase'])} |",
                f"| Regime | {markdown_cell(anchor['regime'])} |",
                f"| Backend | {markdown_cell(anchor['backend'])} |",
                f"| Bottleneck | {markdown_cell(anchor['bottleneck'])} |",
                f"| Primary baseline | `{anchor['primary_baseline_id']}` |",
                f"| Target metrics | {markdown_cell(anchor['target_metrics'])} |",
                f"| Evidence | {markdown_cell(anchor['evidence_refs'])} |",
                "",
                "#### BaselineSet",
                "",
            ]
        )
        for baseline in anchor.get("baselines", []):
            lines.append(
                f"- `{baseline['baseline_id']}` [{baseline['role']}] "
                f"{baseline['name']}: {baseline['description']} "
                f"(evidence: {', '.join(baseline['evidence_refs'])})"
            )
        lines.extend(
            [
                "",
                "#### L1-L6 entries",
                "",
                "| Layer | Entry ID | Role | Modifiable object | Claim/effect | Status |",
                "|---|---|---|---|---|---|",
            ]
        )
        for layer in LAYERS:
            layer_entries = [
                entry
                for entry in anchor.get("entries", [])
                if entry.get("layer") == layer
            ]
            if not layer_entries:
                lines.append(f"| {layer} | — | — | — | 空 | — |")
            for entry in layer_entries:
                claim_effect = entry["claim"]
                if entry.get("expected_effect"):
                    claim_effect += f" → {entry['expected_effect']}"
                lines.append(
                    "| "
                    + " | ".join(
                        markdown_cell(item)
                        for item in (
                            layer,
                            f"`{entry['entry_id']}`",
                            entry["role"],
                            entry["modifiable_object"],
                            claim_effect,
                            entry["status"],
                        )
                    )
                    + " |"
                )
        lines.extend(
            [
                "",
                "#### Entry-level edges / alternatives",
                "",
                "| Edge | From → To | Relation | Interface | Compatibility / condition |",
                "|---|---|---|---|---|",
            ]
        )
        for edge in anchor.get("edges", []):
            lines.append(
                "| "
                + " | ".join(
                    markdown_cell(item)
                    for item in (
                        f"`{edge['edge_id']}`",
                        f"`{edge['from_entry_id']}` → `{edge['to_entry_id']}`",
                        edge["relation"],
                        edge["interface"],
                        f"{edge['compatibility']}; {edge['condition']}",
                    )
                )
                + " |"
            )
        if not anchor.get("edges"):
            lines.append("| — | — | — | 尚无通过验证的 entry-level edge | — |")
        if anchor.get("gaps"):
            lines.extend(["", "Gaps:"])
            lines.extend(f"- {gap}" for gap in anchor["gaps"])
        lines.append("")

    lines.extend(["## Direction Bundles 与专家评阅", ""])
    for direction in sorted(directions, key=direction_sort_key):
        artifact = reviews.get(str(direction["direction_id"]), {})
        review = artifact.get("review", {}) if artifact.get("status") == "complete" else {}
        lines.extend(
            [
                f"### {direction['title']} (`{direction['direction_id']}`)",
                "",
                f"- Anchor: `{direction['anchor_id']}`",
                f"- Kind: `{direction['kind']}`",
                f"- Selected entries: {', '.join(f'`{item}`' for item in direction['selected_entry_ids']) or '—'}",
                f"- Selected edges: {', '.join(f'`{item}`' for item in direction['selected_edge_ids']) or '—'}",
                f"- Baselines: {', '.join(f'`{item}`' for item in direction['baseline_ids']) or '—'}",
                f"- Hypothesis: {direction['hypothesis']}",
                f"- Preconditions: {markdown_cell(direction['preconditions']) or '—'}",
                f"- Expected effects: {markdown_cell(direction['expected_effects']) or '—'}",
                "",
                "Ablations:",
            ]
        )
        lines.extend(f"- {item}" for item in direction["ablation_plan"])
        lines.extend(["", f"Review status: `{artifact.get('status', 'missing')}`"])
        if review:
            lines.extend(
                [
                    "",
                    f"Decision: `{review['decision']}`; "
                    f"E/I/M = `{review['exploration_value']}/"
                    f"{review['implementation_reuse']}/{review['method_reference']}`",
                    "",
                    "Review reasons:",
                ]
            )
            lines.extend(f"- {item}" for item in review["reasons"])
            lines.extend(["", "Minimum implementation plan:"])
            lines.extend(f"- {item}" for item in review["implementation_plan"])
            lines.extend(["", "Baseline/ablation matrix:"])
            lines.extend(f"- {item}" for item in review["baseline_ablation_matrix"])
            lines.extend(["", "Failure conditions:"])
            lines.extend(f"- {item}" for item in review["failure_conditions"])
            if review.get("alternative_entry_refs") or review.get("alternative_edge_refs"):
                lines.extend(
                    [
                        "",
                        "Reviewed alternatives/conflicts:",
                        f"- Entries: {', '.join(f'`{item}`' for item in review.get('alternative_entry_refs', [])) or '—'}",
                        f"- Edges: {', '.join(f'`{item}`' for item in review.get('alternative_edge_refs', [])) or '—'}",
                    ]
                )
        elif artifact.get("pending_reason"):
            lines.extend(["", f"Pending reason: {artifact['pending_reason']}"])
        if direction.get("gaps"):
            lines.extend(["", "Direction gaps:"])
            lines.extend(f"- {item}" for item in direction["gaps"])
        lines.extend(
            [
                "",
                f"Evidence: {', '.join(f'`{item}`' for item in direction['evidence_refs'])}",
                "",
            ]
        )

    all_gaps: list[str] = list(state.get("gaps", []))
    for task in state.get("tasks", {}).values():
        all_gaps.extend(str(item) for item in task.get("gaps", []))
        if task.get("last_error"):
            all_gaps.append(f"{task['task_id']}: {task['last_error']}")
    for anchor in anchors:
        all_gaps.extend(str(item) for item in anchor.get("gaps", []))
    for direction in directions:
        all_gaps.extend(str(item) for item in direction.get("gaps", []))
        artifact = reviews.get(str(direction["direction_id"]), {})
        if artifact.get("pending_reason"):
            all_gaps.append(
                f"{direction['direction_id']}: {artifact['pending_reason']}"
            )
    lines.extend(["## 证据缺口与下一步", ""])
    if all_gaps:
        lines.extend(f"- {item}" for item in sorted(set(all_gaps)))
    else:
        lines.append("- 当前无已记录缺口。")

    lines.extend(
        [
            "",
            "## Evidence Claim 索引",
            "",
            "| Claim ID | Type/Layer | Statement | Source coordinate | Mode |",
            "|---|---|---|---|---|",
        ]
    )
    for claim_id, claim in sorted(claim_index.items()):
        coordinate = (
            f"{claim['source_path']}:{claim['line_start']}-{claim['line_end']}"
        )
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(item)
                for item in (
                    f"`{claim_id}`",
                    f"{claim['claim_type']}/{claim['layer']}",
                    claim["statement"],
                    coordinate,
                    claim["evidence_mode"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Provenance 与验证",
            "",
            "- 该结果由独立 Python 编排器生成；未导入、调用或修改 legacy TypeScript workflow。",
            "- 模型只提出结构化语义候选；本地脚本负责检索、引用核验、ID、图完整性和写盘。",
            f"- Validation: `{'valid' if validation['valid'] else 'invalid'}`; "
            f"errors={len(validation['errors'])}; warnings={len(validation['warnings'])}.",
            "- 完整机器可读验证见 `validation.json`，逐 Direction 评阅见 `reviews/`。",
            "",
        ]
    )
    final_path = work_dir / "final.md"
    atomic_write_text(final_path, "\n".join(lines))
    return final_path


def status_report(work_dir: Path) -> dict[str, Any]:
    state = read_json(work_dir / "state.json")
    if not isinstance(state, dict):
        raise ValueError(f"state.json missing in {work_dir}")
    task_counts: dict[str, int] = {}
    for task in state.get("tasks", {}).values():
        status = str(task.get("status", "unknown"))
        task_counts[status] = task_counts.get(status, 0) + 1
    anchors_doc = read_json(work_dir / "anchors" / "anchors.json", _initial_anchors())
    directions_doc = read_json(
        work_dir / "directions" / "directions.json", _initial_directions()
    )
    reviews = [
        read_json(path, {})
        for path in sorted((work_dir / "reviews").glob("*.json"))
    ]
    return {
        "run_id": state.get("run_id"),
        "topic": state.get("topic"),
        "phase": state.get("phase"),
        "status": state.get("status"),
        "updated_at": state.get("updated_at"),
        "task_counts": task_counts,
        "claim_count": len(read_jsonl(work_dir / "evidence" / "claims.jsonl")),
        "anchor_count": len(anchors_doc.get("anchors", [])),
        "direction_count": len(directions_doc.get("directions", [])),
        "review_complete": sum(
            1 for item in reviews if item.get("status") == "complete"
        ),
        "review_pending": sum(
            1 for item in reviews if item.get("status") != "complete"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evidence-preserving L1-L6 research exploration workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect local/provider readiness")
    doctor.add_argument("--json", action="store_true", help="emit JSON")

    init = subparsers.add_parser("init", help="initialize a new run directory")
    init.add_argument("--topic", required=True)
    init.add_argument("--work-dir", required=True)
    init.add_argument("--constraints", default="")
    init.add_argument(
        "--provider",
        choices=("deepseek-cli", "codex-cli", "fixture"),
        default="deepseek-cli",
    )
    for role in ("discovery", "curator", "direction", "review"):
        init.add_argument(
            f"--{role}-provider",
            choices=("deepseek-cli", "codex-cli", "fixture"),
            default="",
        )
    init.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    init.add_argument("--codex-model", default="")
    init.add_argument("--fixture-file", default="")
    init.add_argument("--knowledge-root", action="append")
    init.add_argument("--layers", default=",".join(LAYERS))
    init.add_argument("--axes", default=",".join(VALUE_AXES))
    init.add_argument("--provider-timeout-seconds", type=int, default=900)
    init.add_argument("--max-budget-usd-per-call", type=float, default=20.0)
    init.add_argument("--discovery-rounds", type=int, default=2)
    init.add_argument("--discovery-workers", type=int, default=3)
    init.add_argument("--snippets-per-round", type=int, default=12)
    init.add_argument("--snippet-context-lines", type=int, default=8)
    init.add_argument("--max-task-attempts", type=int, default=2)
    init.add_argument("--curation-batch-size", type=int, default=40)
    init.add_argument("--curator-rounds-per-batch", type=int, default=4)
    init.add_argument("--max-curator-evidence-requests", type=int, default=8)
    init.add_argument("--direction-rounds-per-anchor", type=int, default=4)
    init.add_argument("--max-direction-evidence-requests", type=int, default=4)
    init.add_argument("--review-rounds-per-direction", type=int, default=12)
    init.add_argument("--max-review-evidence-requests", type=int, default=3)
    init.add_argument(
        "--review-session-mode",
        choices=("stateless", "resume"),
        default="stateless",
        help=(
            "stateless rebuilds every Judge/Evidence turn from canonical Bundle+Q&A; "
            "resume additionally uses provider conversation history"
        ),
    )

    run = subparsers.add_parser("run", help="run or resume all workflow phases")
    run.add_argument("--work-dir", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--stop-after",
        choices=("discovery", "curation", "direction_build", "direction_review"),
        default="",
    )

    status = subparsers.add_parser("status", help="show compact run status")
    status.add_argument("--work-dir", required=True)
    status.add_argument("--json", action="store_true")

    validate = subparsers.add_parser("validate", help="validate canonical artifacts")
    validate.add_argument("--work-dir", required=True)
    validate.add_argument("--json", action="store_true")

    render = subparsers.add_parser("render", help="deterministically render final.md")
    render.add_argument("--work-dir", required=True)
    render.add_argument("--allow-invalid", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = doctor_report()
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print_doctor(report)
            return 0
        if args.command == "init":
            path = initialize_run(args)
            print(path)
            return 0
        work_dir = Path(args.work_dir).resolve()
        if args.command == "run":
            result = Workflow(work_dir).run_all(
                resume=args.resume,
                stop_after=args.stop_after,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if not result.get("errors") else 2
        if args.command == "status":
            report = status_report(work_dir)
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(
                    f"{report['run_id']} phase={report['phase']} "
                    f"status={report['status']} claims={report['claim_count']} "
                    f"anchors={report['anchor_count']} "
                    f"directions={report['direction_count']} "
                    f"reviews={report['review_complete']}/{report['review_pending']}"
                )
                print("tasks: " + canonical_json(report["task_counts"]))
            return 0
        if args.command == "validate":
            result = validate_run(work_dir, write=True)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(
                    f"valid={result['valid']} errors={len(result['errors'])} "
                    f"warnings={len(result['warnings'])}"
                )
                for error in result["errors"]:
                    print(f"ERROR: {error}")
                for warning in result["warnings"]:
                    print(f"WARN: {warning}")
            return 0 if result["valid"] else 2
        if args.command == "render":
            result = validate_run(work_dir, write=True)
            if result["errors"] and not args.allow_invalid:
                print(
                    "render refused because validation has errors; "
                    "use --allow-invalid for an audit rendering",
                    file=sys.stderr,
                )
                return 2
            path = render_run(work_dir)
            print(path)
            return 0
    except KeyboardInterrupt:
        print("interrupted; checkpoints and raw logs were preserved", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
