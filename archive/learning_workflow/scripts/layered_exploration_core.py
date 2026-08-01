#!/usr/bin/env python3
"""Archived core runtime for the standalone layered exploration workflow.

This module is intentionally independent from the legacy TypeScript schedulers.
It owns schemas, stable IDs, provider adapters, local retrieval, evidence
validation, artifact persistence, and deterministic validation/render helpers.

It is retained for historical audit and is not part of the current Simple
Semantic Loop runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
VAULT_ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = ARCHIVE_ROOT / "skills" / "layered-exploration-workflow"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash[1m]"
DEFAULT_KNOWLEDGE_ROOTS = (
    "paper_secs",
    "knowledge_notes",
    "experiment_notes",
    "idea_notes",
    "human_notes",
    "review_notes",
)
PROTOCOL_VERSION = 1

LAYERS: dict[str, dict[str, str]] = {
    "L1": {
        "name": "Algorithm/Pipeline",
        "scope": "graph, workload decomposition, algorithmic dynamicity, approximation, and independent subcomputations",
    },
    "L2": {
        "name": "Serving/Runtime",
        "scope": "requests, batches, stages, queues, placement, caching, and runtime resource scheduling",
    },
    "L3": {
        "name": "Compiler",
        "scope": "IR, dependency analysis, fusion, pass pipeline, multiversion, autotuning, and code generation",
    },
    "L4": {
        "name": "Kernel",
        "scope": "tile/warp/instruction pipeline, synchronization, layouts, data movement, and kernel composition",
    },
    "L5": {
        "name": "Architecture",
        "scope": "execution/control units, schedulers, memory hierarchy, NoC, and hardware concurrency primitives",
    },
    "L6": {
        "name": "Chip/System",
        "scope": "chiplet, PIM, wafer-scale, packaging, die-to-die links, and chip-level resource boundaries",
    },
}
VALUE_AXES = ("exploration", "implementation_reuse", "method_reference")
REVIEW_DIMENSIONS = (
    "scenario_opportunity",
    "baseline_fairness",
    "entry_validity",
    "cross_layer_validity",
    "implementation_reuse",
    "experiment_measurement",
)

LAYER_TERMS: dict[str, tuple[str, ...]] = {
    "L1": ("algorithm", "pipeline", "workload", "dynamic shape", "routing", "算法", "负载", "计算图"),
    "L2": ("serving", "runtime", "scheduler", "request", "batch", "调度", "运行时", "请求"),
    "L3": ("compiler", "IR", "codegen", "fusion", "multiversion", "编译", "中间表示", "代码生成"),
    "L4": ("kernel", "tile", "warp", "instruction", "pipeline", "同步", "流水", "算子"),
    "L5": ("architecture", "memory hierarchy", "scheduler", "NoC", "hardware", "架构", "存储层次", "硬件"),
    "L6": ("chiplet", "PIM", "wafer scale", "interconnect", "package", "芯片", "互联", "存内计算"),
}
AXIS_TERMS: dict[str, tuple[str, ...]] = {
    "exploration": (
        "bottleneck", "opportunity", "overlap", "speedup", "dynamic", "degradation",
        "瓶颈", "加速", "并发", "动态", "退化",
    ),
    "implementation_reuse": (
        "implementation", "code", "repository", "framework", "tool", "profiler",
        "实现", "代码", "框架", "工具", "模拟器",
    ),
    "method_reference": (
        "method", "mechanism", "baseline", "comparison", "constraint",
        "方法", "机制", "基线", "对比", "约束",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: Any, length: int = 14) -> str:
    payload = canonical_json(parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:length].upper()}"


def anchor_id_for(
    *,
    workload: str,
    phase: str,
    regime: str,
    backend: str,
    bottleneck: str,
    target_metrics: Sequence[str],
    primary_baseline_name: str,
) -> str:
    return stable_id(
        "A",
        identity_text(workload),
        identity_text(phase),
        identity_text(regime),
        identity_text(backend),
        identity_text(bottleneck),
        sorted(identity_text(metric) for metric in target_metrics),
        identity_text(primary_baseline_name),
    )


def baseline_id_for(*, anchor_id: str, role: str, name: str) -> str:
    return stable_id("B", anchor_id, role, identity_text(name))


def entity_id_for(*, name: str, entity_type: str) -> str:
    return stable_id("G", identity_text(name), entity_type)


def entry_id_for(
    *,
    anchor_id: str,
    layer: str,
    role: str,
    entity_id: str,
    claim: str,
) -> str:
    return stable_id(
        "E",
        anchor_id,
        layer,
        role,
        entity_id,
        identity_text(claim),
    )


def edge_id_for(
    *,
    anchor_id: str,
    from_entry_id: str,
    to_entry_id: str,
    relation: str,
    interface: str,
) -> str:
    return stable_id(
        "X",
        anchor_id,
        from_entry_id,
        to_entry_id,
        relation,
        identity_text(interface),
    )


def direction_id_for(
    *,
    anchor_id: str,
    title: str,
    selected_entry_ids: Sequence[str],
    hypothesis: str,
) -> str:
    return stable_id(
        "D",
        anchor_id,
        identity_text(title),
        sorted(selected_entry_ids),
        identity_text(hypothesis),
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def identity_text(text: str) -> str:
    return unicodedata.normalize("NFKC", normalize_text(text)).casefold()


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text.lower()).strip("-")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


_JSONL_LOCK = threading.Lock()


def append_jsonl(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    line = canonical_json(value) + "\n"
    with _JSONL_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
        rows.append(value)
    return rows


def load_reference(name: str) -> str:
    path = SKILL_ROOT / "references" / name
    return path.read_text(encoding="utf-8")


def make_object_schema(properties: Mapping[str, Any], required: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(required if required is not None else properties.keys()),
    }


STRING = {"type": "string"}
NONEMPTY_STRING = {"type": "string", "minLength": 1}
STRING_ARRAY = {"type": "array", "items": {"type": "string"}}

DISCOVERY_CANDIDATE_SCHEMA = make_object_schema(
    {
        "statement": NONEMPTY_STRING,
        "claim_type": {
            "type": "string",
            "enum": [
                "scenario", "baseline", "method", "implementation",
                "constraint", "metric", "relation", "evaluation",
            ],
        },
        "layer": {"type": "string", "enum": [*LAYERS.keys(), "GLOBAL"]},
        "entity_name": NONEMPTY_STRING,
        "source_path": NONEMPTY_STRING,
        "line_start": {"type": "integer", "minimum": 1},
        "line_end": {"type": "integer", "minimum": 1},
        "quote": NONEMPTY_STRING,
        "evidence_mode": {"type": "string", "enum": ["direct", "inferred"]},
        "scope": STRING,
        "confidence": {"type": "string", "enum": ["high", "middle", "low"]},
    }
)

DISCOVERY_SCHEMA = make_object_schema(
    {
        "action": {"type": "string", "enum": ["continue", "complete"]},
        "candidates": {"type": "array", "items": DISCOVERY_CANDIDATE_SCHEMA},
        "gaps": STRING_ARRAY,
        "next_queries": STRING_ARRAY,
    }
)

BASELINE_PROPOSAL_SCHEMA = make_object_schema(
    {
        "baseline_key": NONEMPTY_STRING,
        "name": NONEMPTY_STRING,
        "role": {
            "type": "string",
            "enum": ["current_practice", "strong", "tool_evaluation", "reusable_implementation"],
        },
        "description": NONEMPTY_STRING,
        "evidence_refs": STRING_ARRAY,
    }
)

ENTRY_PROPOSAL_SCHEMA = make_object_schema(
    {
        "entry_key": NONEMPTY_STRING,
        "entity_name": NONEMPTY_STRING,
        "entity_type": {
            "type": "string",
            "enum": ["method", "system", "code", "tool", "hardware", "dataset", "metric"],
        },
        "layer": {"type": "string", "enum": list(LAYERS.keys())},
        "role": {
            "type": "string",
            "enum": ["baseline_behavior", "opportunity", "method", "implementation", "constraint", "evaluation"],
        },
        "claim": NONEMPTY_STRING,
        "modifiable_object": STRING,
        "applicable_baseline_keys": STRING_ARRAY,
        "preconditions": STRING_ARRAY,
        "expected_effect": STRING,
        "evidence_refs": STRING_ARRAY,
        "confidence": {"type": "string", "enum": ["high", "middle", "low"]},
        "status": {"type": "string", "enum": ["candidate", "accepted", "needs_evidence", "rejected"]},
    }
)

EDGE_PROPOSAL_SCHEMA = make_object_schema(
    {
        "from_entry_key": NONEMPTY_STRING,
        "to_entry_key": NONEMPTY_STRING,
        "relation": {
            "type": "string",
            "enum": [
                "depends_on", "enables", "controls", "consumes", "produces",
                "complements", "substitutes", "conflicts_with", "measures",
            ],
        },
        "interface": NONEMPTY_STRING,
        "compatibility": {
            "type": "string",
            "enum": ["compatible", "conditional", "incompatible", "unknown"],
        },
        "condition": STRING,
        "evidence_refs": STRING_ARRAY,
        "confidence": {"type": "string", "enum": ["high", "middle", "low"]},
    }
)

ANCHOR_PROPOSAL_SCHEMA = make_object_schema(
    {
        "anchor_key": NONEMPTY_STRING,
        "workload": NONEMPTY_STRING,
        "phase": NONEMPTY_STRING,
        "regime": NONEMPTY_STRING,
        "backend": NONEMPTY_STRING,
        "bottleneck": NONEMPTY_STRING,
        "primary_baseline_key": NONEMPTY_STRING,
        "target_metrics": STRING_ARRAY,
        "evidence_refs": STRING_ARRAY,
        "status": {"type": "string", "enum": ["candidate", "active", "needs_evidence", "rejected"]},
        "baselines": {"type": "array", "items": BASELINE_PROPOSAL_SCHEMA},
        "entries": {"type": "array", "items": ENTRY_PROPOSAL_SCHEMA},
        "edges": {"type": "array", "items": EDGE_PROPOSAL_SCHEMA},
        "gaps": STRING_ARRAY,
    }
)

EVIDENCE_REQUEST_SCHEMA = make_object_schema(
    {
        "target": STRING,
        "missing_claim": STRING,
        "query": STRING,
        "source_scope": STRING,
        "decision_impact": STRING,
    }
)

DISPOSITION_SCHEMA = make_object_schema(
    {
        "claim_id": NONEMPTY_STRING,
        "status": {
            "type": "string",
            "enum": ["integrated", "duplicate", "irrelevant_to_scope", "needs_evidence", "invalid"],
        },
        "reason": NONEMPTY_STRING,
    }
)

CURATION_SCHEMA = make_object_schema(
    {
        "action": {"type": "string", "enum": ["integrate", "request_evidence", "complete"]},
        "anchors": {"type": "array", "items": ANCHOR_PROPOSAL_SCHEMA},
        "evidence_requests": {"type": "array", "items": EVIDENCE_REQUEST_SCHEMA},
        "dispositions": {"type": "array", "items": DISPOSITION_SCHEMA},
        "unresolved_gaps": STRING_ARRAY,
    }
)

DIRECTION_PROPOSAL_SCHEMA = make_object_schema(
    {
        "title": NONEMPTY_STRING,
        "selected_entry_ids": STRING_ARRAY,
        "selected_edge_ids": STRING_ARRAY,
        "baseline_ids": STRING_ARRAY,
        "hypothesis": NONEMPTY_STRING,
        "expected_effects": STRING_ARRAY,
        "preconditions": STRING_ARRAY,
        "ablation_plan": STRING_ARRAY,
        "evidence_refs": STRING_ARRAY,
        "gaps": STRING_ARRAY,
        "kind": {
            "type": "string",
            "enum": ["experiment", "baseline_reference", "implementation_reference", "method_reference"],
        },
    }
)

DIRECTION_SCHEMA = make_object_schema(
    {
        "action": {"type": "string", "enum": ["integrate", "request_evidence", "complete"]},
        "directions": {"type": "array", "items": DIRECTION_PROPOSAL_SCHEMA},
        "evidence_requests": {"type": "array", "items": EVIDENCE_REQUEST_SCHEMA},
        "unresolved_gaps": STRING_ARRAY,
    }
)

REVIEW_SCHEMA = make_object_schema(
    {
        "exploration_value": {"type": "string", "enum": ["high", "middle", "low"]},
        "implementation_reuse": {"type": "string", "enum": ["high", "middle", "low"]},
        "method_reference": {"type": "string", "enum": ["high", "middle", "low"]},
        "baseline_quality": {"type": "string", "enum": ["high", "middle", "low", "not_applicable"]},
        "cross_layer_validity": {"type": "string", "enum": ["high", "middle", "low", "not_applicable"]},
        "experiment_readiness": {"type": "string", "enum": ["ready", "partial", "not_ready"]},
        "decision": {
            "type": "string",
            "enum": ["experiment_candidate", "needs_evidence", "baseline_reference", "rejected"],
        },
        "reasons": STRING_ARRAY,
        "falsifiable_hypothesis": STRING,
        "implementation_plan": STRING_ARRAY,
        "baseline_ablation_matrix": STRING_ARRAY,
        "metrics": STRING_ARRAY,
        "failure_conditions": STRING_ARRAY,
        "gaps": STRING_ARRAY,
        "evidence_refs": STRING_ARRAY,
        "entry_refs": STRING_ARRAY,
        "edge_refs": STRING_ARRAY,
        "alternative_entry_refs": STRING_ARRAY,
        "alternative_edge_refs": STRING_ARRAY,
        "baseline_refs": STRING_ARRAY,
    }
)

def nullable_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(schema)
    schema_type = result.get("type")
    if isinstance(schema_type, str):
        result["type"] = [schema_type, "null"]
    elif isinstance(schema_type, list) and "null" not in schema_type:
        result["type"] = [*schema_type, "null"]
    if "enum" in result and None not in result["enum"]:
        result["enum"] = [*result["enum"], None]
    return result


PROVISIONAL_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        key: nullable_schema(schema)
        for key, schema in REVIEW_SCHEMA["properties"].items()
    },
    "required": list(REVIEW_SCHEMA["properties"].keys()),
}

JUDGE_SCHEMA = make_object_schema(
    {
        "action": {"type": "string", "enum": ["ask", "request_evidence", "complete"]},
        "dimension": {
            "type": "string",
            "enum": [*REVIEW_DIMENSIONS, "complete"],
        },
        "question": STRING,
        "evidence_request": EVIDENCE_REQUEST_SCHEMA,
        "review": PROVISIONAL_REVIEW_SCHEMA,
    }
)

ANSWER_SCHEMA = make_object_schema(
    {
        "answer": NONEMPTY_STRING,
        "evidence_refs": STRING_ARRAY,
        "direct_facts": STRING_ARRAY,
        "inferences": STRING_ARRAY,
        "gaps": STRING_ARRAY,
    }
)

SCHEMAS: dict[str, dict[str, Any]] = {
    "discovery": DISCOVERY_SCHEMA,
    "curation": CURATION_SCHEMA,
    "direction": DIRECTION_SCHEMA,
    "judge": JUDGE_SCHEMA,
    "answer": ANSWER_SCHEMA,
}


def validate_schema(instance: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Validate the JSON Schema subset used by this workflow.

    Claude/Codex CLIs validate outputs too, but local validation is authoritative
    and provider-independent. The supported subset is deliberately small.
    """

    errors: list[str] = []
    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected]

    def matches_type(type_name: Any) -> bool:
        if type_name == "object":
            return isinstance(instance, dict)
        if type_name == "array":
            return isinstance(instance, list)
        if type_name == "string":
            return isinstance(instance, str)
        if type_name == "integer":
            return isinstance(instance, int) and not isinstance(instance, bool)
        if type_name == "number":
            return isinstance(instance, (int, float)) and not isinstance(instance, bool)
        if type_name == "boolean":
            return isinstance(instance, bool)
        if type_name == "null":
            return instance is None
        return type_name is None

    matching_types = [
        type_name for type_name in expected_types if matches_type(type_name)
    ]
    if expected and not matching_types:
        return [f"{path}: expected {expected}, got {type(instance).__name__}"]
    effective_type = matching_types[0] if matching_types else expected

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']!r}")

    if effective_type == "null":
        return errors

    if effective_type == "string":
        if len(instance) < int(schema.get("minLength", 0)):
            errors.append(f"{path}: string shorter than minLength={schema['minLength']}")
        pattern = schema.get("pattern")
        if pattern and not re.search(pattern, instance):
            errors.append(f"{path}: string does not match {pattern!r}")

    if effective_type in ("integer", "number"):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: value below minimum={schema['minimum']}")

    if effective_type == "array":
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: array shorter than minItems={schema['minItems']}")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(instance):
                errors.extend(validate_schema(item, item_schema, f"{path}[{index}]"))

    if effective_type == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"{path}: unexpected property {key!r}")
        for key, child_schema in properties.items():
            if key in instance:
                errors.extend(validate_schema(instance[key], child_schema, f"{path}.{key}"))

    return errors


def extract_first_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    starts = [index for index, char in enumerate(text) if char in "[{"]
    for start in starts:
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError("no valid JSON object found in provider output")


@dataclass
class ProviderResponse:
    data: dict[str, Any]
    provider: str
    model: str
    session_id: str
    telemetry: dict[str, Any] = field(default_factory=dict)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    raw_log_path: str = ""


class ProviderError(RuntimeError):
    pass


class StructuredOutputError(ProviderError):
    def __init__(self, message: str, *, session_id: str = "") -> None:
        super().__init__(message)
        self.session_id = session_id


class Provider:
    name = "provider"

    def call(
        self,
        *,
        role: str,
        prompt: str,
        schema_name: str,
        conversation_key: str,
        session_id: str = "",
    ) -> ProviderResponse:
        raise NotImplementedError


_ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()
_ACTIVE_PROCESSES_LOCK = threading.Lock()
_SIGNAL_HANDLERS_INSTALLED = False


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass


def _cleanup_active_processes(*_: Any) -> None:
    with _ACTIVE_PROCESSES_LOCK:
        active = list(_ACTIVE_PROCESSES)
    for proc in active:
        _terminate_process(proc)


def install_signal_cleanup() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED or threading.current_thread() is not threading.main_thread():
        return
    _SIGNAL_HANDLERS_INSTALLED = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(sig)

        def handler(signum: int, frame: Any, previous_handler: Any = previous) -> None:
            _cleanup_active_processes()
            if callable(previous_handler):
                previous_handler(signum, frame)
            raise KeyboardInterrupt

        signal.signal(sig, handler)


def run_process(
    command: Sequence[str],
    *,
    input_text: str,
    cwd: Path,
    timeout_seconds: int,
) -> tuple[int, str, str, float]:
    install_signal_cleanup()
    started = time.monotonic()
    proc = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    with _ACTIVE_PROCESSES_LOCK:
        _ACTIVE_PROCESSES.add(proc)
    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=timeout_seconds)
        return proc.returncode, stdout, stderr, time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        _terminate_process(proc)
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raise ProviderError(f"provider process timed out after {timeout_seconds}s") from exc
    finally:
        with _ACTIVE_PROCESSES_LOCK:
            _ACTIVE_PROCESSES.discard(proc)


def parse_deepseek_stream(stdout: str) -> tuple[Any, str, dict[str, Any], list[dict[str, Any]]]:
    assistant_text: list[str] = []
    terminal: dict[str, Any] = {}
    session_id = ""
    events: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        if isinstance(event.get("session_id"), str):
            session_id = event["session_id"]
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    assistant_text.append(block["text"])
        if event.get("type") == "result":
            terminal = event

    if terminal.get("is_error"):
        raise ProviderError(
            f"DeepSeek/Claude CLI returned error subtype={terminal.get('subtype', 'unknown')}: "
            f"{terminal.get('result') or terminal.get('errors') or 'unknown error'}"
        )

    candidates: list[Any] = [
        terminal.get("structured_output"),
        terminal.get("structuredOutput"),
        terminal.get("result"),
        "".join(assistant_text),
    ]
    data: Any = None
    last_error: Exception | None = None
    for candidate in candidates:
        if isinstance(candidate, dict):
            data = candidate
            break
        if isinstance(candidate, str) and candidate.strip():
            try:
                parsed = extract_first_json(candidate)
                if isinstance(parsed, dict):
                    data = parsed
                    break
            except Exception as exc:  # noqa: BLE001 - retain best parse error
                last_error = exc
    if not isinstance(data, dict):
        raise ProviderError(f"DeepSeek output contains no structured object: {last_error or 'empty result'}")
    telemetry = {
        key: terminal.get(key)
        for key in (
            "subtype", "stop_reason", "total_cost_usd", "usage",
            "modelUsage", "duration_ms", "duration_api_ms", "num_turns",
        )
        if key in terminal
    }
    return data, session_id, telemetry, events


def parse_codex_stream(stdout: str) -> tuple[Any, str, dict[str, Any], list[dict[str, Any]]]:
    session_id = ""
    final_messages: list[str] = []
    telemetry: dict[str, Any] = {}
    tool_events: list[dict[str, Any]] = []
    error_messages: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type", ""))
        if event_type in ("thread.started", "session.started"):
            session_id = str(event.get("thread_id") or event.get("session_id") or event.get("id") or "")
        if isinstance(event.get("thread_id"), str) and not session_id:
            session_id = event["thread_id"]
        if event_type in ("turn.failed", "error"):
            error_messages.append(str(event.get("error") or event.get("message") or event))
        if event_type == "turn.completed":
            telemetry = {
                "usage": event.get("usage"),
                "turn_id": event.get("turn_id"),
            }

        item = event.get("item")
        item_type = ""
        if isinstance(item, dict):
            item_type = str(item.get("type", ""))
            if item_type in ("agent_message", "assistant_message"):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    final_messages.append(text)
                elif isinstance(text, list):
                    for block in text:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            final_messages.append(block["text"])

        combined_type = f"{event_type}:{item_type}".lower()
        if any(
            marker in combined_type
            for marker in (
                "command_execution", "shell", "tool_call", "mcp", "web_search",
                "file_change", "computer", "apply_patch",
            )
        ):
            tool_events.append(event)

    if error_messages:
        raise ProviderError("Codex CLI error: " + "; ".join(error_messages[-3:]))
    if not final_messages:
        raise ProviderError("Codex JSONL contains no final agent message")
    data = extract_first_json(final_messages[-1])
    if not isinstance(data, dict):
        raise ProviderError("Codex final message is not a JSON object")
    return data, session_id, telemetry, tool_events


class DeepSeekCLIProvider(Provider):
    name = "deepseek-cli"

    def __init__(
        self,
        work_dir: Path,
        *,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        timeout_seconds: int = 900,
        max_budget_usd: float = 20.0,
        allow_repair: bool = True,
    ) -> None:
        self.work_dir = work_dir
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_budget_usd = max_budget_usd
        self.allow_repair = allow_repair
        self._call_lock = threading.Lock()
        self._call_counter = 0

    def _next_call_dir(self, role: str, conversation_key: str) -> Path:
        with self._call_lock:
            self._call_counter += 1
            counter = self._call_counter
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", conversation_key)[:100]
        path = self.work_dir / "logs" / "model_calls" / f"{counter:05d}_{self.name}_{role}_{safe_key}"
        ensure_dir(path)
        return path

    def _invoke(
        self,
        *,
        role: str,
        prompt: str,
        schema_name: str,
        conversation_key: str,
        session_id: str,
    ) -> ProviderResponse:
        schema = SCHEMAS[schema_name]
        actual_session_id = session_id or str(uuid.uuid4())
        command = [
            "claude",
            "-p",
            "--model", self.model,
            "--output-format", "stream-json",
            "--verbose",
            "--json-schema", canonical_json(schema),
            "--tools", "",
            "--permission-mode", "bypassPermissions",
            "--add-dir", str(VAULT_ROOT),
            "--max-budget-usd", str(self.max_budget_usd),
            "--prompt-suggestions", "false",
        ]
        if session_id:
            command.extend(["--resume", session_id])
        else:
            command.extend(["--session-id", actual_session_id])

        call_dir = self._next_call_dir(role, conversation_key)
        metadata = {
            "provider": self.name,
            "model": self.model,
            "role": role,
            "conversation_key": conversation_key,
            "session_id": actual_session_id,
            "schema_name": schema_name,
            "schema_hash": sha256_text(canonical_json(schema)),
            "prompt_hash": sha256_text(prompt),
            "started_at": utc_now(),
        }
        atomic_write_json(call_dir / "request.json", metadata)
        atomic_write_text(call_dir / "prompt.md", prompt)
        return_code, stdout, stderr, elapsed = run_process(
            command,
            input_text=prompt,
            cwd=VAULT_ROOT,
            timeout_seconds=self.timeout_seconds,
        )
        atomic_write_text(call_dir / "raw.jsonl", stdout)
        atomic_write_text(call_dir / "stderr.log", stderr)
        metadata.update({"return_code": return_code, "elapsed_seconds": elapsed, "finished_at": utc_now()})
        atomic_write_json(call_dir / "result_meta.json", metadata)
        if return_code != 0:
            raise ProviderError(f"DeepSeek/Claude CLI exited {return_code}; see {call_dir / 'stderr.log'}")

        data, parsed_session_id, telemetry, _events = parse_deepseek_stream(stdout)
        errors = validate_schema(data, schema)
        if errors:
            raise StructuredOutputError(
                "structured output validation failed: " + "; ".join(errors[:20]),
                session_id=parsed_session_id or actual_session_id,
            )
        atomic_write_json(call_dir / "parsed.json", data)
        return ProviderResponse(
            data=data,
            provider=self.name,
            model=self.model,
            session_id=parsed_session_id or actual_session_id,
            telemetry=telemetry,
            raw_log_path=str(call_dir / "raw.jsonl"),
        )

    def call(
        self,
        *,
        role: str,
        prompt: str,
        schema_name: str,
        conversation_key: str,
        session_id: str = "",
    ) -> ProviderResponse:
        try:
            return self._invoke(
                role=role,
                prompt=prompt,
                schema_name=schema_name,
                conversation_key=conversation_key,
                session_id=session_id,
            )
        except StructuredOutputError as first_error:
            repair_session_id = first_error.session_id or session_id
            if not self.allow_repair or not repair_session_id:
                raise
            repair = (
                "[STRUCTURE_REPAIR]\n"
                f"The previous turn failed local validation: {first_error}.\n"
                f"Return exactly one object matching schema `{schema_name}`. "
                "Do not call tools, retrieve evidence, reset state, or revise substantive conclusions. "
                "Repair structure only."
            )
            return self._invoke(
                role=role,
                prompt=repair,
                schema_name=schema_name,
                conversation_key=conversation_key + ":repair",
                session_id=repair_session_id,
            )


class CodexCLIProvider(Provider):
    name = "codex-cli"

    def __init__(
        self,
        work_dir: Path,
        *,
        model: str = "",
        timeout_seconds: int = 900,
        allow_repair: bool = True,
        reject_tool_events: bool = True,
    ) -> None:
        self.work_dir = work_dir
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.allow_repair = allow_repair
        self.reject_tool_events = reject_tool_events
        self._call_lock = threading.Lock()
        self._call_counter = 0

    def _next_call_dir(self, role: str, conversation_key: str) -> Path:
        with self._call_lock:
            self._call_counter += 1
            counter = self._call_counter
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", conversation_key)[:100]
        path = self.work_dir / "logs" / "model_calls" / f"{counter:05d}_{self.name}_{role}_{safe_key}"
        ensure_dir(path)
        return path

    def _invoke(
        self,
        *,
        role: str,
        prompt: str,
        schema_name: str,
        conversation_key: str,
        session_id: str,
    ) -> ProviderResponse:
        schema = SCHEMAS[schema_name]
        call_dir = self._next_call_dir(role, conversation_key)
        schema_path = call_dir / "schema.json"
        atomic_write_json(schema_path, schema)
        sandbox_dir = self.work_dir / "provider_sandbox" / re.sub(
            r"[^A-Za-z0-9_.-]+", "_", conversation_key
        )[:100]
        ensure_dir(sandbox_dir)

        if session_id:
            command = [
                "codex",
                "--ask-for-approval", "never",
                "--sandbox", "read-only",
                "--cd", str(sandbox_dir),
                "exec", "resume", session_id,
                "--json",
                "--output-schema", str(schema_path),
                "--skip-git-repo-check",
            ]
        else:
            command = [
                "codex",
                "--ask-for-approval", "never",
                "--sandbox", "read-only",
                "--cd", str(sandbox_dir),
                "exec",
                "--json",
                "--output-schema", str(schema_path),
                "--skip-git-repo-check",
            ]
        if self.model:
            command.extend(["--model", self.model])
        command.append("-")

        isolated_prompt = (
            "Tool isolation rule: do not call shell, file, web, MCP, computer, or any other tool. "
            "Use only the evidence and canonical state present in this prompt. "
            "The outer orchestrator will reject the turn if any tool event is emitted.\n\n"
            + prompt
        )
        metadata = {
            "provider": self.name,
            "model": self.model or "<config-default>",
            "role": role,
            "conversation_key": conversation_key,
            "session_id": session_id,
            "schema_name": schema_name,
            "schema_hash": sha256_text(canonical_json(schema)),
            "prompt_hash": sha256_text(isolated_prompt),
            "started_at": utc_now(),
        }
        atomic_write_json(call_dir / "request.json", metadata)
        atomic_write_text(call_dir / "prompt.md", isolated_prompt)
        return_code, stdout, stderr, elapsed = run_process(
            command,
            input_text=isolated_prompt,
            cwd=sandbox_dir,
            timeout_seconds=self.timeout_seconds,
        )
        atomic_write_text(call_dir / "raw.jsonl", stdout)
        atomic_write_text(call_dir / "stderr.log", stderr)
        metadata.update({"return_code": return_code, "elapsed_seconds": elapsed, "finished_at": utc_now()})
        atomic_write_json(call_dir / "result_meta.json", metadata)
        if return_code != 0:
            raise ProviderError(f"Codex CLI exited {return_code}; see {call_dir / 'stderr.log'}")

        data, parsed_session_id, telemetry, tool_events = parse_codex_stream(stdout)
        if self.reject_tool_events and tool_events:
            raise ProviderError(f"Codex turn emitted {len(tool_events)} prohibited tool event(s)")
        errors = validate_schema(data, schema)
        if errors:
            raise StructuredOutputError(
                "structured output validation failed: " + "; ".join(errors[:20]),
                session_id=parsed_session_id or session_id,
            )
        atomic_write_json(call_dir / "parsed.json", data)
        return ProviderResponse(
            data=data,
            provider=self.name,
            model=self.model or "<config-default>",
            session_id=parsed_session_id or session_id,
            telemetry=telemetry,
            tool_events=tool_events,
            raw_log_path=str(call_dir / "raw.jsonl"),
        )

    def call(
        self,
        *,
        role: str,
        prompt: str,
        schema_name: str,
        conversation_key: str,
        session_id: str = "",
    ) -> ProviderResponse:
        try:
            return self._invoke(
                role=role,
                prompt=prompt,
                schema_name=schema_name,
                conversation_key=conversation_key,
                session_id=session_id,
            )
        except StructuredOutputError as first_error:
            repair_session_id = first_error.session_id or session_id
            if not self.allow_repair or not repair_session_id:
                raise
            repair = (
                "[STRUCTURE_REPAIR]\n"
                f"The previous turn failed local validation: {first_error}.\n"
                f"Return exactly one object matching schema `{schema_name}`. "
                "Do not call tools or revise substantive conclusions. Repair structure only."
            )
            return self._invoke(
                role=role,
                prompt=repair,
                schema_name=schema_name,
                conversation_key=conversation_key + ":repair",
                session_id=repair_session_id,
            )


class FixtureProvider(Provider):
    name = "fixture"

    def __init__(self, fixture_path: Path) -> None:
        raw = read_json(fixture_path)
        if not isinstance(raw, list):
            raise ValueError("fixture file must contain a JSON array")
        self.steps = list(raw)
        self.index = 0
        self.lock = threading.Lock()

    def call(
        self,
        *,
        role: str,
        prompt: str,
        schema_name: str,
        conversation_key: str,
        session_id: str = "",
    ) -> ProviderResponse:
        del prompt
        with self.lock:
            if self.index >= len(self.steps):
                raise ProviderError(f"fixture exhausted at role={role}, schema={schema_name}")
            step = self.steps[self.index]
            self.index += 1
        expected_schema = step.get("schema_name")
        if expected_schema and expected_schema != schema_name:
            raise ProviderError(
                f"fixture step {self.index} expected schema={expected_schema}, got {schema_name}"
            )
        data = step.get("data")
        errors = validate_schema(data, SCHEMAS[schema_name])
        if errors:
            raise ProviderError("invalid fixture response: " + "; ".join(errors[:20]))
        return ProviderResponse(
            data=data,
            provider=self.name,
            model="synthetic-fixture",
            session_id=session_id or f"fixture-{normalize_key(conversation_key)}",
            telemetry={"synthetic": True, "step": self.index},
        )


def provider_from_config(
    provider_name: str,
    work_dir: Path,
    config: Mapping[str, Any],
) -> Provider:
    if provider_name == "deepseek-cli":
        return DeepSeekCLIProvider(
            work_dir,
            model=str(config.get("deepseek_model") or DEFAULT_DEEPSEEK_MODEL),
            timeout_seconds=int(config.get("provider_timeout_seconds", 900)),
            max_budget_usd=float(config.get("max_budget_usd_per_call", 20.0)),
        )
    if provider_name == "codex-cli":
        return CodexCLIProvider(
            work_dir,
            model=str(config.get("codex_model") or ""),
            timeout_seconds=int(config.get("provider_timeout_seconds", 900)),
        )
    if provider_name == "fixture":
        fixture_path = config.get("fixture_file")
        if not fixture_path:
            raise ValueError("fixture provider requires fixture_file")
        return FixtureProvider(Path(str(fixture_path)))
    raise ValueError(f"unknown provider: {provider_name}")


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.:+/-]{1,}|[\u4e00-\u9fff]{2,}", query):
        cleaned = token.strip("._:+/-")
        if len(cleaned) >= 2 and cleaned.lower() not in {"the", "and", "for", "with", "from", "this"}:
            terms.append(cleaned)
    unique: list[str] = []
    seen: set[str] = set()
    for term in sorted(terms, key=lambda item: (-len(item), item.lower())):
        key = term.lower()
        if key not in seen:
            unique.append(term)
            seen.add(key)
    return unique[:10]


@dataclass
class EvidenceSnippet:
    snippet_id: str
    source_path: str
    line_start: int
    line_end: int
    score: float
    matched_terms: list[str]
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "snippet_id": self.snippet_id,
            "source_path": self.source_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "score": self.score,
            "matched_terms": self.matched_terms,
            "text": self.text,
        }


class LocalMarkdownRetriever:
    """Bounded lexical retrieval over selected local Markdown roots.

    `rg` finds candidate lines; Python reads only bounded context around hits.
    The model receives line-numbered snippets but cannot access the filesystem.
    """

    def __init__(self, vault_root: Path, roots: Sequence[str | Path]) -> None:
        self.vault_root = vault_root.resolve()
        resolved: list[Path] = []
        for root in roots:
            candidate = Path(root)
            if not candidate.is_absolute():
                candidate = self.vault_root / candidate
            candidate = candidate.resolve()
            if candidate.exists() and candidate.is_dir():
                resolved.append(candidate)
        self.roots = resolved

    def _display_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.vault_root).as_posix()
        except ValueError:
            return str(path.resolve())

    def search(
        self,
        query: str,
        *,
        limit: int = 12,
        context_lines: int = 8,
        exclude_snippets: Iterable[str] = (),
    ) -> list[EvidenceSnippet]:
        if not self.roots:
            return []
        rg = shutil.which("rg")
        if not rg:
            raise RuntimeError("rg is required for local Markdown retrieval")
        terms = _query_terms(query)
        if not terms:
            return []
        excluded = set(exclude_snippets)
        hits: dict[tuple[Path, int], set[str]] = {}
        for term in terms:
            command = [
                rg,
                "--no-heading",
                "--color", "never",
                "-n",
                "-i",
                "-F",
                "--glob", "*.md",
                "--max-count", "25",
                "--max-filesize", "2M",
                "--sort", "path",
                "--",
                term,
                *[str(root) for root in self.roots],
            ]
            proc = subprocess.Popen(
                command,
                cwd=str(self.vault_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert proc.stdout is not None
            hit_limit = max(limit * 12, 96)
            hit_count = 0
            terminated_early = False
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                match = re.match(r"^(.*?):(\d+):(.*)$", line)
                if not match:
                    continue
                path = Path(match.group(1)).resolve()
                line_number = int(match.group(2))
                hits.setdefault((path, line_number), set()).add(term)
                hit_count += 1
                if hit_count >= hit_limit:
                    terminated_early = True
                    proc.terminate()
                    break
            try:
                _remaining_stdout, stderr = proc.communicate(timeout=60)
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                proc.communicate()
                raise RuntimeError(f"rg retrieval timed out for {term!r}") from exc
            if not terminated_early and proc.returncode not in (0, 1):
                raise RuntimeError(f"rg retrieval failed for {term!r}: {stderr.strip()}")

        grouped: dict[Path, list[tuple[int, set[str]]]] = {}
        for (path, line_number), matched_terms in hits.items():
            grouped.setdefault(path, []).append((line_number, matched_terms))

        snippets: list[EvidenceSnippet] = []
        for path, path_hits in grouped.items():
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            path_hits.sort(key=lambda item: item[0])
            windows: list[tuple[int, int, set[str]]] = []
            for line_number, matched_terms in path_hits:
                start = max(1, line_number - context_lines)
                end = min(len(lines), line_number + context_lines)
                if windows and start <= windows[-1][1] + 2:
                    old_start, old_end, old_terms = windows[-1]
                    windows[-1] = (old_start, max(old_end, end), old_terms | matched_terms)
                else:
                    windows.append((start, end, set(matched_terms)))
            for start, end, matched_terms in windows[:4]:
                display_path = self._display_path(path)
                snippet_id = stable_id("S", display_path, start, end)
                if snippet_id in excluded:
                    continue
                numbered = "\n".join(
                    f"{index}:{lines[index - 1]}" for index in range(start, end + 1)
                )
                query_coverage = len(matched_terms) / max(len(terms), 1)
                score = round(len(matched_terms) * 10 + query_coverage, 4)
                snippets.append(
                    EvidenceSnippet(
                        snippet_id=snippet_id,
                        source_path=display_path,
                        line_start=start,
                        line_end=end,
                        score=score,
                        matched_terms=sorted(matched_terms),
                        text=numbered,
                    )
                )
        snippets.sort(key=lambda item: (-item.score, item.source_path, item.line_start))
        return snippets[:limit]


def resolve_source_path(source_path: str, vault_root: Path = VAULT_ROOT) -> Path:
    candidate = Path(source_path)
    if not candidate.is_absolute():
        candidate = vault_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(vault_root.resolve())
    except ValueError as exc:
        raise ValueError(f"source outside vault root: {source_path}") from exc
    return resolved


def candidate_to_claim(
    candidate: Mapping[str, Any],
    *,
    vault_root: Path = VAULT_ROOT,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = validate_schema(candidate, DISCOVERY_CANDIDATE_SCHEMA)
    if errors:
        return None, errors
    source_path = str(candidate["source_path"])
    try:
        path = resolve_source_path(source_path, vault_root)
    except ValueError as exc:
        return None, [str(exc)]
    if not path.exists() or not path.is_file():
        return None, [f"source does not exist: {source_path}"]
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    line_start = int(candidate["line_start"])
    line_end = int(candidate["line_end"])
    if line_end < line_start:
        return None, ["line_end is before line_start"]
    if line_end > len(lines):
        return None, [f"line_end {line_end} exceeds source length {len(lines)}"]
    source_excerpt = "\n".join(lines[line_start - 1:line_end])
    quote = str(candidate["quote"])
    if normalize_text(quote) not in normalize_text(source_excerpt):
        return None, ["quote does not match the declared source line range"]
    display_path = path.relative_to(vault_root.resolve()).as_posix()
    claim_id = stable_id(
        "C",
        normalize_text(str(candidate["statement"])),
        display_path,
        line_start,
        line_end,
        normalize_text(quote),
    )
    claim = {
        "claim_id": claim_id,
        "statement": normalize_text(str(candidate["statement"])),
        "claim_type": candidate["claim_type"],
        "layer": candidate["layer"],
        "entity_name": normalize_text(str(candidate["entity_name"])),
        "source_path": display_path,
        "line_start": line_start,
        "line_end": line_end,
        "quote": quote.strip(),
        "evidence_mode": candidate["evidence_mode"],
        "scope": normalize_text(str(candidate["scope"])),
        "confidence": candidate["confidence"],
        "created_at": utc_now(),
    }
    return claim, []


def merge_by_id(
    existing: list[dict[str, Any]],
    additions: Iterable[dict[str, Any]],
    id_field: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    index = {item[id_field]: item for item in existing}
    conflicts: list[str] = []
    for addition in additions:
        identifier = addition[id_field]
        previous = index.get(identifier)
        if previous is None:
            index[identifier] = addition
            continue
        comparable_previous = {key: value for key, value in previous.items() if key not in ("updated_at",)}
        comparable_addition = {key: value for key, value in addition.items() if key not in ("updated_at",)}
        if comparable_previous != comparable_addition:
            conflicts.append(f"{id_field}={identifier} has conflicting content")
    return sorted(index.values(), key=lambda item: item[id_field]), conflicts


def markdown_cell(value: Any) -> str:
    if isinstance(value, list):
        text = "; ".join(str(item) for item in value)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def build_seed_query(topic: str, layer: str, axis: str, constraints: str = "") -> str:
    return " ".join(
        [
            topic,
            *LAYER_TERMS[layer],
            *AXIS_TERMS[axis],
            "baseline current practice strong comparison implementation evaluation",
            "基线 当前实现 强基线 实验 指标",
            constraints,
        ]
    ).strip()


def format_snippets(snippets: Sequence[EvidenceSnippet], max_chars: int = 80_000) -> str:
    del max_chars  # Evidence packets are bounded by snippet count, never mid-object truncation.
    blocks: list[str] = []
    for snippet in snippets:
        block = (
            f"--- snippet_id={snippet.snippet_id}\n"
            f"source_path={snippet.source_path}\n"
            f"line_range={snippet.line_start}-{snippet.line_end}\n"
            f"matched_terms={', '.join(snippet.matched_terms)}\n"
            f"{snippet.text}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


def claim_index(claims: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(claim["claim_id"]): claim for claim in claims}


def validate_references(refs: Sequence[str], valid_ids: set[str], context: str) -> list[str]:
    return [f"{context}: missing reference {ref}" for ref in refs if ref not in valid_ids]
