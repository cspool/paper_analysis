import type {
  GateActual,
  GateDefinitionDraft,
  GateExpected,
  GateValueType,
  MechanicalGateCheck,
  RegisteredTurnInputMessageType,
  RegisteredTurnOutputMessageType,
  StageContractDraft,
  ValidationReport,
} from "../contracts/index.ts";
import {
  ROLE_MESSAGE_TYPES,
  STAGE_REGISTRY,
  canonicalEqual,
} from "../contracts/index.ts";
import {
  SCHEMA_DEFINITIONS,
  type JsonSchema,
  type SchemaName,
} from "../schemas/schema_definitions.ts";
import {
  addError,
  emptyReport,
} from "../validators/schema_validator.ts";

export const GATE_COMPILER_POLICY_VERSION =
  "simple-semantic-loop-gate-compiler/1";

export interface GateCompilation {
  report: ValidationReport;
  compiled: GateDefinitionDraft;
}

const RUNTIME_FACT_TYPES = Object.freeze({
  allowed_tool_events_only: "boolean",
  allowed_paths_only: "boolean",
  turn_budget_within_contract: "boolean",
  experiment_execution_count: "number",
  external_evidence_used: "boolean",
} as const satisfies Record<string, GateValueType>);

const VALIDATOR_FACT_TYPES = Object.freeze({
  schema_valid: "boolean",
  message_binding_matches: "boolean",
  registered_validator_passes: "boolean",
  references_resolve: "boolean",
  source_context_present: "boolean",
  duplicate_commit: "boolean",
  script_transition_valid: "boolean",
} as const satisfies Record<string, GateValueType>);

/**
 * Compile an untrusted Workflow Agent Gate proposal into the effective frozen
 * Gate. Controller-owned checks are always injected and cannot be shadowed.
 */
export function compileGateDraft(
  stage: StageContractDraft,
  proposed: GateDefinitionDraft,
): GateCompilation {
  const report = emptyReport();
  const registered = STAGE_REGISTRY[stage.stageType];
  const proposedIds = new Set<string>();

  if (proposed.mechanicalChecks.length > 24) {
    addError(
      report,
      "gate.check_count_exceeded",
      "/proposedGateDefinition/mechanicalChecks",
      "Workflow Agent may propose at most 24 Stage-specific Gate checks",
    );
  }
  validateSemanticEvaluation(proposed, report);

  proposed.mechanicalChecks.forEach((check, index) => {
    const at = `/proposedGateDefinition/mechanicalChecks/${index}`;
    if (
      check === null ||
      typeof check !== "object" ||
      check.actual === null ||
      typeof check.actual !== "object" ||
      typeof check.checkId !== "string"
    ) {
      addError(
        report,
        "gate.invalid_check_shape",
        at,
        "Gate check must contain a typed actual operand and string checkId",
      );
      return;
    }
    if (check.checkId.startsWith("controller.")) {
      addError(
        report,
        "gate.reserved_check_id",
        `${at}/checkId`,
        "checkId prefix controller. is reserved for deterministic checks",
      );
    }
    if (check.checkId.length > 96) {
      addError(
        report,
        "gate.check_id_too_long",
        `${at}/checkId`,
        "checkId must be at most 96 characters",
      );
    }
    if (proposedIds.has(check.checkId)) {
      addError(
        report,
        "gate.duplicate_check_id",
        `${at}/checkId`,
        `duplicate proposed Gate check ${check.checkId}`,
      );
    }
    proposedIds.add(check.checkId);
    if (
      check.actual.source === "canonical" &&
      !stage.scope.some(
        (ref) =>
          ref.objectType === check.actual.objectRef.objectType &&
          ref.objectId === check.actual.objectRef.objectId &&
          ref.revision === check.actual.objectRef.revision,
      )
    ) {
      addError(
        report,
        "gate.canonical_ref_outside_stage_scope",
        `${at}/actual/objectRef`,
        "canonical Gate operand must exactly match a Stage scope ref",
      );
    }
    if (
      check.actual.source === "artifact" &&
      !stage.requiredInputs.some(
        (artifact) =>
          artifact.artifactId === check.actual.artifactId,
      )
    ) {
      addError(
        report,
        "gate.artifact_outside_required_inputs",
        `${at}/actual/artifactId`,
        "artifact Gate operand must be a frozen Stage required input",
      );
    }
    validateCheck(
      check,
      at,
      registered.output,
      registered.role
        ? ROLE_MESSAGE_TYPES[registered.role].input
        : null,
      report,
    );
  });

  const mandatory = controllerMandatoryChecks(stage);
  const mandatoryIds = new Set(mandatory.map((check) => check.checkId));
  for (const id of proposedIds) {
    if (mandatoryIds.has(id)) {
      addError(
        report,
        "gate.controller_check_shadowed",
        "/proposedGateDefinition/mechanicalChecks",
        `Workflow proposal cannot shadow mandatory check ${id}`,
      );
    }
  }
  proposed.mechanicalChecks.forEach((check, index) => {
    for (const controllerCheck of mandatory) {
      if (
        check.predicate === "equals" &&
        controllerCheck.predicate === "equals" &&
        canonicalEqual(check.actual, controllerCheck.actual) &&
        !canonicalEqual(check.expected, controllerCheck.expected)
      ) {
        addError(
          report,
          "gate.contradicts_controller_mandatory_check",
          `/proposedGateDefinition/mechanicalChecks/${index}`,
          `criterion conflicts with mandatory check ${controllerCheck.checkId}`,
        );
      }
    }
  });

  return {
    report,
    compiled: {
      ...structuredClone(proposed),
      mechanicalChecks: [
        ...mandatory,
        ...structuredClone(proposed.mechanicalChecks),
      ],
    },
  };
}

function validateSemanticEvaluation(
  proposed: GateDefinitionDraft,
  report: ValidationReport,
): void {
  const semantic = proposed.semanticEvaluation;
  if (semantic.required) {
    addError(
      report,
      "gate.inline_semantic_evaluation_unsupported",
      "/proposedGateDefinition/semanticEvaluation/required",
      "Semantic evaluation must be scheduled as a separate registered evaluator Stage",
    );
    return;
  }
  if (
    semantic.evaluatorRole !== null ||
    semantic.rubricId !== null ||
    semantic.inputProjection.length !== 0 ||
    semantic.expectedOutputMessageType !== null
  ) {
    addError(
      report,
      "gate.disabled_semantic_evaluation_not_empty",
      "/proposedGateDefinition/semanticEvaluation",
      "disabled semanticEvaluation must have null role/rubric/output and an empty projection",
    );
  }
}

export function controllerMandatoryChecks(
  stage: StageContractDraft,
): MechanicalGateCheck[] {
  const checks: MechanicalGateCheck[] = [];
  if (stage.executionKind === "SCRIPT_TRANSITION") {
    checks.push(
      equalsCheck(
        "controller.script_transition_valid",
        validatorActual("script_transition_valid"),
        true,
      ),
    );
  } else {
    checks.push(
      equalsCheck(
        "controller.schema_valid",
        validatorActual("schema_valid"),
        true,
      ),
      equalsCheck(
        "controller.message_binding_matches",
        validatorActual("message_binding_matches"),
        true,
      ),
      equalsCheck(
        "controller.registered_validator_passes",
        validatorActual("registered_validator_passes"),
        true,
      ),
      equalsCheck(
        "controller.allowed_tool_events_only",
        runtimeActual("allowed_tool_events_only", "boolean"),
        true,
      ),
      equalsCheck(
        "controller.allowed_paths_only",
        runtimeActual("allowed_paths_only", "boolean"),
        true,
      ),
      equalsCheck(
        "controller.turn_budget_within_contract",
        runtimeActual("turn_budget_within_contract", "boolean"),
        true,
      ),
      equalsCheck(
        "controller.no_experiment_execution",
        runtimeActual("experiment_execution_count", "number"),
        0,
      ),
      equalsCheck(
        "controller.no_duplicate_commit",
        validatorActual("duplicate_commit"),
        false,
      ),
    );
  }

  if (stage.stageType === "EVIDENCE_READ") {
    checks.push(
      equalsCheck(
        "controller.evidence_references_resolve",
        {
          source: "validator",
          fact: "references_resolve",
          pointer: "/payload/findings",
          valueType: "boolean",
        },
        true,
      ),
      equalsCheck(
        "controller.evidence_source_context_present",
        {
          source: "validator",
          fact: "source_context_present",
          pointer: "/payload/findings",
          valueType: "boolean",
        },
        true,
      ),
      equalsCheck(
        "controller.local_evidence_only",
        runtimeActual("external_evidence_used", "boolean"),
        false,
      ),
    );
  }

  stage.requiredInputs.forEach((artifact, index) => {
    checks.push(
      equalsCheck(
        `controller.required_artifact_${index}_exists`,
        {
          source: "artifact",
          artifactId: artifact.artifactId,
          fact: "exists",
          valueType: "boolean",
        },
        true,
      ),
      equalsCheck(
        `controller.required_artifact_${index}_hash`,
        {
          source: "artifact",
          artifactId: artifact.artifactId,
          fact: "sha256",
          valueType: "string",
        },
        artifact.sha256,
      ),
    );
  });
  return checks;
}

function validateCheck(
  check: MechanicalGateCheck,
  at: string,
  outputMessageType: RegisteredTurnOutputMessageType | null,
  inputMessageType: RegisteredTurnInputMessageType | null,
  report: ValidationReport,
): void {
  if (
    "pointer" in check.actual &&
    check.actual.pointer !== null &&
    check.actual.pointer.length > 512
  ) {
    addError(
      report,
      "gate.pointer_too_long",
      `${at}/actual/pointer`,
      "Gate JSON Pointer must be at most 512 characters",
    );
  }
  if (
    typeof check.expected === "string" &&
    check.expected.length > 4_096
  ) {
    addError(
      report,
      "gate.expected_string_too_long",
      `${at}/expected`,
      "Gate expected string must be at most 4096 characters",
    );
  }
  if (!expectedMatchesValueType(check.expected, check.actual.valueType)) {
    addError(
      report,
      "gate.expected_type_mismatch",
      `${at}/expected`,
      `expected value does not match declared ${check.actual.valueType}`,
    );
  }
  if (check.predicate === "contains_fields") {
    if (
      check.actual.valueType !== "object" ||
      !Array.isArray(check.expected)
    ) {
      addError(
        report,
        "gate.contains_fields_type",
        at,
        "contains_fields requires object actual and string-array expected",
      );
    }
    if (
      Array.isArray(check.expected) &&
      (check.expected.length === 0 ||
        new Set(check.expected).size !== check.expected.length)
    ) {
      addError(
        report,
        "gate.contains_fields_set",
        `${at}/expected`,
        "contains_fields requires a non-empty unique field list",
      );
    }
  } else if (check.actual.valueType === "object") {
    addError(
      report,
      "gate.object_equality_not_supported",
      `${at}/predicate`,
      "object operands require contains_fields; whole-object equality is not admitted",
    );
  }

  switch (check.actual.source) {
    case "result":
      if (
        check.actual.pointer !== "/payload" &&
        !check.actual.pointer.startsWith("/payload/")
      ) {
        addError(
          report,
          "gate.result_pointer_not_stable",
          `${at}/actual/pointer`,
          "result Gate pointers must be /payload or /payload/...; identity and binding metadata such as /attemptId are forbidden",
        );
      }
      validateSchemaOperand(
        check,
        at,
        outputMessageType,
        check.actual.pointer,
        report,
      );
      break;
    case "task":
      if (!isStableTaskPointer(check.actual.pointer)) {
        addError(
          report,
          "gate.task_pointer_not_stable",
          `${at}/actual/pointer`,
          "task Gate operands must address immutable domain input and cannot inspect correction or dispatch metadata",
        );
      }
      validateSchemaOperand(
        check,
        at,
        inputMessageType,
        check.actual.pointer,
        report,
      );
      break;
    case "canonical":
      if (
        check.actual.pointer !== "/revision" ||
        check.actual.valueType !== "number"
      ) {
        addError(
          report,
          "gate.canonical_operand_not_registered",
          `${at}/actual`,
          "canonical operands currently admit only numeric /revision",
        );
      }
      break;
    case "runtime": {
      const expectedType = RUNTIME_FACT_TYPES[check.actual.fact];
      if (check.actual.valueType !== expectedType) {
        addError(
          report,
          "gate.runtime_fact_type",
          `${at}/actual/valueType`,
          `${check.actual.fact} requires valueType ${expectedType}`,
        );
      }
      break;
    }
    case "validator": {
      const expectedType = VALIDATOR_FACT_TYPES[check.actual.fact];
      if (check.actual.valueType !== expectedType) {
        addError(
          report,
          "gate.validator_fact_type",
          `${at}/actual/valueType`,
          `${check.actual.fact} requires valueType ${expectedType}`,
        );
      }
      const pointerRequired = [
        "references_resolve",
        "source_context_present",
      ].includes(check.actual.fact);
      if (pointerRequired !== (check.actual.pointer !== null)) {
        addError(
          report,
          "gate.validator_pointer_contract",
          `${at}/actual/pointer`,
          pointerRequired
            ? `${check.actual.fact} requires a schema-valid result pointer such as /payload/findings; pointer:null is invalid for this fact`
            : `${check.actual.fact} requires pointer:null; only references_resolve and source_context_present accept a result pointer`,
        );
      }
      if (
        check.actual.pointer !== null &&
        outputMessageType !== null
      ) {
        validateSchemaPointer(
          outputMessageType,
          check.actual.pointer,
          `${at}/actual/pointer`,
          report,
        );
      } else if (
        check.actual.pointer !== null &&
        outputMessageType === null
      ) {
        addError(
          report,
          "gate.result_schema_unavailable",
          `${at}/actual/pointer`,
          "script transition has no result schema",
        );
      }
      if (
        outputMessageType === null &&
        check.actual.fact !== "script_transition_valid"
      ) {
        addError(
          report,
          "gate.validator_fact_unavailable_for_script",
          `${at}/actual/fact`,
          "script transitions admit only script_transition_valid validator fact",
        );
      }
      break;
    }
    case "artifact":
      if (
        (check.actual.fact === "exists" &&
          check.actual.valueType !== "boolean") ||
        (check.actual.fact === "sha256" &&
          check.actual.valueType !== "string")
      ) {
        addError(
          report,
          "gate.artifact_fact_type",
          `${at}/actual/valueType`,
          `${check.actual.fact} has an incompatible valueType`,
        );
      }
      break;
  }
}

function isStableTaskPointer(pointer: string): boolean {
  if (!pointer.startsWith("/payload/")) return false;
  return ![
    "/payload/correctionFeedback",
    "/payload/skill",
    "/payload/schema",
    "/payload/permission",
    "/payload/terminationCondition",
  ].some(
    (prefix) =>
      pointer === prefix || pointer.startsWith(`${prefix}/`),
  );
}

function validateSchemaOperand(
  check: MechanicalGateCheck,
  at: string,
  messageType:
    | RegisteredTurnInputMessageType
    | RegisteredTurnOutputMessageType
    | null,
  pointer: string,
  report: ValidationReport,
): void {
  if (messageType === null) {
    addError(
      report,
      "gate.schema_operand_unavailable",
      `${at}/actual`,
      `${check.actual.source} operand is unavailable for this Stage`,
    );
    return;
  }
  const resolved = resolveSchemaPointer(messageType, pointer);
  if (!resolved) {
    addError(
      report,
      "gate.pointer_not_found",
      `${at}/actual/pointer`,
      `${pointer} does not resolve against ${messageType}`,
    );
    return;
  }
  if (!schemaSupportsValueType(resolved, check.actual.valueType)) {
    addError(
      report,
      "gate.pointer_type_mismatch",
      `${at}/actual/valueType`,
      `${pointer} does not have declared type ${check.actual.valueType}`,
    );
  }
  if (
    check.predicate === "contains_fields" &&
    Array.isArray(check.expected)
  ) {
    const properties = collectProperties(resolved);
    for (const field of check.expected) {
      if (!properties.has(field)) {
        addError(
          report,
          "gate.required_field_not_in_schema",
          `${at}/expected`,
          `${field} is not a field of ${pointer}`,
        );
      }
    }
  }
}

function validateSchemaPointer(
  messageType:
    | RegisteredTurnInputMessageType
    | RegisteredTurnOutputMessageType,
  pointer: string,
  at: string,
  report: ValidationReport,
): void {
  if (!resolveSchemaPointer(messageType, pointer)) {
    addError(
      report,
      "gate.pointer_not_found",
      at,
      `${pointer} does not resolve against ${messageType}`,
    );
  }
}

export function resolveSchemaPointer(
  messageType: SchemaName,
  pointer: string,
): JsonSchema | null {
  if (!pointer.startsWith("/") || pointer === "/") return null;
  const segments = pointer
    .slice(1)
    .split("/")
    .map((segment) =>
      segment.replaceAll("~1", "/").replaceAll("~0", "~"),
    );
  let candidates: JsonSchema[] = [SCHEMA_DEFINITIONS[messageType]];
  for (const segment of segments) {
    const next: JsonSchema[] = [];
    for (const candidate of flattenBranches(candidates)) {
      const properties = candidate.properties;
      if (
        properties &&
        typeof properties === "object" &&
        Object.hasOwn(properties, segment)
      ) {
        next.push(
          (properties as Record<string, JsonSchema>)[segment]!,
        );
        continue;
      }
      if (
        candidate.type === "array" &&
        /^\d+$/.test(segment) &&
        candidate.items &&
        typeof candidate.items === "object"
      ) {
        next.push(candidate.items as JsonSchema);
      }
    }
    if (next.length === 0) return null;
    candidates = next;
  }
  const flattened = flattenBranches(candidates);
  if (flattened.length === 1) return flattened[0]!;
  return { anyOf: flattened };
}

function flattenBranches(values: JsonSchema[]): JsonSchema[] {
  return values.flatMap((value) => {
    const branches = [
      ...(Array.isArray(value.anyOf)
        ? (value.anyOf as JsonSchema[])
        : []),
      ...(Array.isArray(value.oneOf)
        ? (value.oneOf as JsonSchema[])
        : []),
    ];
    return branches.length > 0 ? flattenBranches(branches) : [value];
  });
}

function schemaSupportsValueType(
  schema: JsonSchema,
  valueType: GateValueType,
): boolean {
  const branches = flattenBranches([schema]);
  return branches.some((branch) => {
    const types = Array.isArray(branch.type)
      ? branch.type
      : [branch.type];
    switch (valueType) {
      case "string":
        return types.includes("string");
      case "number":
        return types.includes("number") || types.includes("integer");
      case "boolean":
        return types.includes("boolean");
      case "null":
        return types.includes("null");
      case "object":
        return types.includes("object");
      case "string_array": {
        if (!types.includes("array")) return false;
        const items =
          branch.items && typeof branch.items === "object"
            ? (branch.items as JsonSchema)
            : null;
        return Boolean(
          items &&
            schemaSupportsValueType(items, "string"),
        );
      }
    }
  });
}

function collectProperties(schema: JsonSchema): Set<string> {
  const result = new Set<string>();
  for (const branch of flattenBranches([schema])) {
    if (branch.properties && typeof branch.properties === "object") {
      Object.keys(branch.properties).forEach((key) => result.add(key));
    }
  }
  return result;
}

function expectedMatchesValueType(
  expected: GateExpected,
  valueType: GateValueType,
): boolean {
  switch (valueType) {
    case "string":
      return typeof expected === "string";
    case "number":
      return typeof expected === "number" && Number.isFinite(expected);
    case "boolean":
      return typeof expected === "boolean";
    case "null":
      return expected === null;
    case "string_array":
      return (
        Array.isArray(expected) &&
        expected.every((item) => typeof item === "string")
      );
    case "object":
      return Array.isArray(expected);
  }
}

function equalsCheck(
  checkId: string,
  actual: GateActual,
  expected: GateExpected,
): MechanicalGateCheck {
  return { checkId, predicate: "equals", actual, expected };
}

function validatorActual(
  fact:
    | "schema_valid"
    | "message_binding_matches"
    | "registered_validator_passes"
    | "duplicate_commit"
    | "script_transition_valid",
): GateActual {
  return {
    source: "validator",
    fact,
    pointer: null,
    valueType: "boolean",
  };
}

function runtimeActual(
  fact:
    | "allowed_tool_events_only"
    | "allowed_paths_only"
    | "turn_budget_within_contract"
    | "experiment_execution_count"
    | "external_evidence_used",
  valueType: GateValueType,
): GateActual {
  return { source: "runtime", fact, valueType };
}
