import type {
  GateActual,
  GateDefinition,
  MechanicalGateCheck,
  ValidationReport,
} from "../contracts/index.ts";
import { canonicalEqual } from "../contracts/index.ts";
import { addError, emptyReport } from "../validators/schema_validator.ts";

export const GATE_EVALUATOR_VERSION =
  "simple-semantic-loop-gate-evaluator/1";

export type GateResolution =
  | {
      resolved: true;
      value: unknown;
      detail: string;
    }
  | {
      resolved: false;
      errorCode: string;
      detail: string;
    };

export interface GateEvaluationContext {
  resolve(actual: GateActual): GateResolution;
}

export interface GateCheckResult {
  checkId: string;
  predicate: MechanicalGateCheck["predicate"];
  passed: boolean;
  errorCode: string | null;
  detail: string;
}

export interface GateEvaluation {
  gateId: string;
  passed: boolean;
  checks: GateCheckResult[];
}

/**
 * Gate evaluation is a total, fail-closed function. An unresolvable operand,
 * invalid runtime value, or resolver exception becomes a failed check and can
 * never terminate the Controller process.
 */
export function evaluateGate(
  gate: GateDefinition,
  context: GateEvaluationContext,
): GateEvaluation {
  if (
    typeof gate.evaluatorVersion === "string" &&
    gate.evaluatorVersion !== GATE_EVALUATOR_VERSION
  ) {
    return {
      gateId: gate.gateId,
      passed: false,
      checks: [
        {
          checkId: "controller.evaluator_version",
          predicate: "equals",
          passed: false,
          errorCode: "gate.evaluator_version_unsupported",
          detail: `frozen evaluator ${gate.evaluatorVersion} is not ${GATE_EVALUATOR_VERSION}`,
        },
      ],
    };
  }
  const checks = gate.mechanicalChecks.map((check) =>
    evaluateCheck(check, context),
  );
  return {
    gateId: gate.gateId,
    passed: checks.every((check) => check.passed),
    checks,
  };
}

function evaluateCheck(
  check: MechanicalGateCheck,
  context: GateEvaluationContext,
): GateCheckResult {
  let resolution: GateResolution;
  try {
    resolution = context.resolve(check.actual);
  } catch (error) {
    return failed(
      check,
      "gate.resolver_exception",
      error instanceof Error ? error.message : String(error),
    );
  }
  if (!resolution.resolved) {
    return failed(check, resolution.errorCode, resolution.detail);
  }

  try {
    switch (check.predicate) {
      case "equals": {
        const passed = canonicalEqual(resolution.value, check.expected);
        return {
          checkId: check.checkId,
          predicate: check.predicate,
          passed,
          errorCode: passed ? null : "gate.value_mismatch",
          detail: passed
            ? resolution.detail
            : `${resolution.detail}; actual does not equal expected`,
        };
      }
      case "contains_fields": {
        const fields = Array.isArray(check.expected)
          ? check.expected
          : [];
        const value = resolution.value;
        const passed =
          value !== null &&
          typeof value === "object" &&
          !Array.isArray(value) &&
          fields.every((field) =>
            Object.hasOwn(
              value as Record<string, unknown>,
              field,
            ),
          );
        return {
          checkId: check.checkId,
          predicate: check.predicate,
          passed,
          errorCode: passed ? null : "gate.required_field_missing",
          detail: passed
            ? resolution.detail
            : `${resolution.detail}; one or more required fields are absent`,
        };
      }
    }
  } catch (error) {
    return failed(
      check,
      "gate.non_json_or_invalid_value",
      error instanceof Error ? error.message : String(error),
    );
  }
}

function failed(
  check: MechanicalGateCheck,
  errorCode: string,
  detail: string,
): GateCheckResult {
  return {
    checkId: check.checkId,
    predicate: check.predicate,
    passed: false,
    errorCode,
    detail,
  };
}

export function gateEvaluationReport(
  evaluation: GateEvaluation,
): ValidationReport {
  const report = emptyReport();
  evaluation.checks.forEach((check, index) => {
    if (!check.passed) {
      addError(
        report,
        check.errorCode ?? `gate.${check.predicate}`,
        `/mechanicalChecks/${index}`,
        `${check.checkId} failed: ${check.detail}`,
      );
    }
  });
  return report;
}
