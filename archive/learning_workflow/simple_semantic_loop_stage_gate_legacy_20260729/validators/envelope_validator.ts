import type {
  PayloadTurnEnvelope,
  RegisteredRole,
  StateBinding,
  TurnIdentity,
  ValidationReport,
} from "../contracts/index.ts";
import {
  ROLE_MESSAGE_TYPES,
  ROLE_REASONING_EFFORT,
} from "../contracts/index.ts";
import { canonicalEqual } from "../contracts/index.ts";
import { addError, emptyReport } from "./schema_validator.ts";

export interface ExpectedTurnBinding {
  role: RegisteredRole;
  task: TurnIdentity;
  stateBinding: StateBinding;
  inputHash?: string;
  decisionInputHash?: string;
}

export function validateFixedEffort(
  role: RegisteredRole,
  logicalEffort: string,
): ValidationReport {
  const report = emptyReport();
  if (logicalEffort !== ROLE_REASONING_EFFORT[role]) {
    addError(
      report,
      "registry.reasoning_effort",
      "/logicalEffort",
      `${role} requires logical effort ${ROLE_REASONING_EFFORT[role]}`,
    );
  }
  return report;
}

export function validateTurnIdentityEcho(
  response: TurnIdentity,
  expected: ExpectedTurnBinding,
): ValidationReport {
  const report = emptyReport();
  const fields = [
    "protocolVersion",
    "workflowId",
    "runId",
    "taskId",
    "attemptId",
    "stageId",
    "stageContractHash",
  ] as const;
  for (const field of fields) {
    if (response[field] !== expected.task[field]) {
      addError(
        report,
        "binding.identity_mismatch",
        `/${field}`,
        `${field} does not echo the dispatch`,
      );
    }
  }
  const outputType = ROLE_MESSAGE_TYPES[expected.role].output;
  if (response.messageType !== outputType) {
    addError(
      report,
      "registry.output_message_type",
      "/messageType",
      `${expected.role} may only emit ${outputType}`,
    );
  }
  return report;
}

export function validatePayloadEnvelopeBinding(
  response: PayloadTurnEnvelope<unknown>,
  expected: ExpectedTurnBinding,
): ValidationReport {
  const report = validateTurnIdentityEcho(response, expected);
  if (!canonicalEqual(response.stateBinding, expected.stateBinding)) {
    addError(
      report,
      "binding.state_mismatch",
      "/stateBinding",
      "response StateBinding does not match the dispatch snapshot",
    );
  }
  if (expected.inputHash && response.inputHash !== expected.inputHash) {
    addError(
      report,
      "binding.input_hash_mismatch",
      "/inputHash",
      "response inputHash does not match the immutable task",
    );
  }
  return report;
}

