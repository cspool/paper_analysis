import {
  ROLE_MESSAGE_TYPES,
  ROLE_REASONING_EFFORT,
  ROLE_SKILLS,
  type RegisteredRole,
} from "../contracts/index.ts";
import { ALLOWED_EVIDENCE_TOOLS } from "../security/no_experiment_guard.ts";

export interface RoleProfile {
  protocolVersion: 1;
  role: RegisteredRole;
  lifecycle: "fresh_turn";
  reasoningEffort: "high" | "max";
  skill: string;
  tools: readonly string[];
  filesystem: "none" | "vault_read_only_task_paths";
  network: false;
  delegation: false;
  goals: false;
  stateWrite: false;
  experimentExecution: false;
  allowedInputMessageTypes: readonly [string];
  allowedOutputMessageTypes: readonly [string];
}

function profile(
  role: RegisteredRole,
  tools: readonly string[],
  filesystem: RoleProfile["filesystem"],
): RoleProfile {
  return Object.freeze({
    protocolVersion: 1,
    role,
    lifecycle: "fresh_turn",
    reasoningEffort: ROLE_REASONING_EFFORT[role],
    skill: ROLE_SKILLS[role],
    tools: Object.freeze([...tools]),
    filesystem,
    network: false,
    delegation: false,
    goals: false,
    stateWrite: false,
    experimentExecution: false,
    allowedInputMessageTypes: Object.freeze([
      ROLE_MESSAGE_TYPES[role].input,
    ]) as unknown as readonly [string],
    allowedOutputMessageTypes: Object.freeze([
      ROLE_MESSAGE_TYPES[role].output,
    ]) as unknown as readonly [string],
  });
}

export const ROLE_PROFILES = Object.freeze({
  workflow_decision: profile("workflow_decision", [], "none"),
  evidence_reader: profile(
    "evidence_reader",
    ALLOWED_EVIDENCE_TOOLS,
    "vault_read_only_task_paths",
  ),
  direction_reviewer: profile("direction_reviewer", [], "none"),
  closure_reviewer: profile("closure_reviewer", [], "none"),
} as const satisfies Readonly<Record<RegisteredRole, RoleProfile>>);

export interface EffortCapabilityManifest {
  provider: string;
  model: string;
  wireEffortByLogicalEffort: {
    high: string;
    max: string;
  };
  highestWireEffort: string;
}

export function resolveWireEffort(
  role: RegisteredRole,
  manifest: EffortCapabilityManifest,
): string {
  const logical = ROLE_REASONING_EFFORT[role];
  const wire = manifest.wireEffortByLogicalEffort[logical];
  if (!wire) {
    throw new Error(
      `runtime capability does not map logical effort ${logical} for ${manifest.provider}/${manifest.model}`,
    );
  }
  if (logical === "max" && wire !== manifest.highestWireEffort) {
    throw new Error(
      "workflow_decision max must map to the provider's declared highest wire effort",
    );
  }
  return wire;
}

