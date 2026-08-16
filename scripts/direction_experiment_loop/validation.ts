import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { verifyFrozenFiles } from "./setup.ts";
import { DirectionExperimentStore } from "./store.ts";
import {
  DIRECTION_EXPERIMENT_FORMAT_VERSION,
  EVIDENCE_SCOPES,
  EXPERIMENT_DECISIONS,
  JUDGE_ASSESSMENTS,
  READABLE_DIRECTION_EXPERIMENT_FORMAT_VERSIONS,
  type DirectionGoalRecord,
  type DirectionHistoryEntry,
  type DirectionRunFile,
  type DirectionStateFile,
  type ExperimentContractRecord,
  type LabGoalInvocationRecord,
} from "./types.ts";

export interface DirectionValidationReport {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export function validateDirectionExperimentRun(workDir: string): DirectionValidationReport {
  const store = new DirectionExperimentStore(resolve(workDir));
  const errors: string[] = [];
  const warnings: string[] = [];
  for (const required of ["run.json", "state.json", "history.jsonl"]) {
    if (!store.exists(required)) errors.push(`missing ${required}`);
  }
  if (errors.length > 0) return { valid: false, errors, warnings };

  let run: DirectionRunFile;
  let state: DirectionStateFile;
  try {
    run = store.readRun();
    state = store.readState();
  } catch (error) {
    return { valid: false, errors: [`invalid run/state JSON: ${String(error)}`], warnings };
  }
  const runVersion = Number(run.formatVersion);
  const stateVersion = Number(state.formatVersion);
  const readable = (READABLE_DIRECTION_EXPERIMENT_FORMAT_VERSIONS as readonly number[])
    .includes(runVersion);
  const versionsMatch = runVersion === stateVersion;
  if (!versionsMatch || !readable || run.workflow !== "DIRECTION_EXPERIMENT_LOOP") {
    errors.push("unsupported or inconsistent formatVersion/workflow");
  }
  const current = versionsMatch && runVersion === DIRECTION_EXPERIMENT_FORMAT_VERSION;
  if (readable && !current) {
    warnings.push(
      `format v${runVersion} is audit-only; initialize a v${DIRECTION_EXPERIMENT_FORMAT_VERSION} run to continue`,
    );
    validateJsonLines(store, errors);
    if (!existsSync(resolve(run.source.sourceWorkDir))) {
      warnings.push(`source Learning workDir is unavailable: ${run.source.sourceWorkDir}`);
    }
    return { valid: errors.length === 0, errors, warnings };
  }

  errors.push(...verifyFrozenFiles(store));
  validateState(store, run, state, errors);
  const history = validateHistory(store, run, state, errors);
  validateFinal(store, run, state, errors);
  if (!existsSync(resolve(run.source.sourceWorkDir))) {
    warnings.push(`source Learning workDir is unavailable: ${run.source.sourceWorkDir}`);
  }
  const contractCount = history.filter((entry) =>
    entry.kind === "DECISION" && entry.decision === "RUN_LAB"
  ).length;
  if (contractCount !== state.activeContractRevision) {
    errors.push("state activeContractRevision/history contract count mismatch");
  }
  return { valid: errors.length === 0, errors, warnings };
}

function validateState(
  store: DirectionExperimentStore,
  run: DirectionRunFile,
  state: DirectionStateFile,
  errors: string[],
): void {
  if (state.authorizedLabCycles < 1 || state.cycle < 0) {
    errors.push("invalid Lab cycle counters");
  }
  if (state.activeContractRevision === 0) {
    if (state.activeContractRef !== null || state.activeContractHash !== null) {
      errors.push("contract revision zero requires null active contract ref/hash");
    }
  } else if (!state.activeContractRef || !state.activeContractHash) {
    errors.push("positive contract revision requires active contract ref/hash");
  } else {
    validateContract(
      store,
      run,
      state.activeContractRevision,
      state.activeContractRef,
      state.activeContractHash,
      errors,
    );
  }
  if (state.evidenceScope !== null && !isEvidenceScope(state.evidenceScope)) {
    errors.push("state has invalid evidenceScope");
  }
  for (const [name, ref] of [
    ["activeGoalRecordRef", state.activeGoalRecordRef],
    ["activeLabInvocationRef", state.activeLabInvocationRef],
    ["activeJudgeRequestRef", state.activeJudgeRequestRef],
    ["latestLabResultRef", state.latestLabResultRef],
    ["latestCheckpointRef", state.latestCheckpointRef],
    ["latestJudgeRef", state.latestJudgeRef],
    ["latestDecisionRef", state.latestDecisionRef],
  ] as const) {
    if (ref && !store.exists(ref)) errors.push(`${name} is missing: ${ref}`);
  }
  if (state.activeGoalRecordRef && store.exists(state.activeGoalRecordRef)) {
    validateGoal(store, state.activeGoalRecordRef, errors);
  }
  if (state.node === "LAB_GOAL") {
    if (!state.activeContractRef) errors.push("LAB_GOAL node lacks active contract");
    if (!state.latestDecisionRef) errors.push("LAB_GOAL node lacks source Decision");
  }
  if (state.node === "JUDGE" && !state.activeJudgeRequestRef) {
    errors.push("JUDGE node lacks activeJudgeRequestRef");
  }
  if (
    state.lifecycle === "RUNNING" &&
    state.node === "DECISION" &&
    (state.activeGoalRecordRef !== null ||
      state.activeLabInvocationRef !== null ||
      state.activeJudgeRequestRef !== null)
  ) errors.push("running DECISION node retains an active Lab/Judge execution ref");
  if (state.lifecycle === "FINISHED" && state.finalDecision === null) {
    errors.push("finished state lacks finalDecision");
  }
}

function validateGoal(
  store: DirectionExperimentStore,
  goalRef: string,
  errors: string[],
): void {
  const goal = store.readJson<DirectionGoalRecord>(goalRef);
  for (const ref of [goal.contractRef, goal.bindingRef]) {
    if (!store.exists(ref)) errors.push(`Goal has missing binding ref: ${ref}`);
  }
  if (store.exists(goal.bindingRef) && store.sha256(goal.bindingRef) !== goal.bindingHash) {
    errors.push(`Goal cycle binding hash mismatch: ${goal.bindingRef}`);
  }
  for (let index = 0; index < goal.invocationRefs.length; index += 1) {
    const ref = goal.invocationRefs[index]!;
    if (!store.exists(ref)) {
      errors.push(`Goal invocation ref is missing: ${ref}`);
      continue;
    }
    const invocation = store.readJson<LabGoalInvocationRecord>(ref);
    if (
      invocation.ordinal !== index + 1 ||
      invocation.goalRef !== goalRef ||
      invocation.cycle !== goal.cycle ||
      invocation.contractRevision !== goal.contractRevision ||
      invocation.contractHash !== goal.contractHash
    ) errors.push(`Goal invocation binding mismatch: ${ref}`);
  }
}

function validateHistory(
  store: DirectionExperimentStore,
  run: DirectionRunFile,
  state: DirectionStateFile,
  errors: string[],
): DirectionHistoryEntry[] {
  const entries: DirectionHistoryEntry[] = [];
  const lines = store.readText("history.jsonl").split("\n").filter(Boolean);
  let decisionOrdinal = 0;
  let judgmentOrdinal = 0;
  let experimentCycle = 0;
  let contractRevision = 0;
  for (let index = 0; index < lines.length; index += 1) {
    let entry: DirectionHistoryEntry;
    try {
      entry = JSON.parse(lines[index]!) as DirectionHistoryEntry;
      entries.push(entry);
    } catch (error) {
      errors.push(`invalid history JSONL line ${index + 1}: ${String(error)}`);
      continue;
    }
    if (entry.kind === "DECISION") {
      decisionOrdinal += 1;
      if (entry.ordinal !== decisionOrdinal) {
        errors.push(`Decision ordinal mismatch at history line ${index + 1}`);
      }
      if (!EXPERIMENT_DECISIONS.includes(entry.decision)) {
        errors.push(`invalid Decision literal at history line ${index + 1}`);
      }
      if (!isEvidenceScope(entry.evidenceScope)) {
        errors.push(`invalid Decision evidenceScope at history line ${index + 1}`);
      }
      requireRef(store, entry.decisionRef, index, errors);
      if (entry.decision === "RUN_LAB") {
        contractRevision += 1;
        if (
          entry.contractRevision !== contractRevision ||
          !entry.contractRef ||
          !entry.contractHash
        ) errors.push(`RUN_LAB contract binding mismatch at history line ${index + 1}`);
        else {
          validateContract(
            store,
            run,
            contractRevision,
            entry.contractRef,
            entry.contractHash,
            errors,
          );
        }
      } else if (entry.contractRef !== null && entry.contractRevision === null) {
        errors.push(`Decision contract ref/revision mismatch at history line ${index + 1}`);
      }
      if (entry.reviewFocusRef) requireRef(store, entry.reviewFocusRef, index, errors);
    } else if (entry.kind === "EXPERIMENT") {
      experimentCycle += 1;
      if (entry.cycle !== experimentCycle) {
        errors.push(`Experiment cycle mismatch at history line ${index + 1}`);
      }
      for (const ref of [
        entry.contractRef,
        entry.sourceDecisionRef,
        entry.goalRecordRef,
        entry.resultRef,
        ...entry.invocationRefs,
      ]) requireRef(store, ref, index, errors);
      if (entry.checkpointRef) requireRef(store, entry.checkpointRef, index, errors);
      if (store.exists(entry.contractRef) && store.sha256(entry.contractRef) !== entry.contractHash) {
        errors.push(`Experiment contract hash mismatch at history line ${index + 1}`);
      }
      if (!store.isNonEmptyFile(entry.resultRef)) {
        errors.push(`Experiment result is empty at history line ${index + 1}`);
      }
    } else if (entry.kind === "JUDGMENT") {
      judgmentOrdinal += 1;
      if (entry.ordinal !== judgmentOrdinal) {
        errors.push(`Judgment ordinal mismatch at history line ${index + 1}`);
      }
      if (!JUDGE_ASSESSMENTS.includes(entry.assessment)) {
        errors.push(`invalid Judge assessment at history line ${index + 1}`);
      }
      if (!isEvidenceScope(entry.evidenceScope)) {
        errors.push(`invalid Judge evidenceScope at history line ${index + 1}`);
      }
      for (const ref of [entry.requestRef, entry.judgmentRef]) {
        requireRef(store, ref, index, errors);
      }
      if (entry.contractRef) requireRef(store, entry.contractRef, index, errors);
      if (entry.labResultRef) requireRef(store, entry.labResultRef, index, errors);
    } else errors.push(`unknown history kind at line ${index + 1}`);
  }
  if (experimentCycle > state.cycle) {
    errors.push("history has more completed experiments than state cycle");
  }
  return entries;
}

function validateContract(
  store: DirectionExperimentStore,
  run: DirectionRunFile,
  revision: number,
  ref: string,
  hash: string,
  errors: string[],
): void {
  if (!store.exists(ref)) {
    errors.push(`missing experiment contract: ${ref}`);
    return;
  }
  if (store.sha256(ref) !== hash) {
    errors.push(`experiment contract hash mismatch: ${ref}`);
    return;
  }
  const contract = store.readJson<ExperimentContractRecord>(ref);
  if (
    contract.formatVersion !== 2 ||
    contract.contractRevision !== revision ||
    contract.directionId !== run.source.directionId ||
    contract.directionRevision !== run.source.directionRevision ||
    contract.sourceDirectionRef !== run.inputs.directionResult.path ||
    !store.exists(contract.decisionRef)
  ) errors.push(`experiment contract identity mismatch: ${ref}`);
  if (!Array.isArray(contract.stopConditions) || contract.stopConditions.length === 0 ||
    !contract.stopConditions.every((item) => typeof item === "string" && item.trim())) {
    errors.push(`experiment contract lacks stopConditions: ${ref}`);
  }
  if (!Number.isFinite(contract.estimatedMinutes) || contract.estimatedMinutes <= 0) {
    errors.push(`experiment contract has invalid estimatedMinutes: ${ref}`);
  }
  const maxMinutes = Math.max(
    1,
    Math.floor((run.budgets.lab.hardTimeoutMs - run.budgets.labResultReserveMs) / 60_000),
  );
  if (contract.estimatedMinutes > maxMinutes) {
    errors.push(`experiment contract exceeds Lab runtime envelope: ${ref}`);
  }
  const bindingRef = `contracts/contract-${revision}/binding.json`;
  if (!store.exists(bindingRef)) errors.push(`missing experiment contract binding: ${bindingRef}`);
  else {
    const binding = store.readJson<Record<string, unknown>>(bindingRef);
    if (
      binding.contractRef !== ref ||
      binding.contractHash !== hash ||
      binding.contractRevision !== revision ||
      binding.decisionRef !== contract.decisionRef
    ) errors.push(`experiment contract binding mismatch: ${bindingRef}`);
  }
}

function validateFinal(
  store: DirectionExperimentStore,
  run: DirectionRunFile,
  state: DirectionStateFile,
  errors: string[],
): void {
  if (store.exists("final/outcome.json")) {
    const outcome = store.readJson<{ evidenceScope?: unknown }>("final/outcome.json");
    if (outcome.evidenceScope !== state.evidenceScope) {
      errors.push("outcome/state evidenceScope mismatch");
    }
  }
  if (state.lifecycle !== "FINISHED") return;
  for (const path of ["final/report.md", "final/handoff.json", "final/outcome.json"]) {
    if (!store.exists(path)) errors.push(`finished run is missing ${path}`);
  }
  if (!store.exists("final/handoff.json")) return;
  const handoff = store.readJson<Record<string, unknown>>("final/handoff.json");
  if (
    handoff.directionId !== run.source.directionId ||
    handoff.directionRevision !== run.source.directionRevision
  ) errors.push("finished handoff/source Direction mismatch");
  if (
    handoff.finalDecision !== state.finalDecision ||
    handoff.evidenceScope !== state.evidenceScope ||
    handoff.activeContractRevision !== state.activeContractRevision ||
    handoff.activeContractRef !== state.activeContractRef ||
    handoff.activeContractHash !== state.activeContractHash
  ) errors.push("finished handoff/state mismatch");
  const decisionRef = handoff.finalDecisionRef;
  if (typeof decisionRef !== "string" || !store.exists(decisionRef)) {
    errors.push(`finished handoff has missing finalDecisionRef: ${decisionRef}`);
  }
}

function validateJsonLines(store: DirectionExperimentStore, errors: string[]): void {
  const lines = store.readText("history.jsonl").split("\n").filter(Boolean);
  for (let index = 0; index < lines.length; index += 1) {
    try {
      JSON.parse(lines[index]!);
    } catch (error) {
      errors.push(`invalid history JSONL line ${index + 1}: ${String(error)}`);
    }
  }
}

function requireRef(
  store: DirectionExperimentStore,
  ref: string,
  historyIndex: number,
  errors: string[],
): void {
  if (!store.exists(ref)) {
    errors.push(`history line ${historyIndex + 1} has missing ref: ${ref}`);
  }
}

function isEvidenceScope(value: unknown): boolean {
  return typeof value === "string" &&
    EVIDENCE_SCOPES.includes(value as (typeof EVIDENCE_SCOPES)[number]);
}
