import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { ControllerConfig } from "./controller.ts";
import type { TurnBudget } from "./contracts/index.ts";

export interface RunConfigFile extends ControllerConfig {
  protocolVersion: 1;
  runId: string;
  workflowId: string;
}

export function defaultControllerConfig(
  projectRoot: string,
  workDir: string,
  model = "gpt-5.6-sol",
): ControllerConfig {
  const workflowBudget = zeroToolBudget(600_000, 96_000, 32_000);
  const reviewerBudget = zeroToolBudget(300_000, 96_000, 24_000);
  return {
    projectRoot: resolve(projectRoot),
    workDir: resolve(workDir),
    model,
    skillRoot: resolve(projectRoot, ".codex/skills"),
    schemaManifestPath: resolve(
      projectRoot,
      "scripts/simple_semantic_loop/schemas/schema_manifest.json",
    ),
    capabilityManifest: {
      provider: "codex-app-server",
      model,
      wireEffortByLogicalEffort: {
        high: "high",
        max: "max",
      },
      highestWireEffort: "max",
    },
    budgets: {
      workflow: workflowBudget,
      evidence: {
        timeoutMs: 300_000,
        maxInputTokens: 96_000,
        maxOutputTokens: 24_000,
        maxToolCalls: 12,
        evidenceRead: {
          maxLogicalQueries: 3,
          maxSearchToolCalls: 6,
          maxHitsConsidered: 50,
          maxSelectedSources: 3,
          maxContextsRead: 6,
        },
      },
      direction: structuredClone(reviewerBudget),
      closure: structuredClone(reviewerBudget),
    },
    maxTransitionsPerRun: 100,
    noProgressThreshold: 3,
  };
}

export function loadRunConfig(workDir: string): RunConfigFile {
  const config = JSON.parse(
    readFileSync(resolve(workDir, "config.json"), "utf8"),
  ) as RunConfigFile;
  if (config.protocolVersion !== 1) {
    throw new Error("unsupported run config protocol");
  }
  if (
    "roleEffort" in config ||
    "reasoningEffort" in config ||
    "protocolRepair" in config
  ) {
    throw new Error(
      "run config cannot override role effort or introduce repair roles",
    );
  }
  return config;
}

function zeroToolBudget(
  timeoutMs: number,
  maxInputTokens: number,
  maxOutputTokens: number,
): TurnBudget {
  return {
    timeoutMs,
    maxInputTokens,
    maxOutputTokens,
    maxToolCalls: 0,
    evidenceRead: null,
  };
}
