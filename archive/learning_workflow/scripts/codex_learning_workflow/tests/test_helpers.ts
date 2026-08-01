import fs from "node:fs";
import path from "node:path";

import { ROLE_EFFORT } from "../role_profiles.ts";
import type { RunConfig } from "../types.ts";

export function testConfig(root: string): RunConfig {
  const sourceRoot = path.join(root, "source_codex");
  fs.mkdirSync(sourceRoot, { recursive: true });
  fs.mkdirSync(path.join(root, ".codex/skills"), { recursive: true });
  fs.writeFileSync(path.join(sourceRoot, "auth.json"), "{}\n", "utf8");
  fs.writeFileSync(path.join(sourceRoot, "config.toml"), "[mcp_servers.obsidian]\ncommand = \"fake\"\n", "utf8");
  return {
    protocolVersion: 1,
    topic: "test topic",
    constraints: [],
    model: "gpt-5.6-sol",
    roleEffort: { ...ROLE_EFFORT },
    protocolRepairEffort: "low",
    maxAnchors: 1,
    noNewAnchorStop: 2,
    maxStage1Rounds: 2,
    maxStage1Tasks: 2,
    evidenceTasksPerRound: 1,
    anchorEvidenceConcurrency: 2,
    curatorConcurrency: 1,
    maxDirectionsPerAnchor: 1,
    maxPlannerTurnsPerAnchor: 2,
    maxReviewRoundsPerDirection: 8,
    directionConcurrency: 2,
    maxTotalTurns: 50,
    maxPersistentTurns: 12,
    turnTimeoutMs: 3_000,
    requestTimeoutMs: 2_000,
    startupTimeoutMs: 2_000,
    vaultRoot: root,
    evidenceRoots: ["paper_secs", "knowledge_notes"],
    projectRoot: root,
    skillRoot: path.join(root, ".codex/skills"),
    sourceConfigPath: path.join(sourceRoot, "config.toml"),
    sourceAuthPath: path.join(sourceRoot, "auth.json"),
    createdAt: new Date(0).toISOString(),
  };
}
