import path from "node:path";

import type { Effort, Role, RoleProfile } from "./types.ts";

export const ROLE_EFFORT: Record<Role, Effort> = {
  anchor_stage_controller: "high",
  anchor_evidence_worker: "medium",
  anchor_curator_worker: "high",
  direction_planner: "xhigh",
  direction_reviewer: "xhigh",
  review_evidence_worker: "high",
};

const ROLE_DEFINITIONS: Record<Role, Omit<RoleProfile, "skillPath">> = {
  anchor_stage_controller: {
    role: "anchor_stage_controller",
    skillName: "learning-anchor-stage-controller",
    effort: "high",
    persistent: true,
    knowledgeAccess: "none",
    allowedMainMarkers: ["ANCHOR_ROUND_PLAN", "ANCHOR_STAGE_COMPLETE"],
  },
  anchor_evidence_worker: {
    role: "anchor_evidence_worker",
    skillName: "learning-anchor-evidence-worker",
    effort: "medium",
    persistent: false,
    knowledgeAccess: "obsidian_readonly",
    allowedMainMarkers: ["ANCHOR_EVIDENCE_RESULT"],
  },
  anchor_curator_worker: {
    role: "anchor_curator_worker",
    skillName: "learning-anchor-curator-worker",
    effort: "high",
    persistent: false,
    knowledgeAccess: "none",
    allowedMainMarkers: ["ANCHOR_DELTA"],
  },
  direction_planner: {
    role: "direction_planner",
    skillName: "learning-direction-planner",
    effort: "xhigh",
    persistent: true,
    knowledgeAccess: "none",
    allowedMainMarkers: ["DIRECTION_PROPOSAL", "DIRECTION_PLANNING_COMPLETE"],
  },
  direction_reviewer: {
    role: "direction_reviewer",
    skillName: "learning-direction-reviewer",
    effort: "xhigh",
    persistent: true,
    knowledgeAccess: "none",
    allowedMainMarkers: ["REVIEW_QUESTION", "REVIEW_REFERENCE_REQUEST", "DIRECTION_REVIEW_COMPLETE"],
  },
  review_evidence_worker: {
    role: "review_evidence_worker",
    skillName: "learning-review-evidence-worker",
    effort: "high",
    persistent: false,
    knowledgeAccess: "obsidian_readonly",
    allowedMainMarkers: ["REVIEW_EVIDENCE_RESULT"],
  },
};

export function roleProfile(role: Role, skillRoot: string): RoleProfile {
  const definition = ROLE_DEFINITIONS[role];
  return {
    ...definition,
    skillPath: path.join(skillRoot, definition.skillName, "SKILL.md"),
  };
}

export function allRoleProfiles(skillRoot: string): RoleProfile[] {
  return (Object.keys(ROLE_DEFINITIONS) as Role[]).map((role) => roleProfile(role, skillRoot));
}

export const REVIEW_DIMENSIONS = [
  "scenario_opportunity",
  "baseline_fairness",
  "entry_validity",
  "cross_layer_validity",
  "implementation_reuse",
  "experiment_measurement",
] as const;

export const REVIEW_REFERENCE_KEYS = [
  "scenario_and_acceleration",
  "baseline_and_fairness",
  "layer_modification_and_implementation",
  "cross_layer_interface_and_conflict",
  "experiment_tool_and_measurement",
] as const;

export const OBSIDIAN_READONLY_TOOLS = new Set([
  "obsidian_search_notes",
  "obsidian_get_note",
  "obsidian_list_notes",
  "obsidian_read_note",
  "search_notes",
  "get_note",
  "list_notes",
  "read_note",
]);

