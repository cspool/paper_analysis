import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { atomicWriteJson } from "./canonical_store.ts";
import { allRoleProfiles } from "./role_profiles.ts";
import { AppServerClient } from "./app_server_client.ts";
import { canonicalJson } from "./stable_ids.ts";
import type { JsonValue, ProviderManifest, RunState } from "./types.ts";

function commandOutput(command: string, args: string[], env?: NodeJS.ProcessEnv): string {
  const result = spawnSync(command, args, {
    encoding: "utf8",
    env: env ?? process.env,
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.status !== 0) {
    throw new Error(
      `${command} ${args.join(" ")} failed (${result.status}): ${String(result.stderr || result.stdout).trim()}`,
    );
  }
  return String(result.stdout).trim();
}

function filesRecursively(root: string): string[] {
  if (!fs.existsSync(root)) return [];
  const output: string[] = [];
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop()!;
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const absolute = path.join(current, entry.name);
      if (entry.isDirectory()) stack.push(absolute);
      if (entry.isFile()) output.push(absolute);
    }
  }
  return output.sort();
}

function directoryHash(root: string): string {
  const hash = createHash("sha256");
  for (const file of filesRecursively(root)) {
    hash.update(path.relative(root, file));
    hash.update("\0");
    const bytes = fs.readFileSync(file);
    if (file.endsWith(".json")) {
      const parsed = JSON.parse(bytes.toString("utf8")) as JsonValue;
      hash.update(canonicalJson(parsed));
    } else {
      hash.update(bytes);
    }
    hash.update("\0");
  }
  return hash.digest("hex");
}

function supportedMethods(schemaRoot: string): string[] {
  const clientRequest = filesRecursively(schemaRoot).find((file) => path.basename(file) === "ClientRequest.json");
  if (!clientRequest) return [];
  const text = fs.readFileSync(clientRequest, "utf8");
  const methods = new Set<string>();
  for (const match of text.matchAll(/"enum"\s*:\s*\[\s*"([A-Za-z][A-Za-z0-9./_-]+)"\s*\]/g)) {
    if (match[1].includes("/")) methods.add(match[1]);
  }
  return [...methods].sort();
}

function modelListData(response: unknown): Array<Record<string, unknown>> {
  if (!response || typeof response !== "object") return [];
  const data = (response as Record<string, unknown>).data;
  return Array.isArray(data)
    ? data.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object" && !Array.isArray(item))
    : [];
}

function skillNames(response: unknown): Set<string> {
  if (!response || typeof response !== "object") return new Set();
  const root = response as Record<string, unknown>;
  const groups = Array.isArray(root.data) ? root.data : [];
  const names = new Set<string>();
  for (const group of groups) {
    if (!group || typeof group !== "object") continue;
    const skills = Array.isArray((group as Record<string, unknown>).skills)
      ? (group as Record<string, unknown>).skills as unknown[]
      : [];
    for (const skill of skills) {
      if (!skill || typeof skill !== "object") continue;
      const name = (skill as Record<string, unknown>).name;
      if (typeof name === "string") names.add(name);
    }
  }
  return names;
}

function generateSchemas(state: RunState, workDir: string): { hash: string; methods: string[] } {
  const schemaRoot = path.join(workDir, "provider/generated_schema");
  const jsonRoot = path.join(schemaRoot, "json");
  const tsRoot = path.join(schemaRoot, "ts");
  fs.mkdirSync(jsonRoot, { recursive: true });
  fs.mkdirSync(tsRoot, { recursive: true });
  const codexHome = path.join(workDir, "provider/codex_home");
  fs.mkdirSync(codexHome, { recursive: true, mode: 0o700 });
  const env = { ...process.env, CODEX_HOME: codexHome };
  commandOutput("codex", ["app-server", "generate-json-schema", "--out", jsonRoot], env);
  commandOutput("codex", ["app-server", "generate-ts", "--out", tsRoot], env);
  return {
    hash: directoryHash(schemaRoot),
    methods: supportedMethods(jsonRoot),
  };
}

export interface DoctorResult {
  manifest: ProviderManifest;
  warnings: string[];
}

export async function runDoctor(state: RunState, workDir: string): Promise<DoctorResult> {
  const warnings: string[] = [];
  const versionOutput = commandOutput("codex", ["--version"]);
  const versionMatch = /codex-cli\s+([^\s]+)/.exec(versionOutput);
  if (!versionMatch) {
    throw new Error(`cannot parse Codex CLI version: ${versionOutput}`);
  }
  for (const profile of allRoleProfiles(state.config.skillRoot)) {
    if (!fs.existsSync(profile.skillPath)) {
      throw new Error(`required role skill is missing: ${profile.skillPath}`);
    }
  }
  const generated = generateSchemas(state, workDir);
  const previousPath = path.join(workDir, "provider/runtime.json");
  if (fs.existsSync(previousPath)) {
    const previous = JSON.parse(fs.readFileSync(previousPath, "utf8")) as ProviderManifest & {
      schemaHashAlgorithm?: string;
    };
    if (previous.codexCliVersion !== versionMatch[1]) {
      throw new Error(`Codex CLI version changed: ${previous.codexCliVersion} -> ${versionMatch[1]}`);
    }
    if (
      previous.schemaHashAlgorithm === "canonical-json-v1"
      && previous.appServerSchemaHash !== generated.hash
    ) {
      throw new Error("App Server schema hash changed; no migration is defined");
    }
    if (previous.schemaHashAlgorithm !== "canonical-json-v1") {
      warnings.push("migrated legacy order-sensitive schema hash to canonical-json-v1");
    }
  }

  const client = new AppServerClient(state.config, workDir);
  await client.start();
  try {
    const modelsResponse = await client.request("model/list", {
      includeHidden: true,
      limit: 100,
    });
    const model = modelListData(modelsResponse).find((candidate) => candidate.id === state.config.model);
    if (!model) {
      throw new Error(`required model is unavailable: ${state.config.model}`);
    }
    const efforts = Array.isArray(model.supportedReasoningEfforts)
      ? model.supportedReasoningEfforts
        .map((item) => item && typeof item === "object" ? String((item as Record<string, unknown>).reasoningEffort ?? "") : "")
        .filter(Boolean)
      : [];
    const requiredEfforts = new Set([...Object.values(state.config.roleEffort), state.config.protocolRepairEffort]);
    for (const effort of requiredEfforts) {
      if (!efforts.includes(effort)) {
        throw new Error(`${state.config.model} does not support required effort ${effort}`);
      }
    }
    if (Object.values(state.config.roleEffort).includes("ultra" as never)) {
      throw new Error("role configuration illegally includes ultra");
    }

    const skillsResponse = await client.request("skills/list", {
      cwds: [state.config.projectRoot],
      forceReload: true,
    });
    const visibleSkills = skillNames(skillsResponse);
    for (const profile of allRoleProfiles(state.config.skillRoot)) {
      if (!visibleSkills.has(profile.skillName)) {
        warnings.push(`skills/list did not advertise explicitly attached skill ${profile.skillName}`);
      }
    }
    await client.request("permissionProfile/list", {
      cwd: state.config.projectRoot,
      limit: 100,
    });

    const sourceConfig = fs.readFileSync(state.config.sourceConfigPath, "utf8");
    const obsidianConfigured = /\[mcp_servers\.obsidian\]/.test(sourceConfig);
    if (!obsidianConfigured) {
      throw new Error("Obsidian MCP is not configured in the source Codex config");
    }
    const manifest: ProviderManifest = {
      codexCliVersion: versionMatch[1],
      appServerSchemaHash: generated.hash,
      schemaHashAlgorithm: "canonical-json-v1",
      generatedAt: new Date().toISOString(),
      supportedMethods: generated.methods,
      modelVerified: true,
      supportedEfforts: efforts,
      obsidianConfigured,
    };
    atomicWriteJson(previousPath, manifest as unknown as JsonValue);
    return { manifest, warnings };
  } finally {
    await client.stop();
  }
}
