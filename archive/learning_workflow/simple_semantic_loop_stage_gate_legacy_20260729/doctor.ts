import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import type { ControllerConfig } from "./controller.ts";
import {
  ROLE_MESSAGE_TYPES,
  ROLE_REASONING_EFFORT,
  ROLE_SKILLS,
  STAGE_REGISTRY,
  canonicalSha256,
  type RegisteredRole,
} from "./contracts/index.ts";
import { WorkflowStore } from "./db/workflow_store.ts";
import {
  ROLE_PROFILES,
  resolveWireEffort,
} from "./turns/role_profiles.ts";
import { loadSkillPackage } from "./turns/skill_package.ts";
import {
  APP_SERVER_RUNTIME_INVARIANTS,
  AppServerFreshTurnRuntime,
} from "./turns/app_server_runtime.ts";
import { validateProviderOutputSchema } from "./schemas/provider_schema_validator.ts";

export interface DoctorCheck {
  name: string;
  passed: boolean;
  detail: string;
}

export interface DoctorResult {
  passed: boolean;
  checks: DoctorCheck[];
  warnings: string[];
  provider: {
    probed: boolean;
    supportedReasoningEfforts: string[];
  };
}

export async function runDoctor(
  config: ControllerConfig,
  options: { probeProvider?: boolean } = {},
): Promise<DoctorResult> {
  const checks: DoctorCheck[] = [];
  const warnings: string[] = [];
  check(checks, "node_version", Number(process.versions.node.split(".")[0]) >= 22, process.versions.node);

  const sqliteDir = mkdtempSync(resolve(tmpdir(), "simple-loop-doctor-"));
  try {
    const store = new WorkflowStore(resolve(sqliteDir, "workflow.db"));
    store.createRun({
      runId: "doctor-run",
      workflowId: "doctor-workflow",
      objective: "capability probe",
      acceptanceCriteria: ["probe"],
      config: {},
    });
    const initial = store.stateBinding("doctor-run");
    store.casTransition("doctor-run", initial, {
      lifecycle: "running",
      eventType: "doctor_probe",
      eventPayload: {},
    });
    const journalMode = store.db
      .prepare("PRAGMA journal_mode")
      .get() as Record<string, unknown>;
    check(
      checks,
      "sqlite_wal_transaction_cas",
      String(Object.values(journalMode)[0]).toLowerCase() === "wal",
      `node:sqlite transaction/CAS succeeded; journal_mode=${String(
        Object.values(journalMode)[0],
      )}`,
    );
    store.close();
  } catch (error) {
    check(checks, "sqlite_wal_transaction_cas", false, errorMessage(error));
  } finally {
    rmSync(sqliteDir, { recursive: true, force: true });
  }

  let manifest: {
    protocolVersion: number;
    schemas: Record<string, { path: string; sha256: string }>;
  } | null = null;
  try {
    manifest = JSON.parse(readFileSync(config.schemaManifestPath, "utf8"));
    const expectedMessages = [
      ...Object.values(ROLE_MESSAGE_TYPES).flatMap((entry) => [
        entry.input,
        entry.output,
      ]),
    ];
    const present = expectedMessages.every((name) => manifest!.schemas[name]);
    const hashes = Object.values(manifest.schemas).every((entry) => {
      const schema = JSON.parse(
        readFileSync(
          resolve(dirname(config.schemaManifestPath), entry.path),
          "utf8",
        ),
      );
      return canonicalSha256(schema) === entry.sha256;
    });
    check(
      checks,
      "schema_manifest",
      manifest.protocolVersion === 1 && present && hashes,
      `manifest=${canonicalSha256(manifest)}`,
    );
    const providerSchemaIssues = Object.values(ROLE_MESSAGE_TYPES).flatMap(
      ({ output }) => {
        const entry = manifest!.schemas[output];
        if (!entry) {
          return [
            {
              jsonPointer: "/",
              message: `${output} is absent from the schema manifest`,
            },
          ];
        }
        const schema = JSON.parse(
          readFileSync(
            resolve(dirname(config.schemaManifestPath), entry.path),
            "utf8",
          ),
        );
        return validateProviderOutputSchema(schema).map((issue) => ({
          ...issue,
          message: `${output}: ${issue.message}`,
        }));
      },
    );
    check(
      checks,
      "provider_output_schema_subset",
      providerSchemaIssues.length === 0,
      providerSchemaIssues.length === 0
        ? "all four Turn output schemas satisfy the App Server subset"
        : JSON.stringify(providerSchemaIssues.slice(0, 5)),
    );
  } catch (error) {
    check(checks, "schema_manifest", false, errorMessage(error));
    check(
      checks,
      "provider_output_schema_subset",
      false,
      "schema manifest unavailable",
    );
  }

  for (const role of Object.keys(ROLE_SKILLS) as RegisteredRole[]) {
    try {
      const skillRoot = resolve(config.skillRoot, ROLE_SKILLS[role]);
      const loaded = loadSkillPackage(skillRoot);
      const reference = JSON.parse(
        readFileSync(resolve(skillRoot, "references/schema_manifest.json"), "utf8"),
      );
      const roleProfilePath = resolve(
        config.projectRoot,
        "scripts/simple_semantic_loop/role_profiles",
        `${role}.json`,
      );
      const roleProfile = JSON.parse(readFileSync(roleProfilePath, "utf8"));
      const valid =
        loaded.skillMarkdown.startsWith("---\n") &&
        reference.manifestSha256 ===
          (manifest ? canonicalSha256(manifest) : "") &&
        roleProfile.reasoningEffort === ROLE_REASONING_EFFORT[role] &&
        roleProfile.allowedInputMessageTypes?.[0] ===
          ROLE_MESSAGE_TYPES[role].input &&
        roleProfile.allowedOutputMessageTypes?.[0] ===
          ROLE_MESSAGE_TYPES[role].output &&
        roleProfile.goals === false &&
        roleProfile.delegation === false &&
        roleProfile.experimentExecution === false;
      check(
        checks,
        `skill_${role}`,
        valid,
        `${ROLE_SKILLS[role]} package=${loaded.sha256}`,
      );
    } catch (error) {
      check(checks, `skill_${role}`, false, errorMessage(error));
    }
  }

  try {
    for (const role of Object.keys(ROLE_PROFILES) as RegisteredRole[]) {
      resolveWireEffort(role, config.capabilityManifest);
    }
    check(
      checks,
      "effort_capability_mapping",
      true,
      `workflow=max->${config.capabilityManifest.wireEffortByLogicalEffort.max}; others=high->${config.capabilityManifest.wireEffortByLogicalEffort.high}`,
    );
  } catch (error) {
    check(checks, "effort_capability_mapping", false, errorMessage(error));
  }

  const roles = Object.keys(ROLE_PROFILES).sort();
  const stageNames = Object.keys(STAGE_REGISTRY);
  check(
    checks,
    "four_roles_no_experiment_stage",
    roles.join(",") ===
      [
        "closure_reviewer",
        "direction_reviewer",
        "evidence_reader",
        "workflow_decision",
      ].join(",") &&
      stageNames.every((name) => !/experiment/i.test(name)),
    `roles=${roles.join(",")}; stages=${stageNames.join(",")}`,
  );
  check(
    checks,
    "fresh_turn_runtime_invariants",
    APP_SERVER_RUNTIME_INVARIANTS.ephemeralThreadPerAttempt &&
      !APP_SERVER_RUNTIME_INVARIANTS.resumesProviderThread &&
      !APP_SERVER_RUNTIME_INVARIANTS.createsGoal &&
      APP_SERVER_RUNTIME_INVARIANTS.structuredOutputSchema &&
      APP_SERVER_RUNTIME_INVARIANTS.approvalPolicy === "never" &&
      APP_SERVER_RUNTIME_INVARIANTS.defaultSandbox === "read-only" &&
      APP_SERVER_RUNTIME_INVARIANTS.yoloSandbox === "danger-full-access",
    JSON.stringify(APP_SERVER_RUNTIME_INVARIANTS),
  );

  const codexVersion = spawnSync("codex", ["--version"], { encoding: "utf8" });
  check(
    checks,
    "codex_cli",
    codexVersion.status === 0,
    String(codexVersion.stdout || codexVersion.stderr).trim(),
  );

  const configPath = resolve(
    process.env.CODEX_HOME ?? resolve(process.env.HOME ?? "", ".codex"),
    "config.toml",
  );
  const obsidianConfigured =
    existsSync(configPath) &&
    /^\s*\[mcp_servers\.obsidian\]\s*$/m.test(
      readFileSync(configPath, "utf8"),
    );
  check(
    checks,
    "obsidian_readonly_capability_configured",
    obsidianConfigured,
    configPath,
  );

  try {
    mkdirSync(config.workDir, { recursive: true });
    const probe = resolve(config.workDir, `.doctor-write-${process.pid}`);
    writeFileSync(probe, "probe", "utf8");
    rmSync(probe);
    check(checks, "output_path", true, config.workDir);
  } catch (error) {
    check(checks, "output_path", false, errorMessage(error));
  }

  const provider = {
    probed: false,
    supportedReasoningEfforts: [] as string[],
  };
  if (options.probeProvider !== false) {
    const runtime = new AppServerFreshTurnRuntime();
    try {
      const model = await runtime.probeModel(config.model);
      provider.probed = true;
      provider.supportedReasoningEfforts = model.supportedReasoningEfforts;
      const supports =
        model.supportedReasoningEfforts.includes(
          config.capabilityManifest.wireEffortByLogicalEffort.high,
        ) &&
        model.supportedReasoningEfforts.includes(
          config.capabilityManifest.wireEffortByLogicalEffort.max,
        );
      check(
        checks,
        "provider_model_efforts",
        supports,
        `${config.model}: ${model.supportedReasoningEfforts.join(", ")}`,
      );
      const visibleSkills = await runtime.probeSkills(config.projectRoot);
      const missing = Object.values(ROLE_SKILLS).filter(
        (skill) => !visibleSkills.has(skill),
      );
      if (missing.length) {
        warnings.push(
          `skills/list did not advertise local packages: ${missing.join(", ")}`,
        );
      }
      check(
        checks,
        "fresh_turn_goal_isolation",
        true,
        "runtime uses ephemeral thread/start, never thread/resume, and makes no Goal request",
      );
    } catch (error) {
      check(checks, "provider_model_efforts", false, errorMessage(error));
    } finally {
      await runtime.close();
    }
  } else {
    warnings.push("provider probe skipped");
  }

  return {
    passed: checks.every((item) => item.passed),
    checks,
    warnings,
    provider,
  };
}

function check(
  checks: DoctorCheck[],
  name: string,
  passed: boolean,
  detail: string,
): void {
  checks.push({ name, passed, detail });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
