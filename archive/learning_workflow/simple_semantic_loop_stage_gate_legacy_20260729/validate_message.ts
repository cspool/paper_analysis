#!/usr/bin/env node
import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";
import {
  deriveClosureFacts,
  validateClosureReview,
  validateEvidencePacket,
  validateReviewDelta,
  validateSchema,
  validateWorkflowDecisionProposal,
} from "./validators/index.ts";

function load(path: string | undefined, label: string): unknown {
  if (!path) throw new Error(`missing --${label}`);
  return JSON.parse(readFileSync(path, "utf8"));
}

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    task: { type: "string" },
    packet: { type: "string" },
    review: { type: "string" },
    proposal: { type: "string" },
    message: { type: "string" },
    schema: { type: "string" },
    pretty: { type: "boolean", default: false },
  },
});

const kind = positionals[0];
let output: unknown;
let valid = false;
try {
  switch (kind) {
    case "evidence":
      output = validateEvidencePacket(
        load(values.packet, "packet") as never,
        load(values.task, "task") as never,
      );
      valid = (output as { valid: boolean }).valid;
      break;
    case "evidence-task":
      output = validateSchema(
        "EVIDENCE_READER_TASK",
        load(values.task, "task"),
      );
      valid = (output as { valid: boolean }).valid;
      break;
    case "direction":
      output = validateReviewDelta(
        load(values.review, "review") as never,
        load(values.task, "task") as never,
      );
      valid = (output as { valid: boolean }).valid;
      break;
    case "closure":
      output = validateClosureReview(
        load(values.review, "review") as never,
        load(values.task, "task") as never,
      );
      valid = (output as { valid: boolean }).valid;
      break;
    case "closure-derive": {
      output = deriveClosureFacts(load(values.task, "task") as never);
      valid = true;
      break;
    }
    case "workflow":
      output = validateWorkflowDecisionProposal(
        load(values.proposal, "proposal") as never,
        load(values.task, "task") as never,
      );
      valid = (output as { valid: boolean }).valid;
      break;
    case "schema": {
      const schemaName = values.schema;
      if (!schemaName) throw new Error("missing --schema");
      output = validateSchema(
        schemaName as never,
        load(values.message, "message"),
      );
      valid = (output as { valid: boolean }).valid;
      break;
    }
    default:
      throw new Error(
        "usage: validate_message.ts evidence|evidence-task|direction|closure|closure-derive|workflow|schema",
      );
  }
} catch (error) {
  output = {
    validatorVersion: "simple-semantic-loop-validator/1",
    valid: false,
    errors: [
      {
        code: "validator.invocation",
        jsonPointer: null,
        message: error instanceof Error ? error.message : String(error),
      },
    ],
    checkedArtifactHashes: [],
    checkedObjectRefs: [],
  };
  valid = false;
}

process.stdout.write(
  `${JSON.stringify(output, null, values.pretty ? 2 : undefined)}\n`,
);
process.exitCode = valid ? 0 : 1;

