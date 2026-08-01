import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { canonicalSha256 } from "../contracts/index.ts";
import {
  publishableSchema,
  SCHEMA_DEFINITIONS,
  SCHEMA_FILENAMES,
  type SchemaName,
} from "../schemas/schema_definitions.ts";
import { validateProviderOutputSchema } from "../schemas/provider_schema_validator.ts";
import { ROLE_MESSAGE_TYPES } from "../contracts/index.ts";

const schemaDir = resolve(dirname(fileURLToPath(import.meta.url)), "../schemas");

test("published schemas and manifest are generated from runtime definitions", () => {
  const manifest = JSON.parse(
    readFileSync(resolve(schemaDir, "schema_manifest.json"), "utf8"),
  ) as {
    protocolVersion: number;
    schemas: Record<string, { path: string; sha256: string }>;
  };
  assert.equal(manifest.protocolVersion, 1);
  assert.deepEqual(
    Object.keys(manifest.schemas).sort(),
    Object.keys(SCHEMA_DEFINITIONS).sort(),
  );
  for (const name of Object.keys(SCHEMA_DEFINITIONS) as SchemaName[]) {
    const published = JSON.parse(
      readFileSync(resolve(schemaDir, SCHEMA_FILENAMES[name]), "utf8"),
    );
    assert.deepEqual(published, publishableSchema(name));
    assert.equal(manifest.schemas[name]!.path, SCHEMA_FILENAMES[name]);
    assert.equal(manifest.schemas[name]!.sha256, canonicalSha256(published));
  }
});

test("all four Turn outputs stay inside the provider structured-output subset", () => {
  for (const { output } of Object.values(ROLE_MESSAGE_TYPES)) {
    const schema = publishableSchema(output);
    assert.deepEqual(
      validateProviderOutputSchema(schema),
      [],
      `${output} is not provider-compatible`,
    );
  }

  assert.ok(
    validateProviderOutputSchema({
      type: "object",
      properties: {
        values: { type: "array" },
        mode: { const: "bad" },
      },
      required: ["values"],
      uniqueItems: true,
    }).length >= 4,
  );
});
