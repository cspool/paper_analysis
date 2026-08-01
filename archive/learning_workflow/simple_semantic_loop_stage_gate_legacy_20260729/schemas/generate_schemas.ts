#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { canonicalSha256 } from "../contracts/index.ts";
import {
  publishableSchema,
  SCHEMA_DEFINITIONS,
  SCHEMA_FILENAMES,
  type SchemaName,
} from "./schema_definitions.ts";

const here = dirname(fileURLToPath(import.meta.url));
mkdirSync(here, { recursive: true });

const manifestEntries: Record<string, { path: string; sha256: string }> = {};
for (const name of Object.keys(SCHEMA_DEFINITIONS) as SchemaName[]) {
  const schema = publishableSchema(name);
  const filename = SCHEMA_FILENAMES[name];
  writeFileSync(
    resolve(here, filename),
    `${JSON.stringify(schema, null, 2)}\n`,
    "utf8",
  );
  manifestEntries[name] = {
    path: filename,
    sha256: canonicalSha256(schema),
  };
}

const manifest = {
  protocolVersion: 1,
  schemas: manifestEntries,
};
writeFileSync(
  resolve(here, "schema_manifest.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
  "utf8",
);

process.stdout.write(
  `${Object.keys(manifestEntries).length} schemas generated; manifest=${canonicalSha256(manifest)}\n`,
);

