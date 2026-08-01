import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { REFERENCE_TEMPLATE_SCHEMAS } from "./schemas.ts";

const refactorDir = dirname(fileURLToPath(import.meta.url));
const schemaDir = resolve(refactorDir, "../schemas");
mkdirSync(schemaDir, { recursive: true });

const filenames = {
  "work-result-anchor-v2": "work_result_anchor_v2.schema.json",
  "work-result-direction-v2": "work_result_direction_v2.schema.json",
  "review-result-v2": "review_result_v2.schema.json",
} as const;

for (const [name, schema] of Object.entries(REFERENCE_TEMPLATE_SCHEMAS)) {
  writeFileSync(
    resolve(schemaDir, filenames[name as keyof typeof filenames]),
    `${JSON.stringify(schema, null, 2)}\n`,
    "utf8",
  );
}
