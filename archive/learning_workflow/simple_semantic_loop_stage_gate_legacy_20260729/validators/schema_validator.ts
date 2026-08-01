import type { JsonSchema, SchemaName } from "../schemas/schema_definitions.ts";
import { SCHEMA_DEFINITIONS } from "../schemas/schema_definitions.ts";
import type { ValidationReport } from "../contracts/index.ts";
export type { ValidationError, ValidationReport } from "../contracts/index.ts";

const VALIDATOR_VERSION = "simple-semantic-loop-validator/1";

export function emptyReport(): ValidationReport {
  return {
    validatorVersion: VALIDATOR_VERSION,
    valid: true,
    errors: [],
    checkedArtifactHashes: [],
    checkedObjectRefs: [],
  };
}

export function addError(
  report: ValidationReport,
  code: string,
  jsonPointer: string | null,
  message: string,
): void {
  report.valid = false;
  report.errors.push({ code, jsonPointer, message });
}

export function mergeReports(
  ...reports: readonly ValidationReport[]
): ValidationReport {
  const merged = emptyReport();
  for (const report of reports) {
    merged.errors.push(...report.errors);
    merged.checkedArtifactHashes.push(...report.checkedArtifactHashes);
    merged.checkedObjectRefs.push(...report.checkedObjectRefs);
  }
  merged.valid = merged.errors.length === 0;
  return merged;
}

export function validateSchema(
  name: SchemaName,
  value: unknown,
): ValidationReport {
  const report = emptyReport();
  validateNode(SCHEMA_DEFINITIONS[name], value, "", report);
  return report;
}

function escapePointer(value: string): string {
  return value.replaceAll("~", "~0").replaceAll("/", "~1");
}

function pointer(base: string, child: string | number): string {
  return `${base}/${escapePointer(String(child))}`;
}

function matchesType(type: string, value: unknown): boolean {
  switch (type) {
    case "null":
      return value === null;
    case "array":
      return Array.isArray(value);
    case "object":
      return value !== null && typeof value === "object" && !Array.isArray(value);
    case "integer":
      return typeof value === "number" && Number.isInteger(value);
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "string":
      return typeof value === "string";
    case "boolean":
      return typeof value === "boolean";
    default:
      return false;
  }
}

function isolatedMatch(schema: JsonSchema, value: unknown): ValidationReport {
  const report = emptyReport();
  validateNode(schema, value, "", report);
  return report;
}

function validateNode(
  schema: JsonSchema,
  value: unknown,
  at: string,
  report: ValidationReport,
): void {
  if ("const" in schema && value !== schema.const) {
    addError(
      report,
      "schema.const",
      at || "/",
      `must equal ${JSON.stringify(schema.const)}`,
    );
    return;
  }
  if (Array.isArray(schema.enum) && !schema.enum.some((entry) => entry === value)) {
    addError(report, "schema.enum", at || "/", "value is not in the closed enum");
    return;
  }
  if (Array.isArray(schema.oneOf)) {
    const matches = (schema.oneOf as JsonSchema[]).filter(
      (entry) => isolatedMatch(entry, value).valid,
    );
    if (matches.length !== 1) {
      addError(
        report,
        "schema.oneOf",
        at || "/",
        `must match exactly one branch; matched ${matches.length}`,
      );
      return;
    }
    validateNode(matches[0]!, value, at, report);
    return;
  }
  if (Array.isArray(schema.anyOf)) {
    const matches = (schema.anyOf as JsonSchema[]).filter(
      (entry) => isolatedMatch(entry, value).valid,
    );
    if (matches.length === 0) {
      addError(report, "schema.anyOf", at || "/", "must match at least one branch");
      return;
    }
    validateNode(matches[0]!, value, at, report);
    return;
  }
  if (Array.isArray(schema.allOf)) {
    for (const entry of schema.allOf as JsonSchema[]) {
      validateNode(entry, value, at, report);
    }
  }

  const declaredTypes =
    typeof schema.type === "string"
      ? [schema.type]
      : Array.isArray(schema.type)
        ? (schema.type as string[])
        : [];
  if (
    declaredTypes.length > 0 &&
    !declaredTypes.some((type) => matchesType(type, value))
  ) {
    addError(
      report,
      "schema.type",
      at || "/",
      `expected ${declaredTypes.join("|")}`,
    );
    return;
  }

  if (typeof value === "string") {
    if (typeof schema.minLength === "number" && value.length < schema.minLength) {
      addError(report, "schema.minLength", at || "/", "string is too short");
    }
    if (typeof schema.maxLength === "number" && value.length > schema.maxLength) {
      addError(report, "schema.maxLength", at || "/", "string is too long");
    }
    if (
      typeof schema.pattern === "string" &&
      !new RegExp(schema.pattern).test(value)
    ) {
      addError(report, "schema.pattern", at || "/", "string does not match pattern");
    }
  }

  if (typeof value === "number") {
    if (typeof schema.minimum === "number" && value < schema.minimum) {
      addError(report, "schema.minimum", at || "/", "number is below minimum");
    }
    if (typeof schema.maximum === "number" && value > schema.maximum) {
      addError(report, "schema.maximum", at || "/", "number is above maximum");
    }
  }

  if (Array.isArray(value)) {
    if (typeof schema.minItems === "number" && value.length < schema.minItems) {
      addError(report, "schema.minItems", at || "/", "array is too short");
    }
    if (typeof schema.maxItems === "number" && value.length > schema.maxItems) {
      addError(report, "schema.maxItems", at || "/", "array is too long");
    }
    if (schema.uniqueItems === true) {
      const encoded = value.map((entry) => JSON.stringify(entry));
      if (new Set(encoded).size !== encoded.length) {
        addError(report, "schema.uniqueItems", at || "/", "array items must be unique");
      }
    }
    if (schema.items && typeof schema.items === "object") {
      value.forEach((entry, index) =>
        validateNode(schema.items as JsonSchema, entry, pointer(at, index), report),
      );
    }
  }

  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    const properties =
      schema.properties && typeof schema.properties === "object"
        ? (schema.properties as Record<string, JsonSchema>)
        : {};
    if (Array.isArray(schema.required)) {
      for (const key of schema.required as string[]) {
        if (!Object.hasOwn(record, key)) {
          addError(
            report,
            "schema.required",
            pointer(at, key),
            "required property is missing",
          );
        }
      }
    }
    for (const [key, child] of Object.entries(record)) {
      if (properties[key]) {
        validateNode(properties[key], child, pointer(at, key), report);
      } else if (schema.additionalProperties === false) {
        addError(
          report,
          "schema.additionalProperties",
          pointer(at, key),
          "unknown property is not allowed",
        );
      } else if (
        schema.additionalProperties &&
        typeof schema.additionalProperties === "object"
      ) {
        validateNode(
          schema.additionalProperties as JsonSchema,
          child,
          pointer(at, key),
          report,
        );
      }
    }
  }
}
