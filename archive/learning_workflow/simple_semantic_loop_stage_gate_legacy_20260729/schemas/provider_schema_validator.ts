export interface ProviderSchemaIssue {
  jsonPointer: string;
  message: string;
}

/**
 * Validate the conservative JSON-Schema subset exercised by Codex App
 * Server structured output. This is intentionally local and deterministic:
 * provider discovery still verifies model/effort availability, while this
 * check catches schema regressions before a Turn is launched.
 */
export function validateProviderOutputSchema(
  schema: unknown,
): ProviderSchemaIssue[] {
  const issues: ProviderSchemaIssue[] = [];
  walk(schema, "", issues);
  return issues;
}

function walk(
  value: unknown,
  pointer: string,
  issues: ProviderSchemaIssue[],
): void {
  if (!isRecord(value)) return;

  if (Object.hasOwn(value, "oneOf")) {
    issue(issues, pointer, "oneOf is unsupported; use anyOf");
  }
  if (Object.hasOwn(value, "uniqueItems")) {
    issue(
      issues,
      pointer,
      "uniqueItems is unsupported in provider structured output",
    );
  }
  if (
    (Object.hasOwn(value, "enum") || Object.hasOwn(value, "const")) &&
    typeof value.type !== "string"
  ) {
    issue(
      issues,
      pointer,
      "enum and const schemas require an explicit scalar type",
    );
  }
  if (value.type === "array" && !isRecord(value.items)) {
    issue(issues, pointer, "array schemas require an items schema");
  }
  if (value.type === "object") {
    if (!isRecord(value.properties)) {
      issue(issues, pointer, "object schemas require properties");
    } else {
      const propertyNames = Object.keys(value.properties);
      const required = Array.isArray(value.required)
        ? value.required.filter(
            (item): item is string => typeof item === "string",
          )
        : [];
      const missing = propertyNames.filter(
        (property) => !required.includes(property),
      );
      if (missing.length > 0) {
        issue(
          issues,
          pointer,
          `all object properties must be required: ${missing.join(", ")}`,
        );
      }
      for (const [property, child] of Object.entries(value.properties)) {
        walk(child, `${pointer}/properties/${escapePointer(property)}`, issues);
      }
    }
    if (value.additionalProperties !== false) {
      issue(
        issues,
        pointer,
        "object schemas require additionalProperties=false",
      );
    }
  }

  if (Array.isArray(value.anyOf)) {
    value.anyOf.forEach((child, index) => {
      walk(child, `${pointer}/anyOf/${index}`, issues);
    });
  }
  if (isRecord(value.items)) {
    walk(value.items, `${pointer}/items`, issues);
  }
}

function issue(
  issues: ProviderSchemaIssue[],
  pointer: string,
  message: string,
): void {
  issues.push({ jsonPointer: pointer || "/", message });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function escapePointer(value: string): string {
  return value.replaceAll("~", "~0").replaceAll("/", "~1");
}
