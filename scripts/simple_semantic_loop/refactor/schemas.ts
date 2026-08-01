import {
  REVIEW_VERDICTS,
  WORK_OUTCOMES,
  type CoreControlProjection,
  type OutputError,
  type TaskAction,
} from "./types.ts";

type JsonSchema = Record<string, unknown>;

export interface ReferenceTemplateAdvisory {
  path: string;
  message: string;
}

export interface CoreOutputValidation {
  control: CoreControlProjection | null;
  errors: OutputError[];
}

const nonEmptyString: JsonSchema = { type: "string", minLength: 1 };
const nullableNonEmptyString: JsonSchema = {
  anyOf: [nonEmptyString, { type: "null" }],
};
const stringArray = (minItems = 0): JsonSchema => ({
  type: "array",
  items: nonEmptyString,
  ...(minItems > 0 ? { minItems } : {}),
});

const evidenceTemplate: JsonSchema = {
  type: "array",
  items: {
    type: "object",
    properties: {
      sourceRef: nonEmptyString,
      supports: nonEmptyString,
    },
    required: ["sourceRef", "supports"],
    additionalProperties: false,
  },
};

const anchorContentTemplate: JsonSchema = {
  type: "object",
  properties: {
    name: nonEmptyString,
    scenario: nonEmptyString,
    baseline: nonEmptyString,
    performanceTension: nonEmptyString,
    scope6L: {
      type: "object",
      properties: {
        L1: nullableNonEmptyString,
        L2: nullableNonEmptyString,
        L3: nullableNonEmptyString,
        L4: nullableNonEmptyString,
        L5: nullableNonEmptyString,
        L6: nullableNonEmptyString,
      },
      required: ["L1", "L2", "L3", "L4", "L5", "L6"],
      additionalProperties: false,
    },
    constraints: stringArray(),
  },
  required: [
    "name",
    "scenario",
    "baseline",
    "performanceTension",
    "scope6L",
    "constraints",
  ],
  additionalProperties: false,
};

const directionContentTemplate: JsonSchema = {
  type: "object",
  properties: {
    name: nonEmptyString,
    mechanism: nonEmptyString,
    baselineChange: nonEmptyString,
    expectedEffects: {
      type: "array",
      minItems: 1,
      items: {
        type: "object",
        properties: {
          metric: nonEmptyString,
          effect: nonEmptyString,
          conditions: nonEmptyString,
        },
        required: ["metric", "effect", "conditions"],
        additionalProperties: false,
      },
    },
    tradeoffs: stringArray(),
    failureConditions: stringArray(),
    measurementPlan: stringArray(1),
  },
  required: [
    "name",
    "mechanism",
    "baselineChange",
    "expectedEffects",
    "tradeoffs",
    "failureConditions",
    "measurementPlan",
  ],
  additionalProperties: false,
};

function workResultTemplate(content: JsonSchema): JsonSchema {
  return {
    type: "object",
    properties: {
      workOutcome: {
        type: "string",
        enum: [...WORK_OUTCOMES],
      },
      content: {
        anyOf: [content, { type: "null" }],
      },
      evidence: evidenceTemplate,
      unresolved: stringArray(),
    },
    required: ["workOutcome", "content", "evidence", "unresolved"],
    additionalProperties: false,
  };
}

export const REFERENCE_TEMPLATE_SCHEMAS = Object.freeze({
  "work-result-anchor-v2": workResultTemplate(anchorContentTemplate),
  "work-result-direction-v2": workResultTemplate(directionContentTemplate),
  "review-result-v2": {
    type: "object",
    properties: {
      reviewVerdict: {
        type: "string",
        enum: [...REVIEW_VERDICTS],
      },
      summary: nonEmptyString,
      findings: {
        type: "array",
        items: {
          type: "object",
          properties: {
            severity: {
              type: "string",
              enum: ["BLOCKING", "NON_BLOCKING"],
            },
            issue: nonEmptyString,
            basis: nonEmptyString,
            expected: nonEmptyString,
          },
          required: ["severity", "issue", "basis", "expected"],
          additionalProperties: false,
        },
      },
      queryGaps: {
        type: "array",
        items: {
          type: "object",
          properties: {
            question: nonEmptyString,
            dimension: {
              type: "string",
              enum: ["experiment", "idea", "knowledge", "human"],
            },
            reason: nonEmptyString,
          },
          required: ["question", "dimension", "reason"],
          additionalProperties: false,
        },
      },
    },
    required: ["reviewVerdict", "summary", "findings", "queryGaps"],
    additionalProperties: false,
  } satisfies JsonSchema,
});

export function validateCoreOutputForAction(
  action: TaskAction,
  value: unknown,
): CoreOutputValidation {
  if (!isRecord(value)) {
    return {
      control: null,
      errors: [coreError("/", "output must be a JSON object")],
    };
  }
  if (action === "REVIEW_ANCHOR" || action === "REVIEW_DIRECTION") {
    const verdict = value.reviewVerdict;
    if (
      typeof verdict !== "string" ||
      !REVIEW_VERDICTS.includes(
        verdict as (typeof REVIEW_VERDICTS)[number],
      )
    ) {
      return {
        control: null,
        errors: [
          coreError(
            "/reviewVerdict",
            `expected one of ${REVIEW_VERDICTS.join(", ")}`,
          ),
        ],
      };
    }
    return {
      control: {
        role: "REVIEWER",
        reviewVerdict: verdict as (typeof REVIEW_VERDICTS)[number],
      },
      errors: [],
    };
  }

  const outcome = value.workOutcome;
  if (
    typeof outcome !== "string" ||
    !WORK_OUTCOMES.includes(outcome as (typeof WORK_OUTCOMES)[number])
  ) {
    return {
      control: null,
      errors: [
        coreError(
          "/workOutcome",
          `expected one of ${WORK_OUTCOMES.join(", ")}`,
        ),
      ],
    };
  }
  return {
    control: {
      role: "WORKER",
      workOutcome: outcome as (typeof WORK_OUTCOMES)[number],
    },
    errors: [],
  };
}

export function referenceTemplateForAction(action: TaskAction): JsonSchema {
  if (action === "CREATE_ANCHOR" || action === "DEEPEN_ANCHOR") {
    return REFERENCE_TEMPLATE_SCHEMAS["work-result-anchor-v2"];
  }
  if (action === "CREATE_DIRECTION" || action === "DEEPEN_DIRECTION") {
    return REFERENCE_TEMPLATE_SCHEMAS["work-result-direction-v2"];
  }
  return REFERENCE_TEMPLATE_SCHEMAS["review-result-v2"];
}

export function validateReferenceTemplateForAction(
  action: TaskAction,
  value: unknown,
): ReferenceTemplateAdvisory[] {
  return validateJsonSchema(referenceTemplateForAction(action), value);
}

export function validateJsonSchema(
  schema: JsonSchema,
  value: unknown,
): ReferenceTemplateAdvisory[] {
  const advisories: ReferenceTemplateAdvisory[] = [];
  validateNode(schema, value, "", advisories);
  return advisories;
}

function validateNode(
  schema: JsonSchema,
  value: unknown,
  path: string,
  advisories: ReferenceTemplateAdvisory[],
): void {
  if (Array.isArray(schema.anyOf)) {
    const variants = schema.anyOf as JsonSchema[];
    const variantAdvisories = variants.map((variant) => {
      const nested: ReferenceTemplateAdvisory[] = [];
      validateNode(variant, value, path, nested);
      return nested;
    });
    if (variantAdvisories.every((nested) => nested.length > 0)) {
      advisories.push({
        path: path || "/",
        message: "value does not match any reference-template variant",
      });
    }
    return;
  }

  const type = typeof schema.type === "string" ? schema.type : null;
  if (type && !matchesType(type, value)) {
    advisories.push({
      path: path || "/",
      message: `reference template expects ${type}`,
    });
    return;
  }

  if (Array.isArray(schema.enum) && !schema.enum.includes(value)) {
    advisories.push({
      path: path || "/",
      message: `reference template expects one of ${schema.enum.join(", ")}`,
    });
  }

  if (typeof value === "string") {
    if (
      typeof schema.minLength === "number" &&
      value.length < schema.minLength
    ) {
      advisories.push({
        path: path || "/",
        message: `reference-template minimum length is ${schema.minLength}`,
      });
    }
    return;
  }

  if (Array.isArray(value)) {
    if (
      typeof schema.minItems === "number" &&
      value.length < schema.minItems
    ) {
      advisories.push({
        path: path || "/",
        message:
          `reference-template minimum item count is ${schema.minItems}`,
      });
    }
    if (isRecord(schema.items)) {
      value.forEach((item, index) =>
        validateNode(
          schema.items as JsonSchema,
          item,
          `${path}/${index}`,
          advisories,
        )
      );
    }
    return;
  }

  if (isRecord(value)) {
    const properties = isRecord(schema.properties)
      ? schema.properties
      : {};
    const required = Array.isArray(schema.required)
      ? schema.required.filter((item): item is string =>
        typeof item === "string"
      )
      : [];
    for (const key of required) {
      if (!Object.hasOwn(value, key)) {
        advisories.push({
          path: `${path}/${escapePointer(key)}`,
          message: "reference-template property is missing",
        });
      }
    }
    if (schema.additionalProperties === false) {
      for (const key of Object.keys(value)) {
        if (!Object.hasOwn(properties, key)) {
          advisories.push({
            path: `${path}/${escapePointer(key)}`,
            message: "field is outside the reference template",
          });
        }
      }
    }
    for (const [key, child] of Object.entries(properties)) {
      if (Object.hasOwn(value, key) && isRecord(child)) {
        validateNode(
          child,
          value[key],
          `${path}/${escapePointer(key)}`,
          advisories,
        );
      }
    }
  }
}

function matchesType(type: string, value: unknown): boolean {
  switch (type) {
    case "object":
      return isRecord(value);
    case "array":
      return Array.isArray(value);
    case "string":
      return typeof value === "string";
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "integer":
      return Number.isInteger(value);
    case "boolean":
      return typeof value === "boolean";
    case "null":
      return value === null;
    default:
      return true;
  }
}

function coreError(path: string, message: string): OutputError {
  return { check: "CORE_CONTROL", path, message };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function escapePointer(value: string): string {
  return value.replaceAll("~", "~0").replaceAll("/", "~1");
}
