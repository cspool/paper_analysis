import {
  LOOP_DECISIONS,
  type DecisionProtocolResult,
  type LoopDecision,
  type OutputError,
} from "./types.ts";

export interface JsonParseResult {
  parsed: Record<string, unknown> | null;
  errors: OutputError[];
}

export function parseStrictJsonObject(raw: string): JsonParseResult {
  const text = raw.replace(/^\uFEFF/, "").trim();
  if (!text) {
    return {
      parsed: null,
      errors: [jsonError("/", "output is empty")],
    };
  }

  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    return {
      parsed: null,
      errors: [
        jsonError(
          "/",
          error instanceof Error ? error.message : "output is not valid JSON",
        ),
      ],
    };
  }
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return {
      parsed: null,
      errors: [jsonError("/", "output must be exactly one JSON object")],
    };
  }
  return { parsed: value as Record<string, unknown>, errors: [] };
}

export function parseDecisionProtocol(
  raw: string,
  allowed: LoopDecision[],
): { result: DecisionProtocolResult | null; errors: OutputError[] } {
  const decisionValues: string[] = [];
  const guidanceValues: string[] = [];

  for (const sourceLine of raw.replace(/\r\n?/g, "\n").split("\n")) {
    const line = sourceLine.trim();
    if (!line) continue;
    const decision = line.match(
      /^(?:[-*]\s*)?decision\s*[:=]\s*([A-Za-z_]+)\s*$/i,
    );
    if (decision) {
      decisionValues.push(decision[1]!.toUpperCase());
      continue;
    }
    const guidance = line.match(
      /^(?:[-*]\s*)?guidance\s*[:=]\s*(.*?)\s*$/i,
    );
    if (guidance) {
      guidanceValues.push(guidance[1] ?? "");
      continue;
    }
  }

  const errors: OutputError[] = [];
  if (decisionValues.length !== 1) {
    errors.push(protocolError(
      "/decision",
      "output must contain exactly one decision field",
    ));
  }
  if (errors.length > 0) return { result: null, errors };

  const rawDecision = decisionValues[0]!;
  if (!LOOP_DECISIONS.includes(rawDecision as LoopDecision)) {
    return {
      result: null,
      errors: [protocolError(
        "/decision",
        `unknown decision literal ${rawDecision}`,
      )],
    };
  }
  const decision = rawDecision as LoopDecision;
  if (!allowed.includes(decision)) {
    return {
      result: null,
      errors: [protocolError(
        "/decision",
        `${decision} is not in the allowed decision set`,
      )],
    };
  }

  const guidance =
    guidanceValues.map((value) => value.trim()).filter(Boolean).join("\n") ||
    null;

  if (decision === "RUN_EXP_GOAL" && guidance === null) {
    return {
      result: null,
      errors: [protocolError(
        "/guidance",
        "RUN_EXP_GOAL requires a non-empty experiment objective in guidance",
      )],
    };
  }

  return { result: { decision, guidance }, errors: [] };
}

function jsonError(path: string, message: string): OutputError {
  return { check: "JSON_PARSE", path, message };
}

function protocolError(path: string, message: string): OutputError {
  return { check: "DECISION_PROTOCOL", path, message };
}
