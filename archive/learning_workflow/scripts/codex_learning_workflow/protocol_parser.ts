import { roleProfile } from "./role_profiles.ts";
import type { JsonValue, ParsedProtocol, Role } from "./types.ts";

const MAIN_MARKERS = new Set([
  "ANCHOR_ROUND_PLAN",
  "ANCHOR_STAGE_COMPLETE",
  "ANCHOR_EVIDENCE_RESULT",
  "ANCHOR_DELTA",
  "DIRECTION_PROPOSAL",
  "DIRECTION_PLANNING_COMPLETE",
  "REVIEW_QUESTION",
  "REVIEW_REFERENCE_REQUEST",
  "DIRECTION_REVIEW_COMPLETE",
  "REVIEW_EVIDENCE_RESULT",
]);

const NESTED_BLOCKS = [
  "SEMANTIC_PAYLOAD",
  "CLAIMS",
  "GAPS",
  "SOURCES",
  "ANSWER",
] as const;

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function extractBlock(text: string, name: string): string | null {
  const expression = new RegExp(
    `___${escapeRegExp(name)}_START___\\s*([\\s\\S]*?)\\s*___${escapeRegExp(name)}_END___`,
  );
  return expression.exec(text)?.[1]?.trim() ?? null;
}

function parseJsonBlock(text: string, label: string): JsonValue {
  try {
    return JSON.parse(text) as JsonValue;
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function removeNestedBlocks(text: string): string {
  let output = text;
  for (const name of NESTED_BLOCKS) {
    const expression = new RegExp(
      `___${name}_START___[\\s\\S]*?___${name}_END___`,
      "g",
    );
    output = output.replace(expression, "");
  }
  return output;
}

function parseControlFields(mainBody: string): Record<string, string> {
  const control: Record<string, string> = {};
  const withoutNested = removeNestedBlocks(mainBody);
  for (const line of withoutNested.split(/\r?\n/)) {
    const match = /^\s*([a-z][a-z0-9_]*)\s*:\s*(.*?)\s*$/.exec(line);
    if (match) {
      control[match[1]] = match[2];
    }
  }
  return control;
}

function findMainMarkerStarts(text: string): string[] {
  const markers: string[] = [];
  const expression = /___([A-Z][A-Z0-9_]*)_START___/g;
  for (const match of text.matchAll(expression)) {
    if (MAIN_MARKERS.has(match[1])) {
      markers.push(match[1]);
    }
  }
  return markers;
}

function parseLoop(text: string): string | null {
  const matches = [...text.matchAll(/\[LOOP:\s*([^\]\r\n]+)\]/g)];
  if (matches.length > 1) {
    throw new Error("multiple LOOP markers");
  }
  return matches[0]?.[1]?.trim() ?? null;
}

export function parseProtocol(role: Role, text: string, skillRoot = "/unused"): ParsedProtocol {
  if (!text.trim()) {
    throw new Error("agent output is empty");
  }
  const profile = roleProfile(role, skillRoot);
  const mainStarts = findMainMarkerStarts(text);
  if (mainStarts.length !== 1) {
    throw new Error(`expected exactly one main marker, found ${mainStarts.length}: ${mainStarts.join(", ")}`);
  }
  const marker = mainStarts[0];
  if (!profile.allowedMainMarkers.includes(marker)) {
    throw new Error(`marker ${marker} is not allowed for role ${role}`);
  }
  const mainBody = extractBlock(text, marker);
  if (mainBody === null) {
    throw new Error(`missing or malformed end marker for ${marker}`);
  }

  const textBlocks: Record<string, string> = {};
  for (const nested of NESTED_BLOCKS) {
    const block = extractBlock(mainBody, nested);
    if (block !== null) {
      textBlocks[nested] = block;
    }
  }

  let payload: JsonValue | null = null;
  if (textBlocks.SEMANTIC_PAYLOAD !== undefined) {
    payload = parseJsonBlock(textBlocks.SEMANTIC_PAYLOAD, "SEMANTIC_PAYLOAD");
  } else if (textBlocks.CLAIMS !== undefined) {
    payload = parseJsonBlock(textBlocks.CLAIMS, "CLAIMS");
  } else if (textBlocks.SOURCES !== undefined) {
    const sources = parseJsonBlock(textBlocks.SOURCES, "SOURCES");
    const gapsRaw = textBlocks.GAPS ?? "[]";
    let gaps: JsonValue;
    try {
      gaps = JSON.parse(gapsRaw) as JsonValue;
    } catch {
      gaps = gapsRaw
        .split(/\r?\n/)
        .map((line) => line.replace(/^\s*[-*]\s*/, "").trim())
        .filter(Boolean);
    }
    payload = {
      sources,
      gaps,
      answer: textBlocks.ANSWER ?? "",
    };
  }

  const loop = parseLoop(text);
  const terminatedMatches = [...text.matchAll(/\[TASK_TERMINATED\]/g)];
  if (terminatedMatches.length > 1) {
    throw new Error("multiple TASK_TERMINATED markers");
  }
  const terminated = terminatedMatches.length === 1;
  if (profile.persistent && terminated) {
    throw new Error("persistent role emitted TASK_TERMINATED");
  }
  if (!profile.persistent && loop !== null) {
    throw new Error("ephemeral role emitted LOOP");
  }
  if (profile.persistent && loop === null) {
    throw new Error("persistent role is missing LOOP");
  }
  if (!profile.persistent && !terminated) {
    throw new Error("ephemeral role is missing TASK_TERMINATED");
  }

  return {
    role,
    marker,
    control: parseControlFields(mainBody),
    payload,
    textBlocks,
    loop,
    terminated,
    rawText: text,
  };
}

export function findFirstJsonValue(text: string): JsonValue | null {
  const starts: number[] = [];
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === "{" || text[index] === "[") {
      starts.push(index);
    }
  }
  for (const start of starts) {
    const opener = text[start];
    const closer = opener === "{" ? "}" : "]";
    let depth = 0;
    let quoted = false;
    let escaped = false;
    for (let index = start; index < text.length; index += 1) {
      const character = text[index];
      if (quoted) {
        if (escaped) {
          escaped = false;
        } else if (character === "\\") {
          escaped = true;
        } else if (character === "\"") {
          quoted = false;
        }
        continue;
      }
      if (character === "\"") {
        quoted = true;
        continue;
      }
      if (character === opener) depth += 1;
      if (character === closer) depth -= 1;
      if (depth === 0) {
        const candidate = text.slice(start, index + 1);
        try {
          return JSON.parse(candidate) as JsonValue;
        } catch {
          break;
        }
      }
    }
  }
  return null;
}

export function semanticPayloadUnchanged(before: JsonValue, after: JsonValue | null): boolean {
  return after !== null && JSON.stringify(before) === JSON.stringify(after);
}

