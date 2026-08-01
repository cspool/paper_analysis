import { createHash } from "node:crypto";

import type { AnchorSignature, JsonValue } from "./types.ts";

function sortJson(value: JsonValue): JsonValue {
  if (Array.isArray(value)) {
    return value.map(sortJson);
  }
  if (value !== null && typeof value === "object") {
    const output: { [key: string]: JsonValue } = {};
    for (const key of Object.keys(value).sort()) {
      output[key] = sortJson(value[key]);
    }
    return output;
  }
  return value;
}

export function canonicalJson(value: JsonValue): string {
  return JSON.stringify(sortJson(value));
}

export function normalizeIdentityText(value: unknown): string {
  return String(value ?? "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[\s_-]+/g, " ")
    .replace(/[^\p{L}\p{N}.+:/ ]/gu, "")
    .trim();
}

export function stableHash(value: JsonValue, length = 16): string {
  return createHash("sha256").update(canonicalJson(value)).digest("hex").slice(0, length);
}

export function stableId(prefix: string, value: JsonValue, length = 16): string {
  return `${prefix}-${stableHash(value, length)}`;
}

export function normalizeAnchorSignature(signature: AnchorSignature): AnchorSignature {
  return {
    workload: normalizeIdentityText(signature.workload),
    phase: normalizeIdentityText(signature.phase),
    regime: normalizeIdentityText(signature.regime),
    backend: normalizeIdentityText(signature.backend),
    bottleneck: normalizeIdentityText(signature.bottleneck),
    primaryBaselineExecutionPath: normalizeIdentityText(signature.primaryBaselineExecutionPath),
    targetMetrics: [...new Set(signature.targetMetrics.map(normalizeIdentityText).filter(Boolean))].sort(),
  };
}

export function anchorId(signature: AnchorSignature): string {
  return stableId("A", normalizeAnchorSignature(signature) as unknown as JsonValue, 14);
}

export function claimId(sourcePath: string, lineStart: number, lineEnd: number, quote: string): string {
  return stableId("C", [
    normalizeIdentityText(sourcePath),
    lineStart,
    lineEnd,
    normalizeIdentityText(quote),
  ], 16);
}

