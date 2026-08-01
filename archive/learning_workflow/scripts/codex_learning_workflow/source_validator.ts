import fs from "node:fs";
import path from "node:path";

import { claimId, normalizeIdentityText } from "./stable_ids.ts";
import type { Confidence, EvidenceClaim, EvidenceKind, JsonValue, RejectedClaim } from "./types.ts";

const VALID_EVIDENCE_KINDS = new Set(["direct", "inferred"]);
const VALID_CONFIDENCE = new Set(["low", "middle", "high"]);

function objectValue(value: unknown): Record<string, unknown> | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function normalizeQuote(value: string): string {
  return value.normalize("NFKC").replace(/\s+/g, " ").trim();
}

function resolveWithinRoots(sourcePath: string, vaultRoot: string, roots: string[]): string {
  const candidate = path.isAbsolute(sourcePath)
    ? path.resolve(sourcePath)
    : path.resolve(vaultRoot, sourcePath);
  const allowed = roots.some((root) => {
    const absoluteRoot = path.resolve(vaultRoot, root);
    return candidate === absoluteRoot || candidate.startsWith(`${absoluteRoot}${path.sep}`);
  });
  if (!allowed) {
    throw new Error(`source is outside configured evidence roots: ${sourcePath}`);
  }
  return candidate;
}

function locateQuote(lines: string[], quote: string, declaredStart: number, declaredEnd: number): { start: number; end: number } | null {
  const normalizedQuote = normalizeQuote(quote);
  if (!normalizedQuote) {
    return null;
  }
  const boundedStart = Math.max(1, declaredStart);
  const boundedEnd = Math.min(lines.length, Math.max(boundedStart, declaredEnd));
  const declaredText = normalizeQuote(lines.slice(boundedStart - 1, boundedEnd).join("\n"));
  if (declaredText.includes(normalizedQuote)) {
    return { start: boundedStart, end: boundedEnd };
  }

  const matches: Array<{ start: number; end: number; distance: number }> = [];
  const maximumWindow = Math.min(40, Math.max(1, quote.split(/\r?\n/).length + 8));
  for (let start = 0; start < lines.length; start += 1) {
    let aggregate = "";
    for (let end = start; end < Math.min(lines.length, start + maximumWindow); end += 1) {
      aggregate = aggregate ? `${aggregate}\n${lines[end]}` : lines[end];
      const normalizedAggregate = normalizeQuote(aggregate);
      if (normalizedAggregate.includes(normalizedQuote)) {
        matches.push({
          start: start + 1,
          end: end + 1,
          distance: Math.abs(start + 1 - declaredStart),
        });
        break;
      }
      if (normalizedAggregate.length > normalizedQuote.length * 3 + 500) {
        break;
      }
    }
  }
  matches.sort((left, right) => left.distance - right.distance || left.start - right.start);
  return matches[0] ?? null;
}

export function validateClaimCandidate(
  candidate: unknown,
  taskId: string,
  vaultRoot: string,
  roots: string[],
): { claim: EvidenceClaim | null; rejection: RejectedClaim | null } {
  const reasons: string[] = [];
  const object = objectValue(candidate);
  if (!object) {
    reasons.push("claim is not an object");
    return {
      claim: null,
      rejection: {
        candidate: (candidate ?? null) as JsonValue,
        taskId,
        reasons,
        rejectedAt: new Date().toISOString(),
      },
    };
  }

  const statement = String(object.statement ?? "").trim();
  const claimType = String(object.claim_type ?? object.claimType ?? "").trim();
  const evidenceKind = String(object.evidence_kind ?? object.evidenceKind ?? "direct").trim();
  const sourcePath = String(object.source_path ?? object.sourcePath ?? "").trim();
  const quote = String(object.quote ?? "").trim();
  const declaredStart = Number(object.line_start ?? object.lineStart ?? 1);
  const declaredEnd = Number(object.line_end ?? object.lineEnd ?? declaredStart);
  const applicableScope = String(object.applicable_scope ?? object.applicableScope ?? "").trim();
  const confidenceRaw = String(object.confidence ?? "middle").trim();

  if (!statement) reasons.push("statement is empty");
  if (!claimType) reasons.push("claim_type is empty");
  if (!VALID_EVIDENCE_KINDS.has(evidenceKind)) reasons.push(`invalid evidence_kind: ${evidenceKind}`);
  if (!sourcePath) reasons.push("source_path is empty");
  if (!quote) reasons.push("quote is empty");
  if (!Number.isInteger(declaredStart) || declaredStart < 1) reasons.push("line_start must be a positive integer");
  if (!Number.isInteger(declaredEnd) || declaredEnd < declaredStart) reasons.push("line_end must be >= line_start");
  if (!VALID_CONFIDENCE.has(confidenceRaw)) reasons.push(`invalid confidence: ${confidenceRaw}`);

  let absolutePath = "";
  let lines: string[] = [];
  if (sourcePath) {
    try {
      absolutePath = resolveWithinRoots(sourcePath, vaultRoot, roots);
      const stat = fs.statSync(absolutePath);
      if (!stat.isFile()) {
        reasons.push("source_path is not a file");
      } else {
        lines = fs.readFileSync(absolutePath, "utf8").split(/\r?\n/);
      }
    } catch (error) {
      reasons.push(error instanceof Error ? error.message : String(error));
    }
  }

  let located: { start: number; end: number } | null = null;
  if (lines.length > 0 && quote) {
    located = locateQuote(lines, quote, declaredStart, declaredEnd);
    if (!located) {
      reasons.push("quote cannot be located in source");
    }
  }

  if (reasons.length > 0 || !located) {
    return {
      claim: null,
      rejection: {
        candidate: object as unknown as JsonValue,
        taskId,
        reasons,
        rejectedAt: new Date().toISOString(),
      },
    };
  }

  const displayPath = path.relative(path.resolve(vaultRoot), absolutePath).split(path.sep).join("/");
  const canonicalPath = displayPath.startsWith("..") ? absolutePath : displayPath;
  const kind = evidenceKind as EvidenceKind;
  const confidence = confidenceRaw as Confidence;
  const now = new Date().toISOString();
  return {
    claim: {
      claimId: claimId(canonicalPath, located.start, located.end, quote),
      statement,
      claimType,
      evidenceKind: kind,
      sourcePath: canonicalPath,
      lineStart: located.start,
      lineEnd: located.end,
      quote,
      applicableScope,
      confidence,
      taskId,
      createdAt: now,
    },
    rejection: null,
  };
}

export function validateClaimCandidates(
  candidates: unknown[],
  taskId: string,
  vaultRoot: string,
  roots: string[],
): { accepted: EvidenceClaim[]; rejected: RejectedClaim[] } {
  const acceptedById = new Map<string, EvidenceClaim>();
  const rejected: RejectedClaim[] = [];
  for (const candidate of candidates) {
    const result = validateClaimCandidate(candidate, taskId, vaultRoot, roots);
    if (result.claim) {
      acceptedById.set(result.claim.claimId, result.claim);
    }
    if (result.rejection) {
      rejected.push(result.rejection);
    }
  }
  return {
    accepted: [...acceptedById.values()].sort((left, right) => left.claimId.localeCompare(right.claimId)),
    rejected,
  };
}

export function revalidateClaim(claim: EvidenceClaim, vaultRoot: string, roots: string[]): string[] {
  const result = validateClaimCandidate(
    {
      statement: claim.statement,
      claim_type: claim.claimType,
      evidence_kind: claim.evidenceKind,
      source_path: claim.sourcePath,
      line_start: claim.lineStart,
      line_end: claim.lineEnd,
      quote: claim.quote,
      applicable_scope: claim.applicableScope,
      confidence: claim.confidence,
    },
    claim.taskId,
    vaultRoot,
    roots,
  );
  if (!result.claim) {
    return result.rejection?.reasons ?? ["unknown source validation failure"];
  }
  if (normalizeIdentityText(result.claim.quote) !== normalizeIdentityText(claim.quote)) {
    return ["canonical quote changed during validation"];
  }
  return [];
}

