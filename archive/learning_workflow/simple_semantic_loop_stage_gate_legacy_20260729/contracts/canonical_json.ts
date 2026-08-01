import { createHash } from "node:crypto";
import { isAbsolute, posix, relative, resolve, sep } from "node:path";

export type JsonPrimitive = null | boolean | number | string;
export type JsonValue =
  | JsonPrimitive
  | JsonValue[]
  | { readonly [key: string]: JsonValue };

function assertJsonNumber(value: number): void {
  if (!Number.isFinite(value)) {
    throw new TypeError("canonical JSON does not support NaN or Infinity");
  }
}

/**
 * RFC-8785-compatible for the JSON values used by this workflow: object keys
 * are sorted lexicographically, arrays retain order, and undefined/non-JSON
 * values are rejected instead of being silently dropped.
 */
export function canonicalJson(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    assertJsonNumber(value);
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record).sort();
    const entries = keys.map((key) => {
      const child = record[key];
      if (
        child === undefined ||
        typeof child === "function" ||
        typeof child === "symbol" ||
        typeof child === "bigint"
      ) {
        throw new TypeError(`non-JSON value at object key ${JSON.stringify(key)}`);
      }
      return `${JSON.stringify(key)}:${canonicalJson(child)}`;
    });
    return `{${entries.join(",")}}`;
  }
  throw new TypeError(`non-JSON value of type ${typeof value}`);
}

export function canonicalSha256(value: unknown): string {
  return createHash("sha256").update(canonicalJson(value), "utf8").digest("hex");
}

export function sha256Bytes(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

export function canonicalEqual(left: unknown, right: unknown): boolean {
  return canonicalJson(left) === canonicalJson(right);
}

export function assertSha256(value: string, field = "sha256"): void {
  if (!/^[a-f0-9]{64}$/.test(value)) {
    throw new TypeError(`${field} must be a lowercase SHA-256 hex digest`);
  }
}

/**
 * Return a normalized POSIX relative path contained by root.
 * Symlink containment is checked by the caller after the target exists.
 */
export function normalizeContainedRelativePath(
  root: string,
  candidate: string,
): string {
  if (!root || !isAbsolute(root)) {
    throw new TypeError("root must be an absolute path");
  }
  if (!candidate || candidate.includes("\0") || isAbsolute(candidate)) {
    throw new TypeError("candidate must be a non-empty relative path");
  }
  const platformCandidate = candidate.split("/").join(sep);
  const absolute = resolve(root, platformCandidate);
  const rel = relative(resolve(root), absolute);
  if (rel === "" || rel === ".") return ".";
  if (rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel)) {
    throw new TypeError(`path escapes approved root: ${candidate}`);
  }
  const normalized = rel.split(sep).join("/");
  if (normalized.split("/").some((part) => part === "" || part === "..")) {
    throw new TypeError(`path is not canonical: ${candidate}`);
  }
  return posix.normalize(normalized);
}

