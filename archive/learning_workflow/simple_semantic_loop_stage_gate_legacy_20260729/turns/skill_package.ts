import { readFileSync, readdirSync, statSync } from "node:fs";
import { relative, resolve } from "node:path";
import { canonicalSha256, sha256Bytes } from "../contracts/index.ts";

const IGNORED_SEGMENTS = new Set(["__pycache__", ".pytest_cache"]);

export interface LoadedSkillPackage {
  root: string;
  skillMarkdown: string;
  files: Array<{ relativePath: string; sha256: string; sizeBytes: number }>;
  sha256: string;
}

export function loadSkillPackage(root: string): LoadedSkillPackage {
  const absoluteRoot = resolve(root);
  const files = walk(absoluteRoot)
    .filter((path) => !path.endsWith(".pyc"))
    .map((path) => {
      const bytes = readFileSync(path);
      return {
        relativePath: relative(absoluteRoot, path).split("\\").join("/"),
        sha256: sha256Bytes(bytes),
        sizeBytes: bytes.byteLength,
      };
    })
    .sort((left, right) => left.relativePath.localeCompare(right.relativePath));
  const skillPath = resolve(absoluteRoot, "SKILL.md");
  if (!files.some((file) => file.relativePath === "SKILL.md")) {
    throw new Error(`Skill package has no SKILL.md: ${absoluteRoot}`);
  }
  return {
    root: absoluteRoot,
    skillMarkdown: readFileSync(skillPath, "utf8"),
    files,
    sha256: canonicalSha256(files),
  };
}

function walk(root: string): string[] {
  const output: string[] = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    if (IGNORED_SEGMENTS.has(entry.name)) continue;
    const path = resolve(root, entry.name);
    if (entry.isDirectory()) output.push(...walk(path));
    else if (entry.isFile() && statSync(path).isFile()) output.push(path);
  }
  return output;
}

