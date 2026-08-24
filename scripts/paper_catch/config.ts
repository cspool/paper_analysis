import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { CatchConfig, SourceSpec } from "./types.ts";

const URL_PATTERN = /https?:\/\/[^\s<>()]+/g;

export function parseCatchConfig(configPath: string): CatchConfig {
  const absolute = resolve(configPath);
  const text = readFileSync(absolute, "utf8");
  const lines = text.split(/\r?\n/);
  const urls: string[] = [];
  let interestStart = -1;

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index] ?? "";
    if (/^\s*感兴趣主题\s*[：:]/.test(line)) {
      interestStart = index;
      break;
    }
    for (const match of line.matchAll(URL_PATTERN)) {
      const url = match[0].replace(/[.,;，。；]+$/, "");
      if (!urls.includes(url)) urls.push(url);
    }
  }

  if (urls.length === 0) {
    throw new Error(`no source URLs found in config: ${absolute}`);
  }
  if (interestStart < 0) {
    throw new Error(`missing 感兴趣主题 section in config: ${absolute}`);
  }

  const first = (lines[interestStart] ?? "").replace(
    /^\s*感兴趣主题\s*[：:]\s*/,
    "",
  );
  const rest = lines.slice(interestStart + 1).join("\n").trim();
  const interest = [first.trim(), rest].filter(Boolean).join("\n").trim();
  if (!interest) {
    throw new Error(`interest topic is empty in config: ${absolute}`);
  }

  return {
    schemaVersion: "paper-catch-config-v1",
    configPath: absolute,
    configHash: createHash("sha256").update(text).digest("hex"),
    urls,
    interest,
  };
}

export function sourceSpecFromUrl(rawUrl: string): SourceSpec {
  const url = new URL(rawUrl);
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new Error(`unsupported source URL protocol: ${rawUrl}`);
  }
  if (url.hostname.toLowerCase() !== "github.com") {
    throw new Error(`paper catch currently requires a GitHub repository URL: ${rawUrl}`);
  }
  const segments = url.pathname.split("/").filter(Boolean);
  if (segments.length < 2) {
    throw new Error(`GitHub URL must contain owner/repository: ${rawUrl}`);
  }
  const owner = segments[0]!;
  const repository = segments[1]!.replace(/\.git$/i, "");
  const canonical = `https://github.com/${owner}/${repository}`;
  const fragment = url.hash ? decodeURIComponent(url.hash.slice(1)) : null;
  const identity = `${canonical}#${fragment ?? ""}`;
  const suffix = createHash("sha256").update(identity).digest("hex").slice(0, 10);
  return {
    sourceId: `${slug(owner)}-${slug(repository)}-${suffix}`,
    url: rawUrl,
    cloneUrl: `${canonical}.git`,
    fragment,
    displayName: `${owner}/${repository}${fragment ? `#${fragment}` : ""}`,
  };
}

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}
