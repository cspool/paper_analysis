import { createHash } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
import { basename, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import type {
  CandidateSourceRef,
  CommitSummary,
  PaperCandidate,
  ReportManifest,
  SourceSnapshot,
  SourceSpec,
} from "./types.ts";

interface AddedLine {
  filePath: string;
  line: number;
  text: string;
}

interface CandidateDraft {
  title: string;
  paperUrl: string | null;
  codeUrls: string[];
  urls: string[];
  sourceRef: CandidateSourceRef;
}

export interface CollectSourceOptions {
  spec: SourceSpec;
  cacheRoot: string;
  baselineTimestamp: string;
  previousManifest: ReportManifest | null;
  checkedAt: string;
}

export interface CollectedSource {
  snapshot: SourceSnapshot;
  candidates: PaperCandidate[];
}

export function collectSourceUpdate(options: CollectSourceOptions): CollectedSource {
  const mirror = resolve(options.cacheRoot, `${options.spec.sourceId}.git`);
  ensureMirror(options.spec, mirror);
  const currentHead = git(mirror, ["rev-parse", "HEAD"]).trim();
  const warnings: string[] = [];
  const previous = options.previousManifest?.sourceHeads[options.spec.sourceId];
  let baselineMode: SourceSnapshot["baselineMode"];
  let baselineHead: string;

  if (previous && objectExists(mirror, `${previous.head}^{commit}`)) {
    baselineMode = "PREVIOUS_REPORT_HEAD";
    baselineHead = previous.head;
  } else {
    baselineMode = options.previousManifest
      ? "REPORT_TIMESTAMP"
      : "INITIAL_LOOKBACK";
    baselineHead = gitAllowFailure(mirror, [
      "rev-list",
      "-1",
      `--before=${options.baselineTimestamp}`,
      currentHead,
    ]).stdout.trim();
    if (!baselineHead) baselineHead = emptyTree(mirror);
  }

  if (objectExists(mirror, `${baselineHead}^{commit}`)) {
    const ancestry = gitAllowFailure(mirror, [
      "merge-base",
      "--is-ancestor",
      baselineHead,
      currentHead,
    ]);
    if (ancestry.status !== 0) {
      const mergeBase = gitAllowFailure(mirror, ["merge-base", baselineHead, currentHead])
        .stdout.trim();
      warnings.push(
        `previous head is not an ancestor of current HEAD; using ${mergeBase || "empty tree"}`,
      );
      baselineHead = mergeBase || emptyTree(mirror);
    }
  }

  const changed = baselineHead !== currentHead;
  const commits = changed
    ? readCommits(mirror, baselineHead, currentHead, options.baselineTimestamp)
    : [];
  const numstat = changed
    ? git(mirror, [
      "-c",
      "core.quotePath=false",
      "diff",
      "--numstat",
      baselineHead,
      currentHead,
      "--",
      "*.md",
      "*.markdown",
    ])
    : "";
  const stats = parseNumstat(numstat);
  const patch = changed
    ? git(mirror, [
      "-c",
      "core.quotePath=false",
      "diff",
      "--unified=0",
      "--no-ext-diff",
      "--no-renames",
      baselineHead,
      currentHead,
      "--",
      "*.md",
      "*.markdown",
    ], 20 * 1024 * 1024)
    : "";
  const added = parseAddedLines(patch);
  const contentCache = new Map<string, string[]>();
  const rangeCache = new Map<string, [number, number] | null>();

  if (options.spec.fragment) {
    const readmes = [...new Set(added.map((line) => line.filePath))]
      .filter((path) => /^readme(?:\.[^.]+)?\.md$/i.test(basename(path)) || /^readme\.md$/i.test(basename(path)));
    let found = false;
    for (const path of readmes) {
      const lines = readFileLines(mirror, currentHead, path, contentCache);
      const range = findGithubSectionRange(lines, options.spec.fragment);
      rangeCache.set(path, range);
      if (range) found = true;
    }
    if (!found && added.length > 0) {
      throw new Error(
        `URL fragment #${options.spec.fragment} was not found in changed README files for ${options.spec.url}`,
      );
    }
  }

  const baselineIndex = added.length > 0
    ? readBaselinePaperIndex(mirror, baselineHead, options.spec.fragment)
    : { titles: new Set<string>(), paperUrls: new Set<string>() };
  const drafts: CandidateDraft[] = [];
  for (const line of added) {
    if (options.spec.fragment) {
      const range = rangeCache.get(line.filePath) ?? null;
      if (!range || line.line < range[0] || line.line > range[1]) continue;
    }
    const fileLines = readFileLines(mirror, currentHead, line.filePath, contentCache);
    const context = contextFor(fileLines, line.line, 2);
    const extracted = extractPaperCandidateFromLine(line.text, context);
    if (!extracted) continue;
    if (baselineIndex.titles.has(normalizeTitle(extracted.title))) continue;
    const paperKey = extracted.paperUrl ? canonicalPaperUrl(extracted.paperUrl) : null;
    if (paperKey && baselineIndex.paperUrls.has(paperKey)) continue;
    const blame = blameLine(mirror, currentHead, line.filePath, line.line, commits.at(-1));
    const sourceRef: CandidateSourceRef = {
      sourceId: options.spec.sourceId,
      sourceUrl: options.spec.url,
      repositoryUrl: options.spec.cloneUrl.replace(/\.git$/, ""),
      fragment: options.spec.fragment,
      filePath: line.filePath,
      line: line.line,
      addedLine: line.text,
      context,
      commitSha: blame.sha,
      committedAt: blame.committedAt,
      commitSubject: blame.subject,
      urls: extracted.urls,
    };
    drafts.push({ ...extracted, sourceRef });
  }

  const candidates = mergeCandidateDrafts(drafts);
  const snapshot: SourceSnapshot = {
    sourceId: options.spec.sourceId,
    url: options.spec.url,
    cloneUrl: options.spec.cloneUrl,
    fragment: options.spec.fragment,
    baselineHead,
    baselineMode,
    currentHead,
    checkedAt: options.checkedAt,
    changed,
    commits,
    changedMarkdownFiles: stats.files,
    stats: {
      markdownFilesChanged: stats.files.length,
      linesAdded: stats.added,
      linesDeleted: stats.deleted,
      candidateLines: candidates.length,
    },
    warnings,
  };
  return { snapshot, candidates };
}

function readBaselinePaperIndex(
  mirror: string,
  baseline: string,
  fragment: string | null,
): { titles: Set<string>; paperUrls: Set<string> } {
  const titles = new Set<string>();
  const paperUrls = new Set<string>();
  const paths = gitAllowFailure(mirror, [
    "-c",
    "core.quotePath=false",
    "ls-tree",
    "-r",
    "--name-only",
    baseline,
  ], 20 * 1024 * 1024).stdout.split("\n")
    .filter((path) => /\.(?:md|markdown)$/i.test(path));
  const selectedPaths = fragment
    ? paths.filter((path) => /^readme(?:\.[^.]+)?\.md$/i.test(basename(path)) || /^readme\.md$/i.test(basename(path)))
    : paths;
  for (const path of selectedPaths) {
    const result = gitAllowFailure(mirror, ["show", `${baseline}:${path}`], 20 * 1024 * 1024);
    if (result.status !== 0) continue;
    const lines = result.stdout.split(/\r?\n/);
    const range = fragment ? findGithubSectionRange(lines, fragment) : [1, lines.length] as [number, number];
    if (!range) continue;
    for (let line = range[0]; line <= range[1]; line += 1) {
      const text = lines[line - 1] ?? "";
      const context = contextFor(lines, line, 2);
      const candidate = extractPaperCandidateFromLine(text, context);
      if (!candidate) continue;
      titles.add(normalizeTitle(candidate.title));
      if (candidate.paperUrl) paperUrls.add(canonicalPaperUrl(candidate.paperUrl));
    }
  }
  return { titles, paperUrls };
}

function canonicalPaperUrl(value: string): string {
  try {
    const url = new URL(value);
    const arxiv = /\/(?:abs|pdf)\/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$/i.exec(url.pathname);
    if (/arxiv\.org$/i.test(url.hostname) && arxiv) return `arxiv:${arxiv[1]}`;
    url.hash = "";
    url.search = "";
    url.hostname = url.hostname.toLowerCase();
    url.pathname = url.pathname.replace(/\/$/, "");
    return url.toString();
  } catch {
    return value;
  }
}

export function mergePaperCandidates(groups: PaperCandidate[][]): PaperCandidate[] {
  const merged = new Map<string, PaperCandidate>();
  for (const candidate of groups.flat()) {
    const existing = merged.get(candidate.normalizedTitle);
    if (!existing) {
      merged.set(candidate.normalizedTitle, structuredClone(candidate));
      continue;
    }
    existing.urls = unique([...existing.urls, ...candidate.urls]);
    existing.codeUrls = unique([...existing.codeUrls, ...candidate.codeUrls]);
    existing.sourceRefs.push(...candidate.sourceRefs);
    existing.sourceRefs.sort((a, b) => b.committedAt.localeCompare(a.committedAt));
    if (candidate.latestCommittedAt > existing.latestCommittedAt) {
      existing.latestCommittedAt = candidate.latestCommittedAt;
      existing.title = candidate.title;
    }
    existing.paperUrl ??= candidate.paperUrl;
  }
  return [...merged.values()].sort((a, b) =>
    b.latestCommittedAt.localeCompare(a.latestCommittedAt) ||
    a.normalizedTitle.localeCompare(b.normalizedTitle)
  );
}

export function extractPaperCandidateFromLine(
  line: string,
  context = line,
): Omit<CandidateDraft, "sourceRef"> | null {
  const trimmed = line.trim();
  if (!trimmed || /^(#{1,6}\s|<!--|```)/.test(trimmed)) return null;
  const links = markdownLinks(line);
  const urls = unique([...urlsIn(context), ...links.map((link) => link.url)]);
  const meaningful = links.filter((link) => isMeaningfulTitleLabel(link.label));
  const hasVenuePrefix = /^\s*(?:[-*+]\s*)?(?:\|\s*)?(?:\[[^\]]*(?:19|20)\d{2}[^\]]*\]\s*)+/i.test(line) ||
    /\b(?:arxiv|neurips|icml|iclr|asplos|sosp|osdi|nsdi|mlsys|hpca|isca|micro|eurosys|ppopp|acl|emnlp|cvpr|sigcomm|atc|sc|saa)['’]?\s*\d{2,4}\b/i.test(line);
  const hasScholarlyUrl = urls.some(isScholarlyUrl);
  const indentation = line.length - line.trimStart().length;
  if (meaningful.length === 0 && !hasScholarlyUrl && !(hasVenuePrefix && indentation <= 4)) {
    return null;
  }

  let title: string;
  let paperUrl: string | null = null;
  if (meaningful.length > 0) {
    const best = meaningful.sort((a, b) => b.label.length - a.label.length)[0]!;
    title = cleanTitle(best.label);
    paperUrl = isCodeUrl(best.url) ? null : best.url;
  } else {
    const replaced = line.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, (_all, label: string) =>
      isGenericLinkLabel(label) ? " " : ` ${label} `
    );
    const cells = replaced.split("|").map(cleanTitle).filter(looksLikeTitle);
    title = cells.sort((a, b) => b.length - a.length)[0] ?? cleanTitle(replaced);
  }
  if (!looksLikeTitle(title)) return null;
  if (GENERIC_TITLES.has(title.toLowerCase())) return null;

  const codeUrls = unique(urls.filter(isCodeUrl));
  paperUrl ??= urls.find((url) => isScholarlyUrl(url) && !isCodeUrl(url)) ?? null;
  return { title, paperUrl, codeUrls, urls };
}

export function findGithubSectionRange(
  lines: string[],
  fragment: string,
): [number, number] | null {
  const target = fragment.toLowerCase().replace(/^#/, "");
  const seen = new Map<string, number>();
  const headings: Array<{ line: number; level: number; slug: string }> = [];
  for (let index = 0; index < lines.length; index += 1) {
    const match = /^(#{1,6})\s+(.+?)\s*$/.exec(lines[index] ?? "");
    if (!match) continue;
    const base = githubSlug(match[2]!);
    const count = seen.get(base) ?? 0;
    seen.set(base, count + 1);
    headings.push({
      line: index + 1,
      level: match[1]!.length,
      slug: count === 0 ? base : `${base}-${count}`,
    });
  }
  const position = headings.findIndex((heading) => heading.slug === target);
  if (position < 0) return null;
  const heading = headings[position]!;
  const next = headings.slice(position + 1).find((candidate) => candidate.level <= heading.level);
  return [heading.line + 1, (next?.line ?? lines.length + 1) - 1];
}

function ensureMirror(spec: SourceSpec, mirror: string): void {
  mkdirSync(resolve(mirror, ".."), { recursive: true });
  if (!existsSync(mirror)) {
    const result = spawnSync("git", ["clone", "--mirror", spec.cloneUrl, mirror], {
      encoding: "utf8",
      timeout: 300_000,
      env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
    });
    if (result.status !== 0) {
      throw new Error(`git clone failed for ${spec.url}: ${result.stderr.trim()}`);
    }
    return;
  }
  git(mirror, ["remote", "set-url", "origin", spec.cloneUrl]);
  git(mirror, ["remote", "update", "--prune"]);
}

function git(mirror: string, args: string[], maxBuffer = 5 * 1024 * 1024): string {
  const result = gitAllowFailure(mirror, args, maxBuffer);
  if (result.status !== 0) {
    throw new Error(`git ${args.join(" ")} failed: ${result.stderr.trim()}`);
  }
  return result.stdout;
}

function gitAllowFailure(
  mirror: string,
  args: string[],
  maxBuffer = 5 * 1024 * 1024,
): { status: number | null; stdout: string; stderr: string } {
  const result = spawnSync("git", ["--git-dir", mirror, ...args], {
    encoding: "utf8",
    timeout: 300_000,
    maxBuffer,
    env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
  });
  return {
    status: result.status,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
  };
}

function objectExists(mirror: string, object: string): boolean {
  return gitAllowFailure(mirror, ["cat-file", "-e", object]).status === 0;
}

function emptyTree(mirror: string): string {
  const result = spawnSync("git", ["--git-dir", mirror, "hash-object", "-t", "tree", "--stdin"], {
    input: "",
    encoding: "utf8",
  });
  if (result.status !== 0) throw new Error(`cannot create empty Git tree: ${result.stderr}`);
  return result.stdout.trim();
}

function readCommits(
  mirror: string,
  baseline: string,
  head: string,
  since: string,
): CommitSummary[] {
  const args = objectExists(mirror, `${baseline}^{commit}`)
    ? ["log", "--reverse", "--format=%H%x1f%cI%x1f%s", `${baseline}..${head}`]
    : ["log", "--reverse", `--since=${since}`, "--format=%H%x1f%cI%x1f%s", head];
  const text = git(mirror, args);
  return text.split("\n").filter(Boolean).map((line) => {
    const [sha = "", committedAt = "", subject = ""] = line.split("\x1f");
    return { sha, committedAt, subject };
  });
}

function parseNumstat(text: string): { files: string[]; added: number; deleted: number } {
  const files: string[] = [];
  let added = 0;
  let deleted = 0;
  for (const line of text.split("\n")) {
    if (!line) continue;
    const [add = "0", remove = "0", path = ""] = line.split("\t");
    files.push(path);
    if (/^\d+$/.test(add)) added += Number(add);
    if (/^\d+$/.test(remove)) deleted += Number(remove);
  }
  return { files: unique(files.filter(Boolean)), added, deleted };
}

function parseAddedLines(patch: string): AddedLine[] {
  const output: AddedLine[] = [];
  let filePath = "";
  let currentLine = 0;
  for (const line of patch.split("\n")) {
    if (line.startsWith("+++ b/")) {
      filePath = line.slice(6);
      continue;
    }
    const hunk = /^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line);
    if (hunk) {
      currentLine = Number(hunk[1]);
      continue;
    }
    if (!filePath || line.startsWith("diff --git") || line.startsWith("--- ")) continue;
    if (line.startsWith("+") && !line.startsWith("+++")) {
      output.push({ filePath, line: currentLine, text: line.slice(1) });
      currentLine += 1;
    } else if (line.startsWith(" ")) {
      currentLine += 1;
    }
  }
  return output;
}

function readFileLines(
  mirror: string,
  head: string,
  path: string,
  cache: Map<string, string[]>,
): string[] {
  const existing = cache.get(path);
  if (existing) return existing;
  const result = gitAllowFailure(mirror, ["show", `${head}:${path}`], 20 * 1024 * 1024);
  const lines = result.status === 0 ? result.stdout.split(/\r?\n/) : [];
  cache.set(path, lines);
  return lines;
}

function contextFor(lines: string[], line: number, radius: number): string {
  const start = Math.max(0, line - 1 - radius);
  const end = Math.min(lines.length, line + radius);
  return lines.slice(start, end).map((text, index) => `${start + index + 1}: ${text}`).join("\n");
}

function blameLine(
  mirror: string,
  head: string,
  path: string,
  line: number,
  fallback?: CommitSummary,
): CommitSummary {
  const result = gitAllowFailure(mirror, [
    "blame",
    "--line-porcelain",
    "-L",
    `${line},${line}`,
    head,
    "--",
    path,
  ]);
  if (result.status !== 0) {
    return fallback ?? { sha: head, committedAt: new Date(0).toISOString(), subject: "unknown" };
  }
  const first = result.stdout.split("\n")[0]?.split(" ")[0] ?? head;
  const timestamp = /^committer-time (\d+)$/m.exec(result.stdout)?.[1];
  const subject = /^summary (.*)$/m.exec(result.stdout)?.[1] ?? "unknown";
  return {
    sha: first,
    committedAt: timestamp ? new Date(Number(timestamp) * 1000).toISOString() : fallback?.committedAt ?? new Date(0).toISOString(),
    subject,
  };
}

function mergeCandidateDrafts(drafts: CandidateDraft[]): PaperCandidate[] {
  const result = new Map<string, PaperCandidate>();
  for (const draft of drafts) {
    const normalizedTitle = normalizeTitle(draft.title);
    if (!normalizedTitle) continue;
    const existing = result.get(normalizedTitle);
    if (existing) {
      existing.urls = unique([...existing.urls, ...draft.urls]);
      existing.codeUrls = unique([...existing.codeUrls, ...draft.codeUrls]);
      existing.sourceRefs.push(draft.sourceRef);
      existing.paperUrl ??= draft.paperUrl;
      if (draft.sourceRef.committedAt > existing.latestCommittedAt) {
        existing.latestCommittedAt = draft.sourceRef.committedAt;
        existing.title = draft.title;
      }
      continue;
    }
    result.set(normalizedTitle, {
      candidateId: `paper-${createHash("sha256").update(normalizedTitle).digest("hex").slice(0, 16)}`,
      title: draft.title,
      normalizedTitle,
      paperUrl: draft.paperUrl,
      codeUrls: draft.codeUrls,
      urls: draft.urls,
      latestCommittedAt: draft.sourceRef.committedAt,
      sourceRefs: [draft.sourceRef],
    });
  }
  return [...result.values()];
}

export function normalizeTitle(value: string): string {
  return value.normalize("NFKD").replace(/[^\p{L}\p{N}]+/gu, " ").toLowerCase().trim();
}

function markdownLinks(text: string): Array<{ label: string; url: string }> {
  return [...text.matchAll(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g)].map((match) => ({
    label: cleanTitle(match[1] ?? ""),
    url: match[2] ?? "",
  }));
}

function urlsIn(text: string): string[] {
  return [...text.matchAll(/https?:\/\/[^\s<>()\]]+/g)].map((match) =>
    (match[0] ?? "").replace(/[.,;]+$/, "")
  );
}

function cleanTitle(value: string): string {
  const cells = value
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/[*_`~]/g, "")
    .replace(/^\s*(?:[-*+]\s*|\d+[.)]\s*)/, "")
    .replace(/^\s*\|+\s*|\s*\|+\s*$/g, "")
    .replace(/^(?:\[[^\]]*(?:(?:19|20)\d{2}|['’]\d{2})[^\]]*\]\s*)+/i, "")
    .replace(/^(?:paper|title)\s*:\s*/i, "")
    .replace(/\s+/g, " ")
    .trim();
  return cells.replace(/[\s|,;:-]+$/, "").trim();
}

function looksLikeTitle(value: string): boolean {
  if (value.length < 12 || value.length > 320) return false;
  const words = value.match(/[\p{L}\p{N}][\p{L}\p{N}'’-]*/gu) ?? [];
  if (words.length < 3 || words.length > 50) return false;
  if (/^https?:\/\//i.test(value)) return false;
  return true;
}

function isMeaningfulTitleLabel(label: string): boolean {
  return looksLikeTitle(label) && !isGenericLinkLabel(label);
}

function isGenericLinkLabel(label: string): boolean {
  const normalized = label.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  return /^(paper|pdf|code|github|project|project page|website|arxiv|link|slides?|homepage|demo|dataset|model|hugging face|hf)$/.test(normalized) ||
    /^(arxiv|neurips|icml|iclr|asplos|sosp|osdi|nsdi|mlsys|hpca|isca|micro|eurosys|ppopp|acl|emnlp|cvpr|sigcomm|atc|sc|saa)\s*\d{0,4}$/.test(normalized);
}

function isCodeUrl(value: string): boolean {
  try {
    const host = new URL(value).hostname.toLowerCase();
    return host === "github.com" || host === "gitlab.com" || host === "bitbucket.org" ||
      host === "huggingface.co" && !/\/papers\//.test(new URL(value).pathname);
  } catch {
    return false;
  }
}

function isScholarlyUrl(value: string): boolean {
  try {
    const url = new URL(value);
    const host = url.hostname.toLowerCase();
    return /(?:arxiv\.org|openreview\.net|aclanthology\.org|dl\.acm\.org|ieeexplore\.ieee\.org|proceedings\.mlr\.press|usenix\.org|neurips\.cc|openaccess\.thecvf\.com|doi\.org)$/.test(host) ||
      /\.pdf(?:$|\?)/i.test(url.pathname + url.search);
  } catch {
    return false;
  }
}

function githubSlug(value: string): string {
  return value
    .replace(/<[^>]+>/g, "")
    .replace(/[*_`~]/g, "")
    .toLowerCase()
    .trim()
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

const GENERIC_TITLES = new Set([
  "table of contents",
  "accepted papers",
  "week papers",
  "month papers",
  "latest papers",
  "paper list",
  "news",
  "references",
]);
