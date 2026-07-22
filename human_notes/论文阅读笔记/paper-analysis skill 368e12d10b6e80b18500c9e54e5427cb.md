# paper-analysis skill

处理论文md，去重复、结构化提取section。ok！

/data3/paper_analysis/.claude/skills/obsidian-keyword-explain修改。

1、搜索使用obsidian的search api（ominisearch+关键词+path）。

2、语义分割后提取的关键词不再聚类分组，对每个关键词执行：

2.1： /data3/paper_analysis/paper_secs下根据关键词搜索，记录至多5个返回的path，优先分数最高的。

2.2：在3个notes（/data3/paper_analysis/experiment_notes, /data3/paper_analysis/idea_notes, /data3/paper_analysis/knowledge_notes）下分别搜索关键词，记录md的path和score，每个notes的path下分别调用api，并分别记录至多5个path（优先分数最高）。

2.3：相关性计算方式改变，直接使用每个match的score。

3、所有关键词的搜索和记录path完成后，去重，读取后加入上下文。

4、基于段落的语义分割，对每个segment，基于上下文进行解释（解释内容和方式、细节不变），最后对整个段落进行解释。

Tool name: obsidian_search_notes
Full name: mcp__obsidian__obsidian_search_notes

Parameters:
● mode (required): string - Which search algorithm to run. `text` matches a substring
case-insensitively across filenames and note bodies, returning surrounding context windows.
Search the vault by text substring, JSONLogic predicate, or BM25-ranked Omnisearch query. Pick the
mode that matches the query shape — `omnisearch` is best for ranked relevance, typo tolerance, and
PDF/OCR coverage (via the Text Extractor plugin). Results paginate via opaque cursors: omit `cursor`
for the first page, then pass `nextCursor` from the prior response. Text-mode hits additionally clip
per file at `maxMatchesPerHit`.

Parameters:
● mode (required): string - Which search algorithm to run. `text` matches a substring
case-insensitively across filenames and note bodies, returning surrounding context windows.
`jsonlogic` evaluates a JSONLogic tree against each note, with `var` paths into `path`,
`content`, `frontmatter.<key>`, `tags`, and `stat.{ctime,mtime,size}`, plus `glob` and `regexp`
operators. `omnisearch` runs a BM25-ranked query via the Omnisearch plugin — supports quoted
phrases, `-exclusion`, `path:` / `ext:` filters, typo tolerance, and PDF/OCR (with Text
Extractor); upstream caps results at 50.
● query: string - The query string. Required for `text` and `omnisearch` modes; ignored in
`jsonlogic` mode (use `logic` instead — passing a JSONLogic tree here will fail Zod validation
since this field must be a string).
● logic: object - JSONLogic tree. Required for `jsonlogic` mode; ignored in `text` and `omnisearch`
modes (use `query` instead — passing a string here will fail Zod validation since this field
must be an object).
● contextLength: integer - Characters of context on each side of the match (text mode only).
● pathPrefix: string - Filter returned filenames by prefix (text mode only, applied client-side).
● maxMatchesPerHit: integer - Cap on match contexts returned per file in text mode. When clipped,
the hit carries `truncated: true` and `totalMatches`.
● cursor: string - Opaque cursor from a prior response. Omit for the first page. Page size is
server-determined; do not assume a fixed value.

learning-survey