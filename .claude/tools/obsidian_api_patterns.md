# Obsidian API Verified Calling Patterns

> 提取自经过验证的 skill 文件:
> - `.claude/skills/learning-experiment-from-notes-answer/SKILL.md`
> - `.claude/skills/obsidian-keyword-explain/SKILL.md`
>
> 编写/修改 skill 时必须遵守这些参数格式。违反任一规则都会导致 MCP 验证错误。

---

## 1. `obsidian_search_notes` — 搜索笔记

### 1.1 omnisearch 模式（主模式，推荐）

**`path:` 过滤嵌入 query 字符串内。没有 `pathPrefix` 参数（那是 text 模式专用）。**

```
obsidian_search_notes(mode="omnisearch", query="path:<dir>  <keyword>")
```

单关键词:
```
obsidian_search_notes(mode="omnisearch", query="path:paper_secs  KV-Cache")
```

多词短语（双引号包围）:
```
obsidian_search_notes(mode="omnisearch", query="path:paper_secs  \"KV Cache\"")
obsidian_search_notes(mode="omnisearch", query="path:knowledge_notes  \"GPU occupancy\" \"concurrent kernel\"")
```

**关键约束**:
- `mode` 必须为 `"omnisearch"`
- `path:<dir>` 是 query 的一部分，不是独立参数
- 多词短语必须用双引号包围；单关键词不用
- 上游硬上限 50 条；用 `-exclusion` 和 `path:`/`ext:` 过滤缩小范围
- 可分页：`cursor` / `nextCursor`

### 1.2 text 模式（降级回退，文件名/内容子串匹配）

**使用 `pathPrefix` 参数限定目录。有独立的 `pathPrefix` 参数。**

```
obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="paper_secs/")
obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="knowledge_notes/")
obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="experiment_notes/")
obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="idea_notes/")
obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="human_notes/")
obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="learning_outputs/")
```

**关键约束**:
- `mode` 必须为 `"text"`
- `pathPrefix` 是独立参数（不在 query 内）
- 大小写不敏感子串匹配（文件名 + 笔记正文）
- `contextLength` 控制匹配上下文字符数（默认 100）

### 1.3 jsonlogic 模式

```
obsidian_search_notes(mode="jsonlogic", logic={...})
```

---

## 2. `obsidian_get_note` — 读取笔记

### 2.1 target 参数：discriminated union（必须含 `type` 字段）

**正确**:
```json
{
  "format": "content",
  "target": {
    "type": "path",
    "path": "knowledge_notes/example.md"
  }
}
```

**错误**（会导致 MCP 验证错误）:
```
❌ "target": "knowledge_notes/example.md"          ← 不能传裸字符串
❌ "target": {"path": "knowledge_notes/example.md"} ← 缺少 type 字段
```

### 2.2 format 投影

| `format` | 返回内容 | 使用场景 |
|----------|---------|---------|
| `"content"` | 原始 markdown 正文 | 读取笔记正文构建上下文（默认首选） |
| `"full"` | 正文 + frontmatter + tags + 文件元数据 | 需要 frontmatter 上下文时。可传 `includeLinks: true` 解析外链 |
| `"document-map"` | 目录：headings、block IDs、frontmatter 字段名 | 发现可用结构目标（section 读取前） |
| `"section"` | 单个 heading 子树 / block 正文 / frontmatter 字段值 | 精准读取特定章节。需要 `section: { type, target }` |

### 2.3 section 精确读取

**嵌套标题用 `Parent::Child` 语法**:

```json
{
  "format": "section",
  "target": {"type": "path", "path": "paper_secs/xxx/paper.md"},
  "section": {"type": "heading", "target": "4 Implementation"}
}
```

嵌套示例:
```json
{
  "format": "section",
  "target": {"type": "path", "path": "paper_secs/xxx/paper.md"},
  "section": {"type": "heading", "target": "4 Implementation::4.1 Compiler Design"}
}
```

**读取前先调 `document-map`** 发现可用 headings，避免 heading 未找到错误。

### 2.4 其他 target 类型

| target.type | 含义 |
|-------------|------|
| `"path"` | vault 相对路径，如 `"paper_secs/2025/xxx.md"` |
| `"active"` | Obsidian 中当前打开的文件 |
| `"periodic"` | 周期笔记（daily/weekly/monthly/quarterly/yearly），可选 ISO 日期 |

---

## 3. `obsidian_list_notes` — 列目录

```
obsidian_list_notes(path="paper_secs/", depth=1)
```

- `depth`: 递归深度（1 = 仅目标目录）
- `extension`: 过滤扩展名（如 `"md"`）
- `nameRegex`: ECMAScript 正则过滤名称

---

## 4. 标准搜索目录

六个本地 vault 搜索目录（所有 Obsidian API 搜索必须仅限这些目录）:

| 目录 | 用途 |
|------|------|
| `paper_secs/` | 论文原文 |
| `knowledge_notes/` | 知识笔记 |
| `experiment_notes/` | 实验笔记 |
| `idea_notes/` | idea 笔记 |
| `human_notes/` | 人工笔记 |
| `learning_outputs/` | 学习输出 |

---

## 5. 硬限制

1. **所有 vault 笔记检索只能通过 Obsidian API**：搜索用 `obsidian_search_notes`，读取用 `obsidian_get_note`
2. **禁止使用文件系统搜索**作为证据检索手段（包括 `rg`、`grep`、`find`、`ls`、Python 脚本扫描、shell 通配符）
3. Web/联网搜索仅允许作为外部补充证据，不能替代本地 Obsidian API 搜索
4. **omnisearch 的 `path:` 过滤是嵌入 query 的**，不是独立参数；**text 模式用 `pathPrefix` 独立参数**
5. **`obsidian_get_note` 的 target 必须是 discriminated object** `{type: "path", path: "..."}`
6. **`obsidian_search_notes` 的 `mode` 参数是必选的**，不能省略

---

## 6. 常见错误速查

| 错误 | 原因 | 正确写法 |
|------|------|---------|
| `Input validation error: Invalid option: expected "text"\|"jsonlogic"\|"omnisearch"` | 缺少 `mode` 参数 | 加 `mode="text"` 或 `mode="omnisearch"` |
| `Input validation error: Invalid input: expected object, rec` | `obsidian_get_note` 的 `target` 缺 `type` | `target={type: "path", path: "..."}` |
| `Heading 'X' not found` | heading 名称不对或嵌套标题需 `::` | 先调 `document-map` 获取精确 heading 名 |
| omnisearch 零命中但 text 模式能搜到 | omnisearch 是 BM25 排序，需要中文/缩写变体 | 降级到 text 模式 |
| `No such tool available: mcp__obsidian__obsidian_read_note` | 工具名错误 | 用 `obsidian_get_note`，不是 `obsidian_read_note` |
