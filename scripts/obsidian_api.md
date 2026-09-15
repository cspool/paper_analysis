# Obsidian API：用法与配置

本文记录本机 Obsidian 知识库对外接口的全部链路：Obsidian App → Local REST API 插件 →
`obsidian` MCP server → Claude Code / 各 Skill；以及让 Obsidian 在 CLI 与 GUI 之间切换时始终
在线的 systemd 托管方案。配置文件副本在 [`scripts/obsidian_service/`](obsidian_service/)。

## 1. 组件与端口

```text
Claude Code / Skill
  └─ MCP "obsidian"  (stdio; ~/.bun/bin/bunx obsidian-mcp-server@latest, 当前 3.5.3)
       └─ HTTP  http://127.0.0.1:27123   (insecure server, 默认使用)
          HTTPS https://127.0.0.1:27124  (自签名证书)
            └─ Obsidian App 内的插件 obsidian-local-rest-api (4.0.2, "Local REST API & MCP Server")
                 └─ vault = /data3/paper_analysis
```

| 组件 | 位置 | 说明 |
|---|---|---|
| Obsidian App | `~/Desktop/Obsidian-1.12.7.AppImage`，软链 `~/Applications/Obsidian.AppImage` | 应用包会自更新（日志显示已加载 1.13.7 asar）；升级 AppImage 后只需重指软链 |
| Local REST API 插件 | `/data3/paper_analysis/.obsidian/plugins/obsidian-local-rest-api/` | `data.json` 保存端口、证书、API key；`enableInsecureServer=true` |
| MCP server | `bunx obsidian-mcp-server@latest`（缓存在 `/tmp/bunx-1000-obsidian-mcp-server@latest/`） | 每个 Claude 会话按需拉起，进程退出即结束 |
| MCP 配置（项目） | `/data3/paper_analysis/.mcp.json` | **已纳入 git 跟踪且内含 `OBSIDIAN_API_KEY`**，见 §6 |
| MCP 配置（全局） | `~/.claude.json` → `mcpServers.obsidian` | 与项目配置相同 |
| `paper-analysis-jobs` MCP | `http://127.0.0.1:3020/mcp`，Docker 容器 `paper-analysis-job-mcp`（`restart=unless-stopped`） | **与 Obsidian 无关**，GUI 退出不受影响 |

只有 `obsidian` MCP 依赖 Obsidian App 进程：App 一退出，27123/27124 消失，所有 `obsidian_*`
工具立即失败。

## 2. MCP server 配置

`.mcp.json` 片段（key 已省略）：

```json
{
  "mcpServers": {
    "obsidian": {
      "type": "stdio",
      "command": "/home/descfly/.bun/bin/bunx",
      "args": ["obsidian-mcp-server@latest"],
      "env": {
        "MCP_TRANSPORT_TYPE": "stdio",
        "MCP_LOG_LEVEL": "info",
        "OBSIDIAN_API_KEY": "<插件设置页里的 API Key>",
        "MCP_ENABLE_COMMANDS": "true",
        "OBSIDIAN_REQUEST_TIMEOUT_MS": "60000"
      }
    }
  }
}
```

`obsidian-mcp-server` 识别的环境变量（3.5.3）：

| 变量 | 默认 | 用途 |
|---|---|---|
| `OBSIDIAN_API_KEY` | 必填 | 插件 API Key |
| `OBSIDIAN_BASE_URL` | `http://127.0.0.1:27123` | 改成 `https://127.0.0.1:27124` 需配合 `OBSIDIAN_VERIFY_SSL=false` |
| `OBSIDIAN_VERIFY_SSL` | `true` | 自签名证书时置 `false` |
| `OBSIDIAN_REQUEST_TIMEOUT_MS` | — | 单请求超时；本项目设 60000 |
| `OBSIDIAN_ENABLE_COMMANDS` / `MCP_ENABLE_COMMANDS` | — | 允许执行 Obsidian 命令 |
| `OBSIDIAN_READ_ONLY` | — | 只读模式 |
| `OBSIDIAN_READ_PATHS` / `OBSIDIAN_WRITE_PATHS` | — | 按路径限制读写范围 |
| `OBSIDIAN_OMNISEARCH_URL` | — | 接 Omnisearch 插件做全文检索（本 vault 已装 omnisearch） |

暴露的工具：`obsidian_list_notes`、`obsidian_get_note`、`obsidian_search_notes`、
`obsidian_write_note`、`obsidian_append_to_note`、`obsidian_patch_note`、`obsidian_replace_in_note`、
`obsidian_delete_note`、`obsidian_manage_frontmatter`、`obsidian_manage_tags`、`obsidian_list_tags`、
`obsidian_open_in_ui`。笔记按 vault 相对路径（含 `.md`）寻址，例如
`paper_catch/20260915_223659_merged.md`；`obsidian_list_notes` 用 `path`（不是 `directory`）、
`depth`、`extension`、`nameRegex` 过滤，单次上限 1000 条。

## 3. 直接调用 REST API

```bash
KEY=$(python3 -c "import json;print(json.load(open('/data3/paper_analysis/.mcp.json'))['mcpServers']['obsidian']['env']['OBSIDIAN_API_KEY'])")
H="Authorization: Bearer $KEY"

# 健康检查（无需 key；返回插件 manifest 与 authenticated 字段）
curl -s http://127.0.0.1:27123/

# 列目录 / 读笔记 / 写笔记 / 追加 / 删除
curl -s -H "$H" http://127.0.0.1:27123/vault/paper_catch/
curl -s -H "$H" "http://127.0.0.1:27123/vault/paper_catch/20260915_223659_merged.titles.md"
curl -s -H "$H" -X PUT  -H "Content-Type: text/markdown" --data-binary @note.md "http://127.0.0.1:27123/vault/tmp/note.md"
curl -s -H "$H" -X POST -H "Content-Type: text/markdown" --data-binary $'\n追加内容' "http://127.0.0.1:27123/vault/tmp/note.md"
curl -s -H "$H" -X DELETE "http://127.0.0.1:27123/vault/tmp/note.md"

# 读笔记的结构化形式（含 frontmatter、tags、stat）
curl -s -H "$H" -H "Accept: application/vnd.olrapi.note+json" \
  "http://127.0.0.1:27123/vault/paper_catch/20260915_223659_merged.md"

# 全文搜索（已实测，返回 filename/score/matches/context）
curl -s -H "$H" -X POST "http://127.0.0.1:27123/search/simple/?query=AccelOpt&contextLength=80"
# /search/ 只接受 application/vnd.olrapi.jsonlogic+json（Dataview DQL 需要 Dataview 插件，本 vault 未装）；
# JsonLogic 会扫描整个 vault，在本 vault 上 20 秒内不返回，日常请用 /search/simple/ 或 MCP 的 obsidian_search_notes。

# 标签、当前活动笔记、命令列表、完整 OpenAPI
curl -s -H "$H" http://127.0.0.1:27123/tags/
curl -s -H "$H" http://127.0.0.1:27123/active/
curl -s -H "$H" http://127.0.0.1:27123/commands/
curl -s -H "$H" http://127.0.0.1:27123/openapi.yaml

# HTTPS 端口需忽略自签名证书
curl -s -k -H "$H" https://127.0.0.1:27124/vault/
```

路径与方法（来自 `/openapi.yaml`）：`/` GET；`/vault/{path}` GET/PUT/POST/PATCH/DELETE；
`/active/` GET/PUT/POST/PATCH/DELETE；`/search/` POST；`/search/simple/` POST；`/tags/` GET；
`/commands/` GET/POST；`/mcp/` GET/POST（插件 4.x 自带的 MCP 端点，本项目未使用）；
`/obsidian-local-rest-api.crt` GET。无 `Authorization` 头时返回 401。

## 4. 让 Obsidian 始终在线：systemd 托管

### 问题

- Obsidian 是 Electron 应用，默认需要显示器；注销/切换会话时退出，`obsidian` MCP 随之失效。
- 从 CLI 直接运行 AppImage 会因 Ubuntu 的 `kernel.apparmor_restrict_unprivileged_userns=1`
  触发 Chromium SUID sandbox 致命错误（"chrome-sandbox is owned by root and has mode 4755"），
  必须加 `--no-sandbox`。
- 本机无 Xvfb、sudo 需要密码；但 Electron 支持 `--ozone-platform=headless --disable-gpu`，
  不需要任何 X/Wayland 也能启动，REST API 约 7 秒就绪。

### 方案（已部署，2026-09-15）

| 文件 | 部署路径 | 作用 |
|---|---|---|
| [`obsidian_service/obsidian.service`](obsidian_service/obsidian.service) | `~/.config/systemd/user/obsidian.service` | 常驻单元：`WantedBy=default.target`、`Restart=always`、3 秒重启、`StartLimitIntervalSec=0` |
| [`obsidian_service/obsidian-managed`](obsidian_service/obsidian-managed) | `~/.local/bin/obsidian-managed` | 启动脚本：`graphical-session.target` 活跃且显示 socket 存在 → 带窗口启动；否则 → 无头启动。`OBSIDIAN_FORCE_HEADLESS=1` 可强制无头；`OBSIDIAN_APPIMAGE` 可换路径 |
| [`obsidian_service/obsidian-gui-switch.service`](obsidian_service/obsidian-gui-switch.service) | `~/.config/systemd/user/obsidian-gui-switch.service` | 绑定 `graphical-session.target`（`PartOf`/`WantedBy`）：GUI 登录时 `restart obsidian.service` 切到窗口模式，注销时再 `restart` 切到无头 |

行为矩阵：

| 场景 | 结果 |
|---|---|
| 开机、无人登录 | `loginctl` 已开 `Linger=yes` → user manager 启动 → 无头 Obsidian，API 可用 |
| GUI 登录 | switch 单元触发 restart → 带窗口实例接管（约 10 秒中断） |
| 手动关闭窗口 / 崩溃 / `kill` | 3 秒后自动重启，模式不变（实测 12 秒内 API 恢复） |
| 注销 / 会话结束 | switch 单元触发 restart → 无头实例接管 |
| 任意 shell / SSH | `systemctl --user start|restart obsidian.service` 即可启动 |

### 安装 / 更新

```bash
install -d ~/Applications ~/.local/bin ~/.config/systemd/user
ln -sfn ~/Desktop/Obsidian-1.12.7.AppImage ~/Applications/Obsidian.AppImage   # 升级时改这里
install -m 755 scripts/obsidian_service/obsidian-managed ~/.local/bin/
install -m 644 scripts/obsidian_service/obsidian.service scripts/obsidian_service/obsidian-gui-switch.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now obsidian.service obsidian-gui-switch.service
loginctl enable-linger "$USER"     # 已启用；开机无登录也拉起
```

### 日常命令

```bash
systemctl --user status obsidian.service            # 状态、当前 MainPID、重启次数
journalctl --user -u obsidian.service -f -o cat     # "starting with a window" / "starting headless"
systemctl --user restart obsidian.service           # 重新判定模式并重启
ps -o cmd= -p "$(systemctl --user show obsidian.service -p MainPID --value)"   # 含 ozone-platform=headless 即无头
curl -s http://127.0.0.1:27123/ | head -3           # API 健康
```

**不要再从桌面双击 AppImage**：Electron 单实例锁会把请求转发给服务实例，桌面上看不到新窗口。
需要窗口时 `systemctl --user restart obsidian.service`（GUI 会话存在时自动选择窗口模式）。

## 5. 故障排查

| 现象 | 检查 | 处理 |
|---|---|---|
| `obsidian_*` 工具连接错误 / ECONNREFUSED | `ss -ltn \| grep 2712`；`systemctl --user status obsidian.service` | `systemctl --user restart obsidian.service`，10 秒后重试 |
| 服务 `activating (auto-restart)` 循环，进程存活 0 秒 | `systemd-run --user --wait --pipe --collect timeout 8 ~/Applications/Obsidian.AppImage --no-sandbox` 看 stderr | 常见为 sandbox 错误（缺 `--no-sandbox`）或 AppImage 路径/软链失效 |
| 401 Unauthorized | key 与插件设置页不一致 | 在 Obsidian → 设置 → Local REST API 里查看/重置 key，同步到 `.mcp.json` 和 `~/.claude.json` |
| 插件端口变化 | `.obsidian/plugins/obsidian-local-rest-api/data.json` 的 `port`/`insecurePort` | 同步改 `OBSIDIAN_BASE_URL` |
| 桌面双击无窗口 | 服务实例正在运行（无头或已有窗口） | 见 §4 说明 |
| `paper-analysis-worker-tunnel.service` 持续失败 | 反向 SSH 到 worker-0 不可达 | 与本地 MCP 无关；不再使用可 `systemctl --user disable --now paper-analysis-worker-tunnel.service` |

## 6. 安全注意

- `.mcp.json` 被 git 跟踪且包含 `OBSIDIAN_API_KEY`；REST API 只监听 127.0.0.1，但仓库若推到
  公开远端会泄露 key。建议把 key 移到 `~/.claude.json`（已存在同样配置）并在 `.mcp.json` 中
  改为 `"OBSIDIAN_API_KEY": "${OBSIDIAN_API_KEY}"` 由环境注入，或直接在插件里重置 key。
- 插件 `data.json` 内含自签名证书私钥，同样只用于回环地址。
- `--no-sandbox` 关闭的是 Obsidian 内部 Chromium 渲染沙箱，对本机私有 vault 可接受；
  如需恢复沙箱，需要 root 执行 `sysctl kernel.apparmor_restrict_unprivileged_userns=0`
  或为 AppImage 配置 AppArmor profile。
