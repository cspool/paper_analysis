# 新容器接入 Paper Analysis MCP

适用于与 Paper Analysis 服务运行在同一台 Linux 宿主机上的开发容器。

客户端只注册两个 HTTP MCP：

```text
http://127.0.0.1:3010/mcp  只读检索和读取
http://127.0.0.1:3020/mcp  分析、下载、OCR 和批处理 Job
```

不要注册内部端口 `3021/3022`，也不要再配置依赖 `bun` 或 `bunx` 的旧 STDIO
`obsidian-mcp-server`。

## 1. 宿主机启动并验证服务

```bash
cd /data3/Projects/paper_analysis_service

make up
make job-up
python3 client/verify-mcp.py
```

成功结果应包含 Read MCP、Job MCP 和 `status: passed`。

## 2. 创建容器时使用 host 网络

Docker 命令必须包含：

```bash
--network host
```

Compose 配置使用：

```yaml
network_mode: host
```

检查新容器；将 `new_container` 替换为实际容器名：

```bash
docker inspect new_container --format '{{.HostConfig.NetworkMode}}'
```

预期输出为 `host`。已使用默认 bridge 网络创建的容器不能通过容器内
`127.0.0.1` 访问宿主机 MCP，应使用 host 网络重新创建。

## 3. 从容器内检查连通性

```bash
docker exec new_container sh -lc '
curl --noproxy "*" -sS -o /dev/null -w "Read MCP: %{http_code}\n" \
  http://127.0.0.1:3010/mcp
curl --noproxy "*" -sS -o /dev/null -w "Job MCP: %{http_code}\n" \
  http://127.0.0.1:3020/healthz
'
```

两个状态码都应为 `200`。

## 4. 注册到 Codex

进入容器：

```bash
docker exec -it new_container bash
```

在容器内执行：

```bash
codex mcp add paper-analysis-read \
  --url http://127.0.0.1:3010/mcp

codex mcp add paper-analysis-jobs \
  --url http://127.0.0.1:3020/mcp

codex mcp list
```

## 5. 注册到 Claude Code

先在容器内进入实际项目目录，再使用 `local` scope，避免修改其他项目：

```bash
cd /workspace/new_project

claude mcp add --transport http --scope local \
  paper-analysis-read \
  http://127.0.0.1:3010/mcp

claude mcp add --transport http --scope local \
  paper-analysis-jobs \
  http://127.0.0.1:3020/mcp

claude mcp list
```

将 `/workspace/new_project` 替换为容器中的实际项目目录。注册后重新启动
Codex/Claude 会话；工具说明和参数 Schema 会通过 MCP `tools/list` 自动提供。

## 6. 可选：移除失效的旧配置

只有确认旧 `obsidian` STDIO 配置不再使用时才执行：

```bash
codex mcp remove obsidian
claude mcp remove obsidian
```

## 7. 配置持久化与安全

- 容器删除后仍需保留配置时，应持久化容器用户的 `~/.codex` 和 Claude 配置。
- 如果这些路径直接挂载自宿主机，容器内修改也会影响宿主机配置。
- 若容器设置了 HTTP 代理，应确保 `127.0.0.1,localhost` 不经过代理；上面的
  `curl --noproxy "*"` 可用于排除代理影响。
- 两个 MCP 无远程鉴权并仅监听宿主机回环地址，只应开放给可信的 host-network
  容器，不要改为公网监听。
