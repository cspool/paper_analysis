# Codex Skills 清单与迁移安装命令

> 历史环境审计快照：本文记录 2026-07-24 当日的 Skill 数量、版本、
> 本地路径和仓库提交，不是当前环境的权威清单。其中的快照迁移方法和
> 校验步骤仍可用于参考；执行前必须重新核对当前 `SKILL.md`、路径、版本和
> 提交号。

审计日期：2026-07-24
源机器 Codex：`codex-cli 0.144.1`

## 1. 当前状态摘要

本机共有 **131 个互不重复的 Skill**：

| 类型 | 已安装 | 当前启用 |
|---|---:|---:|
| Codex 内置系统 Skill | 5 | 5 |
| `paper_analysis` 项目级 Skill | 5 | 5 |
| 全局自定义 Skill | 24 | 24 |
| Agent Skills 兼容目录 | 2 | 2 |
| Orchestra 研究 Skill | 95 | 76 |
| **合计** | **131** | **112** |

Orchestra 中有 19 个 Skill 已安装，但在当前 `~/.codex/config.toml` 中显式设置为 `enabled = false`。因此：

- 当前会话可用：112 个。
- 磁盘完整安装：131 个。
- 进入 `paper_analysis` 项目之外的目录时，5 个项目级 Skill 通常不会加载。

来源与版本状态：

- 项目级 Skill：`git@github.com:cspool/paper_analysis.git`，审计时提交 `a70d44fc4e44b1de16e31e7b27e95aaf7049e2ff`，工作区干净。
- 全局自定义 Skill：`git@github.com:cspool/ty-codex-skills.git`，审计时提交 `9eb3b36aa96443147e9cbe8a6070f020a61ac5d8`。
- 全局自定义 Skill 仓库目前存在未提交修改和 4 个未跟踪 Skill；只执行 `git clone` **不能完整复现本机状态**。
- Orchestra：通过 `npx @orchestra-research/ai-research-skills` 安装，锁文件记录安装时间为 2026-06-15。
- `humanizer-zh`：`op7418/Humanizer-zh`。
- `find-skills`：`vercel-labs/skills` 中的 `skills/find-skills`。

## 2. 推荐方案：精确快照迁移

此方案会保留本机全局自定义 Skill 的未提交修改、未跟踪 Skill，以及当前安装的 Orchestra 内容。它不会复制版本相关的 `.system` 目录；目标机器应由自己的 Codex 安装提供系统 Skill。

### 2.1 在当前源机器打包

```bash
set -euo pipefail

SOURCE_CODEX_SKILLS_ROOT="/home/descfly/.codex/skills"
SOURCE_AGENT_SKILLS_ROOT="/home/descfly/.agents/skills"
SOURCE_PROJECT_SKILLS_ROOT="/data3/paper_analysis/.codex/skills"
SKILL_SNAPSHOT_DIR="/tmp/codex-skills-snapshot-20260724"

install -d "$SKILL_SNAPSHOT_DIR"

# --dereference 会把指向 ~/.orchestra 的绝对符号链接展开为普通目录，
# 因而快照可迁移到用户名或 HOME 路径不同的机器。
tar --dereference \
  --exclude='./.git' \
  --exclude='./.system' \
  -C "$SOURCE_CODEX_SKILLS_ROOT" \
  -czf "$SKILL_SNAPSHOT_DIR/global-codex-skills.tar.gz" \
  .

tar -C "$SOURCE_AGENT_SKILLS_ROOT" \
  -czf "$SKILL_SNAPSHOT_DIR/agent-compatible-skills.tar.gz" \
  find-skills humanizer-zh

tar -C "$SOURCE_PROJECT_SKILLS_ROOT" \
  -czf "$SKILL_SNAPSHOT_DIR/paper-analysis-project-skills.tar.gz" \
  .

(
  cd "$SKILL_SNAPSHOT_DIR"
  sha256sum ./*.tar.gz > SHA256SUMS
)

du -sh "$SKILL_SNAPSHOT_DIR"
```

把快照传到目标机器；替换示例中的用户名与主机名：

```bash
scp -r /tmp/codex-skills-snapshot-20260724 \
  user@target-host:/tmp/
```

### 2.2 在目标机器安装全局 Skill

建议先安装并至少启动一次 Codex，使目标机器自行创建 `.system` 内置 Skill。

以下命令按“新机器、尚无自定义全局 Skill”设计。检测到已有非系统内容时会停止，避免静默覆盖。

```bash
set -euo pipefail

SKILL_SNAPSHOT_DIR="/tmp/codex-skills-snapshot-20260724"
TARGET_CODEX_ROOT="${CODEX_HOME:-$HOME/.codex}"
TARGET_CODEX_SKILLS_ROOT="$TARGET_CODEX_ROOT/skills"

cd "$SKILL_SNAPSHOT_DIR"
sha256sum -c SHA256SUMS

install -d "$TARGET_CODEX_SKILLS_ROOT"

if find "$TARGET_CODEX_SKILLS_ROOT" \
  -mindepth 1 -maxdepth 1 \
  ! -name '.system' \
  -print -quit | grep -q .; then
  echo "目标 skills 目录已有非系统内容；请先备份并人工合并。"
  exit 1
fi

tar -C "$TARGET_CODEX_SKILLS_ROOT" \
  -xzf "$SKILL_SNAPSHOT_DIR/global-codex-skills.tar.gz"

tar -C "$TARGET_CODEX_SKILLS_ROOT" \
  -xzf "$SKILL_SNAPSHOT_DIR/agent-compatible-skills.tar.gz"
```

此安装把 Orchestra Skill 展开为普通目录，而不是保留源机器上的绝对符号链接；对 Codex 的使用效果相同，但更适合跨机器迁移。

### 2.3 复现当前 19 个禁用项

若希望目标机器与当前会话一样只启用 112 个 Skill，继续执行：

```bash
set -euo pipefail

TARGET_CODEX_ROOT="${CODEX_HOME:-$HOME/.codex}"
TARGET_CODEX_SKILLS_ROOT="$TARGET_CODEX_ROOT/skills"
TARGET_CODEX_CONFIG="$TARGET_CODEX_ROOT/config.toml"

install -d "$TARGET_CODEX_ROOT"
touch "$TARGET_CODEX_CONFIG"

for disabled_skill_dir in \
  academic-plotting \
  deepspeed \
  megatron-core \
  miles \
  ml-training-recipes \
  moe-training \
  nemo-curator \
  nemo-guardrails \
  openrlhf \
  pinecone \
  presenting-conference-talks \
  prompt-guard \
  ray-train \
  simpo \
  skypilot \
  slime \
  torchforge \
  torchtitan \
  verl
do
  disabled_skill_path="$TARGET_CODEX_SKILLS_ROOT/$disabled_skill_dir/SKILL.md"

  if ! grep -Fq "path = \"$disabled_skill_path\"" "$TARGET_CODEX_CONFIG"; then
    printf '\n[[skills.config]]\npath = "%s"\nenabled = false\n' \
      "$disabled_skill_path" >> "$TARGET_CODEX_CONFIG"
  fi
done
```

若希望使用全部 131 个 Skill，则跳过本节。

### 2.4 安装项目级 Skill

推荐直接克隆项目；5 个项目级 Skill 已包含在仓库中：

```bash
TARGET_PROJECT_PARENT="$HOME/projects"
TARGET_PROJECT_ROOT="$TARGET_PROJECT_PARENT/paper_analysis"

install -d "$TARGET_PROJECT_PARENT"
git clone git@github.com:cspool/paper_analysis.git "$TARGET_PROJECT_ROOT"
git -C "$TARGET_PROJECT_ROOT" \
  checkout a70d44fc4e44b1de16e31e7b27e95aaf7049e2ff
```

若目标机器已有通过其他方式同步的项目，而其 `.codex/skills` 为空，可改用快照：

```bash
SKILL_SNAPSHOT_DIR="/tmp/codex-skills-snapshot-20260724"
TARGET_PROJECT_ROOT="/absolute/path/to/paper_analysis"

install -d "$TARGET_PROJECT_ROOT/.codex/skills"
tar -C "$TARGET_PROJECT_ROOT/.codex/skills" \
  -xzf "$SKILL_SNAPSHOT_DIR/paper-analysis-project-skills.tar.gz"
```

不要在已有文件的 `.codex/skills` 上直接解包，以免覆盖目标机器上的修改。

## 3. 可联网重装方案

此方案适合重新获取上游版本，不保证与源机器的本地修改逐字节一致。

### 3.1 Codex 内置 Skill

以下 5 个 Skill 由 Codex 自身提供，不应从源机器覆盖目标机器的 `.system`：

- `imagegen`
- `openai-docs`
- `plugin-creator`
- `skill-creator`
- `skill-installer`

安装或更新 Codex 后重启即可加载。若要求严格一致，应在两台机器使用同一 Codex CLI 版本；本机审计版本为 `0.144.1`。

### 3.2 Orchestra 的 95 个 Skill

目标机器需要 Node.js/npm，然后运行：

```bash
npx @orchestra-research/ai-research-skills
```

按交互提示选择 Codex。安装结束后，如果要复现当前启用状态，再按第 2.3 节禁用 19 项。

### 3.3 `humanizer-zh` 与 `find-skills`

```bash
npx skills add https://github.com/op7418/Humanizer-zh.git -g -y
npx skills add vercel-labs/skills@find-skills -g -y
```

### 3.4 将项目级 Skill 直接安装为全局 Skill

如果不准备克隆整个 `paper_analysis` 项目，可以使用 Codex 内置安装器：

```bash
TARGET_CODEX_ROOT="${CODEX_HOME:-$HOME/.codex}"
TARGET_CODEX_SKILLS_ROOT="$TARGET_CODEX_ROOT/skills"
SKILL_INSTALLER_SCRIPT="$TARGET_CODEX_SKILLS_ROOT/.system/skill-installer/scripts/install-skill-from-github.py"

python3 "$SKILL_INSTALLER_SCRIPT" \
  --repo cspool/paper_analysis \
  --ref a70d44fc4e44b1de16e31e7b27e95aaf7049e2ff \
  --dest "$TARGET_CODEX_SKILLS_ROOT" \
  --path \
    .codex/skills/export-conversation-notes \
    .codex/skills/obsidian-keyword-explainer \
    .codex/skills/paper-experiment-idea \
    .codex/skills/paper-knowledge-base \
    .codex/skills/paper-single-analysis
```

若仓库为私有仓库，应先在目标机器配置 GitHub 凭据或 `GH_TOKEN`。安装器遇到同名目标目录会停止，不会覆盖。

### 3.5 从 Git 安装全局自定义 Skill 的已提交基线

下列命令只安装 `ty-codex-skills` 提交 `9eb3b36a...` 中仍在当前环境使用的部分，并包含 Nature 系列依赖的 `_shared`。它不包含当前机器的未提交修改，也不包含后述 4 个未跟踪 Skill。

```bash
set -euo pipefail

TARGET_CODEX_ROOT="${CODEX_HOME:-$HOME/.codex}"
TARGET_CODEX_SKILLS_ROOT="$TARGET_CODEX_ROOT/skills"
CUSTOM_SKILLS_STAGE="$(mktemp -d)"
CUSTOM_SKILLS_REPO="$CUSTOM_SKILLS_STAGE/ty-codex-skills"

install -d "$TARGET_CODEX_SKILLS_ROOT"
git clone git@github.com:cspool/ty-codex-skills.git "$CUSTOM_SKILLS_REPO"
git -C "$CUSTOM_SKILLS_REPO" \
  checkout 9eb3b36aa96443147e9cbe8a6070f020a61ac5d8

CUSTOM_SKILL_ITEMS=(
  _shared
  dispatch-layer-reconstruct-onnx
  nature-academic-search
  nature-citation
  nature-data
  nature-figure
  nature-paper-to-patent
  nature-paper2ppt
  nature-polishing
  nature-reader
  nature-response
  nature-reviewer
  nature-writing
  project-docker-runner
  trace-patch-target-discovery
  visipruner-fx-process-visualization
  visipruner-fx-trace-workflow
  visipruner-process-performance-breakdown
  visipruner-same-input-evidence
  visipruner-sampled-latency-attribution
  visipruner-trace-dispatch-profile
)

for custom_skill_item in "${CUSTOM_SKILL_ITEMS[@]}"; do
  if test -e "$TARGET_CODEX_SKILLS_ROOT/$custom_skill_item"; then
    echo "目标已存在：$TARGET_CODEX_SKILLS_ROOT/$custom_skill_item"
    exit 1
  fi
done

for custom_skill_item in "${CUSTOM_SKILL_ITEMS[@]}"; do
  cp -a \
    "$CUSTOM_SKILLS_REPO/$custom_skill_item" \
    "$TARGET_CODEX_SKILLS_ROOT/"
done
```

以下 4 个当前 Skill 尚未进入该仓库提交，只能通过第 2 节快照或在源机器完成提交并推送后获取：

- `remote-http-ssh-codex`
- `visipruner-fx-process-nvtx-instrumentation`
- `visipruner-same-input-layer-wise-workflow`
- `visipruner-segmented-process-attribution`

## 4. 完整 Skill 清单

### 4.1 Codex 内置系统 Skill：5

`imagegen`、`openai-docs`、`plugin-creator`、`skill-creator`、`skill-installer`

### 4.2 `paper_analysis` 项目级 Skill：5

`export-conversation-notes`、`obsidian-keyword-explainer`、`paper-experiment-idea`、`paper-knowledge-base`、`paper-single-analysis`

### 4.3 全局自定义 Skill：24

`dispatch-layer-reconstruct-onnx`、`nature-academic-search`、`nature-citation`、`nature-data`、`nature-figure`、`nature-paper-to-patent`、`nature-paper2ppt`、`nature-polishing`、`nature-reader`、`nature-response`、`nature-reviewer`、`nature-writing`、`project-docker-runner`、`remote-http-ssh-codex`、`trace-patch-target-discovery`、`visipruner-fx-process-nvtx-instrumentation`、`visipruner-fx-process-visualization`、`visipruner-fx-trace-workflow`、`visipruner-process-performance-breakdown`、`visipruner-same-input-evidence`、`visipruner-same-input-layer-wise-workflow`、`visipruner-sampled-latency-attribution`、`visipruner-segmented-process-attribution`、`visipruner-trace-dispatch-profile`

### 4.4 Agent Skills 兼容目录：2

`find-skills`、`humanizer-zh`

### 4.5 当前启用的 Orchestra Skill：76

- 独立研究工作流：`autoresearch`
- 模型架构：`implementing-llms-litgpt`、`mamba-architecture`、`nanogpt`、`rwkv-architecture`
- Tokenization：`huggingface-tokenizers`、`sentencepiece`
- 微调：`axolotl`、`llama-factory`、`peft-fine-tuning`、`unsloth`
- 机制可解释性：`nnsight-remote-interpretability`、`pyvene-interventions`、`sparse-autoencoder-training`、`transformer-lens-interpretability`
- 数据处理：`ray-data`
- 后训练：`grpo-rl-training`、`fine-tuning-with-trl`
- 安全与对齐：`constitutional-ai`、`llamaguard`
- 分布式训练：`huggingface-accelerate`、`pytorch-fsdp2`、`pytorch-lightning`
- 基础设施：`lambda-labs-gpu-cloud`、`modal-serverless-gpu`
- 优化与量化：`awq-quantization`、`quantizing-models-bitsandbytes`、`optimizing-attention-flash`、`gguf-quantization`、`gptq`、`hqq-quantization`
- 评测：`evaluating-code-models`、`evaluating-llms-harness`、`nemo-evaluator-sdk`
- 推理与服务：`llama-cpp`、`sglang`、`tensorrt-llm`、`serving-llms-vllm`
- MLOps：`mlflow`、`experiment-tracking-swanlab`、`tensorboard`、`weights-and-biases`
- Agent：`evolving-ai-agents`、`autogpt-agents`、`crewai-multi-agent`、`langchain`、`llamaindex`
- RAG：`chroma`、`faiss`、`qdrant-vector-search`、`sentence-transformers`
- Prompt Engineering：`dspy`、`guidance`、`instructor`、`outlines`
- 可观测性：`langsmith-observability`、`phoenix-observability`
- 多模态：`audiocraft-audio-generation`、`blip-2-vision-language`、`clip`、`evaluating-cosmos-policy`、`llava`、`fine-tuning-serving-openpi`、`fine-tuning-openvla-oft`、`segment-anything-model`、`stable-diffusion-image-generation`、`whisper`
- 新兴技术：`knowledge-distillation`、`long-context`、`model-merging`、`model-pruning`、`speculative-decoding`
- 论文写作：`ml-paper-writing`、`systems-paper-writing`
- 研究创意：`brainstorming-research-ideas`、`creative-thinking-for-research`

### 4.6 已安装但当前禁用的 Orchestra Skill：19

`distributed-llm-pretraining-torchtitan`、`nemo-curator`、`miles-rl-training`、`openrlhf-training`、`simpo-training`、`slime-rl-training`、`torchforge-rl-training`、`verl-rl-training`、`nemo-guardrails`、`prompt-guard`、`deepspeed`、`training-llms-megatron`、`ray-train`、`skypilot-multi-cloud-orchestration`、`ml-training-recipes`、`pinecone`、`moe-training`、`academic-plotting`、`presenting-conference-talks`

## 5. 安装后校验

先完全退出并重新启动 Codex，再执行：

```bash
TARGET_CODEX_ROOT="${CODEX_HOME:-$HOME/.codex}"
TARGET_CODEX_SKILLS_ROOT="$TARGET_CODEX_ROOT/skills"

find -L "$TARGET_CODEX_SKILLS_ROOT" \
  -type f -name SKILL.md \
  -print | sort

printf '全局目录中的 SKILL.md 数量：'
find -L "$TARGET_CODEX_SKILLS_ROOT" \
  -type f -name SKILL.md \
  -print | wc -l
```

使用精确快照并已由 Codex 创建 5 个 `.system` Skill 时，全局目录应有 **126** 个 `SKILL.md`；进入带有 5 个项目级 Skill 的 `paper_analysis` 后，磁盘完整总数为 **131**。若复现了 19 个禁用配置，当前可用总数应为 **112**。

还应检查：

```bash
codex --version
test -f "$TARGET_CODEX_SKILLS_ROOT/nature-writing/SKILL.md"
test -f "$TARGET_CODEX_SKILLS_ROOT/_shared/core/terminology-ledger.md"
test -f "$TARGET_CODEX_SKILLS_ROOT/autoresearch/SKILL.md" \
  || test -f "$TARGET_CODEX_SKILLS_ROOT/0-autoresearch-skill/SKILL.md"
```

Skill 通常是指令、脚本和参考资料的集合。完成目录安装并不等于自动安装其中提到的全部 Python 包、GPU 运行库、外部 CLI、Obsidian 服务或 API 凭据；这些依赖应在实际使用对应 Skill 时按其 `SKILL.md` 单独配置。
