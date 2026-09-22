# 20260922 下载清单：DiT 跨步缓存对照组（dit_cache_comparators）
## 用法：python3 scripts/paper_download.py --file paper_catch/20260922-dit_cache_comparators.md --output papers_pdf/paper_dit_cache_comparators
## 规则：`#` 行为说明（下载器跳过）；论文一行一条，链接为首选来源
## 背景：调研报告方向 3（DiT block 级可训练三态门控）的对照组，六层 survey 中 **全 vault 零命中**，D3 的核心论断此前没有本地对手可比
## 去重结果：7 篇经 Obsidian/omnisearch 检索均无本地笔记（paper_secs / knowledge_notes / experiment_notes / human_notes 全零命中），无一篇需剔除

## A. 可训练 / 低秩缓存校准（与 D3 的 rank-r 中间态 + 可训练门控直接竞争）
- [DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](https://arxiv.org/abs/2602.05449)
- [LearniBridge: Learnable Calibration of Feature Caching for Diffusion Models Acceleration](https://arxiv.org/abs/2606.26778)

## B. 预测式 / 高阶外推缓存（"复用→预测"范式，D3 第三态的近邻）
- [From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/abs/2503.06923)

## C. 稀疏增量缓存（与 D3 的"规整 rank-r 优于不规则列稀疏 delta"论断正面对立）
- [Chipmunk: Training-Free Acceleration of Diffusion Transformers with Dynamic Column-Sparse Deltas](https://arxiv.org/abs/2506.03275)
- [Accelerating Diffusion Transformers with Token-wise Feature Caching](https://arxiv.org/abs/2410.05317)

## D. 自适应步长 / 阈值缓存（D3 要取代的"固定阈值"基线）
- [Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://arxiv.org/abs/2411.19108)
- [Adaptive Caching for Faster Video Generation with Diffusion Transformers](https://arxiv.org/abs/2411.02397)

---

# 下载结果（2026-09-22，7/7 成功）

- **PDF 存放**：`/data3/paper_analysis/papers_pdf/paper_dit_cache_comparators/`
- **机器可读记录**：同目录 `results.json`（`total=7, success=7, downloaded=7, failed=0`）
- **来源**：全部经 `source_link` 命中 arXiv PDF，无需标题模糊匹配、无手工补抓
- **完整性校验**：7 个文件均为有效 PDF，页数 11–32，逐一核对首页标题与目标论文一致

| # | 简称 | 标题 | arXiv | PDF 文件名 | 页数 | 大小 |
|---|---|---|---|---|---|---|
| 1 | DisCa | DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching | [2602.05449](https://arxiv.org/abs/2602.05449) | `DisCa Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching.pdf` | 18 | 2.4 MB |
| 2 | LearniBridge | LearniBridge: Learnable Calibration of Feature Caching for Diffusion Models Acceleration | [2606.26778](https://arxiv.org/abs/2606.26778) | `LearniBridge Learnable Calibration of Feature Caching for Diffusion Models Acceleration.pdf` | 11 | 20.3 MB |
| 3 | TaylorSeer | From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers | [2503.06923](https://arxiv.org/abs/2503.06923) | `From Reusing to Forecasting Accelerating Diffusion Models with TaylorSeers.pdf` | 15 | 33.5 MB |
| 4 | Chipmunk | Chipmunk: Training-Free Acceleration of Diffusion Transformers with Dynamic Column-Sparse Deltas | [2506.03275](https://arxiv.org/abs/2506.03275) | `Chipmunk Training-Free Acceleration of Diffusion Transformers with Dynamic Column-Sparse Deltas.pdf` | 32 | 44.2 MB |
| 5 | ToCa | Accelerating Diffusion Transformers with Token-wise Feature Caching | [2410.05317](https://arxiv.org/abs/2410.05317) | `Accelerating Diffusion Transformers with Token-wise Feature Caching.pdf` | 22 | 4.4 MB |
| 6 | TeaCache | Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model | [2411.19108](https://arxiv.org/abs/2411.19108) | `Timestep Embedding Tells It's Time to Cache for Video Diffusion Model.pdf` | 11 | 4.0 MB |
| 7 | AdaCache | Adaptive Caching for Faster Video Generation with Diffusion Transformers | [2411.02397](https://arxiv.org/abs/2411.02397) | `Adaptive Caching for Faster Video Generation with Diffusion Transformers.pdf` | 24 | 49.6 MB |

## ⚠️ 检索期即已发现的新颖性风险（未读全文，仅据摘要/页面）

这批论文是为了核验 [[调研报告-DiT多模态结构改动-训练保精度-kernel协同-当前版]] 方向 3 的三条新颖性主张（(a) rank-r 增量作中间态、(b) 门控可训练、(c) GPU 文生视频）。**抓取过程中两篇已构成直接威胁，读全文前不要动笔写 D3 的 intro**：

- **LearniBridge（ICML 2026）**：以 **LoRA 低秩更新**校准缓存特征，主张"feature reuse 所需的更新落在一个**跨 prompt 共享的低秩子空间**"，仅需 3–5 个训练样本；FLUX 5.87× / HunyuanVideo 5.75× / **Wan2.1 4.10×**。**同时压中 (a) 与 (b)。**
- **DisCa（CVPR 2026，腾讯混元，开源）**：**Distillation-Compatible Learnable** Feature Caching，单末层缓存 + 小型神经网络预测器，最高 **11.8×**。**压中 (b)，且"与蒸馏兼容"正对 §0.8-3 的"少步蒸馏吃同一份时间冗余"问题。**

另：**Chipmunk** 的"动态列稀疏 delta"是 D3"规整 rank-r 优于不规则列稀疏 delta"论断的正面对手，其 ColumnSparseAttn 声称 9.3× vs FlashAttention-3、ColumnSparseGEMM 2.5× vs cuBLAS —— **D3 的 C2 最优性论证必须与这两个数正面比较。**

## 后续命令（GPU 1，参照 scripts/README.md 正式链路）

```bash
cd /data3/paper_analysis

# 1) PDF → Markdown（GPU 1）
CUDA_VISIBLE_DEVICES=1 python3 scripts/pdf_to_md.py batch \
  /data3/paper_analysis/papers_pdf/paper_dit_cache_comparators \
  --output /data3/paper_analysis/papers_md/md_dit_cache_comparators \
  --workers 2 --skip-existing

# 2) 图片 OCR 原位插入（GPU 1）
python3 scripts/md_image_ocr_injector.py \
  --path /data3/paper_analysis/papers_md/md_dit_cache_comparators \
  --cuda-devices 1

# 3) 按一级标题拆分
python3 scripts/paper_mdsplit_batch.py \
  /data3/paper_analysis/papers_md/md_dit_cache_comparators \
  /data3/paper_analysis/paper_secs/secs_dit_cache_comparators

# 4) 顺序分析
#    ⚠️ run_all_papers.py 默认模型是 deepseek-v4-flash[1m]，但 .bashrc 不导出代理变量，
#       不手动注入则请求发到官方 API，日志出现 unrecognized_model，每篇立即失败（README §3.4）
deepseek-proxy-ensure
DS_TOKEN=$(grep -oP 'ANTHROPIC_AUTH_TOKEN="\K[^"]+' ~/.bashrc)
ANTHROPIC_BASE_URL="http://127.0.0.1:8787" \
ANTHROPIC_AUTH_TOKEN="$DS_TOKEN" \
ANTHROPIC_MODEL="deepseek-v4-flash[1m]" \
ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-pro[1m]" \
ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash[1m]" \
ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash[1m]" \
CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash[1m]" \
python3 scripts/run_all_papers.py \
  --paper-base-dir paper_secs/secs_dit_cache_comparators \
  --checkpoint-dir paper_extract_checkpoints/dit_cache_comparators \
  --output-repo-dir repos/repo_dit_cache_comparators

# 5) repo 汇总拆分入库
python3 scripts/repo_mdsplit_batch.py \
  /data3/paper_analysis/repos/repo_dit_cache_comparators \
  --notes-base /data3/paper_analysis
```
