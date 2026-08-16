# 13 个 Direction Experiment Loop v7 最新启动命令

更新状态：2026-08-06，适用于 `direction_experiment_loop` format v7 和
`ATOMIC_DECISION_CONTRACT_V7`。已核对 13 个 `--direction-result` 均存在，下面使用的
`work-dir` 当前均未占用。这些命令用于创建新的 v7 run；不要对旧 format v2-v6 run
执行 `resume`，也不要对已经初始化的目录再次执行 `init`。

在 `/data3/paper_analysis` 下按编号串行运行。每条命令初始化一个
`Experiment Decision → Direction Lab Goal → Evidence Judge → Experiment Decision` run，
授权 5 个原子 Lab cycle；一次被接受的 `RUN_LAB` 合同消耗一个 cycle，同一 cycle 因
checkpoint 暂停后的恢复只创建新 invocation，不重复消耗 cycle。实验合同的调整不再
使用独立 Direction revision 预算，旧参数 `--max-revisions` 已移除。

Decision、Judge 和 Lab timeout 未显式填写时使用脚本当前默认值：Decision/Judge idle
5 分钟、hard 15 分钟，Lab idle 15 分钟、单次 invocation hard 6 小时。Lab hard timeout
只是 watchdog，并非实验科研预算。以下代码块可直接复制执行。

## 01. 缓存状态感知的多模态 DAG 剩余 Slack 排序

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-24da9cfe-2ca3-4f73-8a97-701b4208cb3b.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/01_cache_aware_dag_slack && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/01_cache_aware_dag_slack
```

## 02. MM-SP/TP 交叉切换

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-bde0b682-183a-4670-b037-151d742ee582.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/02_mmsp_tp_switch && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/02_mmsp_tp_switch
```

## 03. 分任务视觉 Token 预算开关

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-893f0c70-4cc2-44bc-8052-2e852f08339a.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/03_task_visual_token_budget && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/03_task_visual_token_budget
```

## 04. 共享层序列自投机解码

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-1440de74-478b-4ae8-a5f9-7761c3f16186.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/04_shared_layer_self_spec_decode && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/04_shared_layer_self_spec_decode
```

## 05. DeepSeek-VL2-Tiny 统一 Top-4

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-fbc67f43-86c6-4897-99ba-3abc185e8001.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/05_deepseek_vl2_top4 && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/05_deepseek_vl2_top4
```

## 06. 视觉前缀专属 INT4 KV 驻留

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-63af3818-9a99-40c6-9a92-692c7b810a5d.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/06_visual_prefix_int4_kv && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/06_visual_prefix_int4_kv
```

## 07. 提示 KV 隔步刷新

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-3f4bc947-4666-4582-8f2d-0a8fa56cf51a.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/07_bounded_stale_prompt_kv_refresh && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/07_bounded_stale_prompt_kv_refresh
```

## 08. SIC 逐 Tile 盈亏门控

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-417eb71e-e396-47f9-870f-f1deb38875fd.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/08_sic_tile_profit_gate && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/08_sic_tile_profit_gate
```

## 09. AR 图捕获尺寸集

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-f9182846-6b0d-4a9c-a511-ebe7d4c4aebb.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/09_ar_graph_capture_set && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/09_ar_graph_capture_set
```

## 10. MoDM GPU 分配器

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-34fd54f4-1ce5-4f0b-b384-3c2c0028dad6.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/10_modm_oldest_age_allocator && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/10_modm_oldest_age_allocator
```

## 11. dTask 窗口关闭

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-47381a36-6bc5-46a7-bdc0-23325c4f5e39.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/11_dtask_slo_slack_window && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/11_dtask_slo_slack_window
```

## 12. 曲率事件锚保护的视觉 KV 驱逐

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-257afdaa-3769-42fe-880d-252ccb5da0f6.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/12_curvature_anchor_kv_eviction && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/12_curvature_anchor_kv_eviction
```

## 13. 主动感知截止门

```bash
node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first_v10_negative_convergence_reset_20260803/results/turn-7087e8ee-3224-4b2e-872d-b1136c02ab6e.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/13_active_perception_deadline_gate && \
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/13_active_perception_deadline_gate
```

## 暂停后恢复

普通暂停、Lab 已写入 checkpoint，或 provider 中断后没有最终结果时，使用：

```bash
node scripts/direction_experiment_loop.ts resume --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/01_cache_aware_dag_slack
```

恢复会复用同一个持久 Lab Goal/thread，并为本次执行建立新的 invocation、deadline 和
运行日志；脚本只补做 checkpoint 中尚未完成的工作。

若暂停原因是 Lab cycle 授权耗尽：

```bash
node scripts/direction_experiment_loop.ts resume --yolo \
  --additional-cycles 2 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/01_cache_aware_dag_slack
```

## 运行中检查与安全暂停

在另一个终端查看状态，不会停止当前实验：

```bash
node scripts/direction_experiment_loop.ts status \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/01_cache_aware_dag_slack
```

请求当前 Lab 在安全边界写入结果或 checkpoint 后暂停：

```bash
node scripts/direction_experiment_loop.ts pause \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/01_cache_aware_dag_slack
```

前台按 `Ctrl-C` 与上述 `pause` 使用同一条优雅暂停路径。若 Lab 已原子写入非空
`result.md`，脚本会优先接管结果并交给 Judge；若只有绑定正确的 `checkpoint.json`，
则保存恢复点后暂停；两者都没有时会异常暂停，不会伪造实验结果。

## 完成后校验

```bash
node scripts/direction_experiment_loop.ts validate \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/serial_direction_experiment_atomic_v7_20260806/01_cache_aware_dag_slack
```

终态报告位于对应目录的 `final/report.md`，机器可读 handoff 位于
`final/handoff.json`。把示例中的 `01_cache_aware_dag_slack` 替换为其余编号目录即可用于
对应 Direction。
