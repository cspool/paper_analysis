## AU Arithmetic Intensity (ARI) for Operator-Level AU Selection

术语是什么？

ARI (Arithmetic Intensity for AU Selection) 是AUM论文提出的轻量级AU使用率判定指标，用于在operator级别选择最优AU (AMX vs AVX)。定义为：
- Prefill phase: ARI = 6(1/d + 3/BL)^(-1)，其中d=model hidden dimension, B=batch size, L=input sequence length
- Decode phase: ARI = 6(1/d + 3/B)^(-1)
ARI越高表示AU使用率U_AU越高——大d/大B/大L → higher ARI → 应使用AMX (TMUL compute-bound优势)；小d/小B/小L → lower ARI → 应使用AVX (避免AMX tile register配置overhead)。

从kernel调度角度拆解术语：

ARI-based AU selection的决策逻辑：
```
function select_au(phase, batch_size B, model_dim d, input_len L):
    if phase == "prefill":
        ARI = 6 * (1/d + 3/(B*L))^(-1)
        // d=4096, B=16, L=512 → ARI = 6/(0.000244+0.000366) = 9836
        // → High U_AU → AMX + High-AU region (2.1-2.5 GHz)
    elif phase == "decode":
        ARI = 6 * (1/d + 3/B)^(-1)
        // d=4096, B=16 → ARI = 6/(0.000244+0.1875) = 31.9
        // → Low U_AU → AVX + Low-AU region (2.8-3.1 GHz)
    
    return (U_AU, frequency_region)
    // U_AU determines:
    //   - AU choice: AMX vs AVX
    //   - Frequency region: C_H/C_L/C_N
    //   - Resource allocation: R_AU from AUV Model
```

术语一般如何实现？如何使用？

AUM Background Profiler offline: 对每个LLM operator计算ARI→判定U_AU→记录到AUV Model bucket。Runtime Controller online: 按bucket中U_AU分配频率region→查性能表P_a/P_t。ARI使AUM适应新模型仅需d, B, L参数（无需重profiling）。Paper基于先前研究推导的公式 [36][37]，与实测AMX cycle ratio (prefill 14.4%/decode 1.5%) 和 uop ratio (AMX FP ops: prefill 3.7%/decode 0.5%) 吻合。ARI阈值基于server-level AU usage distribution设定。

涉及论文标题：
- AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

