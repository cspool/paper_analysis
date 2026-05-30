## Token Control Logic Module (TCLM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Control Logic Module (TCLM) 是 Squat 论文中实现Token自适应量化的运行时控制模块。TCLM负责在推理（和训练）的每一步中：(1)根据最新注意力图的初始token列评估每个token的重要性；(2)通过Heapsort快速TopK选择指定比例的重要token；(3)将重要token（8-bit）和非重要token（4-bit）分别拼接分组；(4)调度对应的multiplier（INT8 multiplier和INT4 multiplier in MKMP）执行混合精度MAC。TCLM本身作为轻量级逻辑控制单元，其Heapsort和Concatenation开销可忽略（论文称"marginal overhead"）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TCLM在推理过程中的伪代码：
```
// TCLM: Token Control Logic Module
// 输入: attention_map [N, N], activations [N, D], rho (重要比例)
// 输出: 分组后的量化激活 + 位宽mask

struct TCLMOutput {
    int8_t* x_8bit_group;   // 8-bit量化后拼接的重要tokens
    int4_t* x_4bit_group;   // 4-bit量化后拼接的非重要tokens
    int* index_map;          // 结果排序的索引映射 [N]
    int n_8, n_4;            // 各组token数
};

TCLMOutput tclm_forward(float* attn, float* x, int N, int D, float rho) {
    // Step 1: 评估重要性
    float scores[N];
    for (int i = 0; i < N; i++)
        scores[i] = attn[i * N];  // attn[:, 0], 对初始token的注意力
    
    // Step 2: Heapsort TopK (O(N log k), k = rho*N)
    int k = (int)(rho * N);
    float threshold = heapsort_topk(scores, N, k);
    
    // Step 3: 分组 + 拼接
    vector<float> grp8, grp4;
    vector<int> idx8, idx4;
    for (int i = 0; i < N; i++) {
        if (scores[i] >= threshold) {
            grp8.insert(grp8.end(), &x[i*D], &x[(i+1)*D]);
            idx8.push_back(i);
        } else {
            grp4.insert(grp4.end(), &x[i*D], &x[(i+1)*D]);
            idx4.push_back(i);
        }
    }
    
    out.x_8bit_group = layer_wise_quant8(grp8.data(), grp8.size());
    out.x_4bit_group = layer_wise_quant4(grp4.data(), grp4.size());
    out.n_8 = idx8.size(); out.n_4 = idx4.size();
    // index_map记录原始顺序，用于合并时恢复
    merge_index_map(out.index_map, idx8, idx4);
    return out;
}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TCLM实现要点：(1)Heapsort TopK是标准C++算法库函数，开销O(N log k) vs 全排序O(N log N)；(2)分组拼接操作可通过memcpy或指针重排实现，不涉及计算；(3)在MKMP multiplier中，TCLM的输出直接路由到对应的INT8或INT4 kernel，kernel间无缝衔接。训练时每步调用TCLM（基于当前注意力图），推理时同样动态执行。TCLM本身无训练参数，仅做逻辑控制。Squat中的TCLM实现：代码开源https://github.com/shawnricecake/squant。

涉及论文标题：
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
