## Dyn-Loop and Dyn-Modi Dynamic PIM Instructions

术语是什么？通过联网搜索让回答具体和精准。

Dyn-Loop和Dyn-Modi是PIMphony compiler生成的两种动态PIM指令类型，用于在PIM hardware上实现与运行时token length相关的可变loop bound和可变operand addressing。传统PIM指令的loop count和operand address在编译期固定（绑定到Tmax），导致系统必须按最大context length静态预留KV cache。Dyn-Loop动态指令的loop bound来自请求当前token index Tcur（runtime值），而非编译期Tmax；Dyn-Modi动态指令在loop body内按stride自动修改row/col等operand field，形成逻辑virtual address。两者配合on-module dispatcher的VA→PA translation，实现真正的运行时dynamic addressing。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。

```
// Compiler生成Dyn-Loop和Dyn-Modi（编译时）:

// Dyn-Loop encoding:
//   opcode: DYN_LOOP
//   bound_src: REG_Tcur  // 引用runtime register存储的当前token length
//   body: [指令序列]      // loop body内的指令可以是Dyn-Modi

dyn_loop:
    bound = read_reg(REG_Tcur)  // runtime由host更新
    for i = 0 to bound:
        // Dyn-Modi encoding:
        //   opcode: MAC_DYN
        //   gbuf_entry: 0
        //   row_base: Tcur / (nCH * nBank)  // 逻辑row基址
        //   row_stride: 1
        //   col_base: 0
        //   col_stride: dh / 16  // 16个FP16 per 32B tile

        // Runtime dispatcher decode:
        //   va_row = row_base + i * row_stride
        //   va_col = col_base + i * col_stride
        //   pa_row, pa_col = VA2PA.translate(va_row, va_col, req_id)
        MAC(GBuf[0], DRAM[pa_row][pa_col], OBuf[out])

// 对比静态指令:
// 静态: for i = 0 to Tmax:  // Tmax编译期固定
//       MAC(GBuf[0], DRAM[base + i][col], OBuf[out])  // 地址编译期固定
// → Tmax=128K, Tcur=16K时→ 浪费112K iterations + 7/8的KV cache容量
```

Compiler的职责：识别哪些loop/operand需要动态化（attention中沿token维度的loop和KV cache address），生成Dyn-Loop/Dyn-Modi encoding，嵌入VA2PA table index reference。Runtime dispatcher的职责：在指令decode时从configuration buffer读取Tcur，执行VA→PA翻译，生成物理命令。Paper中DPA的chunk allocation策略（1MB chunk lazy alloc）通过host runtime在Tcur增长超过当前chunk容量时分配新chunk并更新VA2PA table实现。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Dyn-Loop/Dyn-Modi在compiler中以PIM ISA extension编码：新增opcode位区分静态/动态指令，动态指令的operand field不存储absolute address而存储base/stride配置和register reference。PIM controller在decode阶段识别动态指令→查询configuration buffer获取Tcur→计算virtual address→查询VA2PA table完成翻译→生成物理DRAM command。整个流程在PIM HUB的dispatcher pipeline中完成，增加少量cycle overhead（dispatcher lookup < 10 cycles per instruction per paper estimate）。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System
