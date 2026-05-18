## Dynamic PIM Access (DPA)

术语是什么？通过联网搜索让回答具体和精准。

Dynamic PIM Access（DPA）是PIMphony提出的动态KV cache内存管理机制，通过两类动态PIM指令和on-module dispatcher实现运行时virtual-to-physical地址翻译，使KV cache可以按实际token length以1MB chunk lazy allocation而非按最大context length Tmax静态预留。Dyn-Loop指令的loop bound来自请求当前token index Tcur（而非编译期Tmax），Dyn-Modi指令在loop内按stride自动修改row/col等operand field形成逻辑virtual address。On-module dispatcher在PIM HUB内查询VA2PA table将virtual address翻译到已分配的物理KV cache chunk地址。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// DPA执行流程:

// 1. Compiler生成DPA指令（编译时，不绑定物理地址）:
//    Dyn-Loop: for i in 0..Tcur:  // Tcur在runtime确定
//    Dyn-Modi: row = (Tcur / (nCH * nBank)) + i * stride_row
//              col = (Tcur % (nCH * nBank)) + i * stride_col
//    MAC(GBuf[0], K_virtual[row][col], OBuf[out])

// 2. Runtime VA→PA translation (on-module dispatcher):
function translate(va_row, va_col, req_id):
    chunk_id = va_row / CHUNK_SIZE_IN_ROWS  // 确定目标chunk
    pa_base = VA2PA[req_id][chunk_id]        // 查表得物理chunk基址
    if pa_base == NULL:
        // 需要新chunk → signal host
        signal_host_alloc(req_id, chunk_id)
        pa_base = wait_for_alloc()
    pa_row = (va_row % CHUNK_SIZE_IN_ROWS) + pa_base
    return pa_row, va_col  // col通常不变

// 3. Chunk lifecycle:
//    请求进入: host alloc chunk_0 → update VA2PA
//    请求增长: Tcur增加→需要新chunk → host alloc chunk_1 → update VA2PA
//    请求结束: host free all chunks of this request

// 对比静态分配:
// 静态: pre-alloc Tmax * dh * sizeof(FP16) → 大量浪费
// DPA:  alloc ceil(Tcur / CHUNK_CAPACITY) chunks → 仅最后chunk有碎片
```

DPA的capacity utilization提升：静态方案在不同workload上仅31.0%-40.5% utilization（因按Tmax预留），DPA达到平均75.6% utilization。关键trade-off：chunk粒度1MB——过大则碎片多，过小则VA2PA table entry多和host-PIM通信频繁。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DPA由compiler（生成Dyn-Loop/Dyn-Modi指令编码）、on-module dispatcher hardware（VA2PA lookup + address translation）和host runtime（chunk allocation/release + VA2PA table update）三层协同实现。Compiler不枚举物理地址，而是生成参数化指令；runtime在请求进入/增长/结束时更新mapping；dispatcher在每条指令decode时执行轻量级翻译。Paper在CENT和NeuPIMs simulator中建模dispatcher的VA2PA lookup延迟和chunk allocation通信开销。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

