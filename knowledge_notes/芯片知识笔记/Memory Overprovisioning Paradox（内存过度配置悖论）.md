## Memory Overprovisioning Paradox（内存过度配置悖论）

术语是什么？通过联网搜索让回答具体和精准。

Memory Overprovisioning Paradox（内存过度配置悖论）是RPU论文提出的系统级设计效率问题：当为了获得更多memory bandwidth而扩展HBM stack数量时，同时也过度配置了远超过低batch LLM decode所需的memory capacity，导致系统成本和能耗不必要的膨胀。具体机制：如果模型能放入单个GPU，将socket数翻倍可加倍bandwidth，但也将每socket所需capacity减半。但当前HBM将bandwidth和capacity强绑定（HBM3e: 1280GB/s + 48GB per stack），系统架构师被迫"为bandwidth买capacity"——即使用capacity的利用率很低（Llama3-405B在8K seq len下仅利用HBM3e容量的~7.9%），却要支付capacity对应的硅面积、功耗和封装成本。

从芯片设计角度拆解：

该悖论根植于HBM的物理设计约束：HBM stack的bandwidth来自TSV（Through-Silicon Via）阵列的shoreline IO密度（每mm约102.5 GB/s），而capacity来自多层DRAM dies的3D堆叠（每die含多个ranks/banks/subarrays）。由于HBM标准将高bandwidth和高capacity设计目标不可分割地打包，在低batch decode这类capacity需求远低于bandwidth需求的场景中，DRAM内部的long wires（跨die跨subarray数据移动）产生不成比例的能耗（>74% memory device energy spent on internal data movement [43][45]），die area和cost也被大量"浪费"在未使用的存储单元上。

HBM-CO是论文对这一悖论的直接回应：通过chiplet-based modular memory architecture，将bandwidth和capacity解耦为可独立选择的两个维度——系统通过增加HBM-CO chiplet数量scale bandwidth，而每个chiplet的capacity通过上述结构缩减来控制。

术语一般如何实现？如何使用？

该术语本身是一个概念/观察，而非具体实现。它用于指导面向低batch LLM推理的memory系统设计决策——核心原则：memory provisioning应基于deployment scale（单node vs rack vs datacenter），而非无差别使用HBM3e导致过度配置。带宽应通过module数量扩展，capacity应通过每module的HBM-CO参数选择来匹配实际需求。

涉及论文标题：
- RPU - A Reasoning Processing Unit
