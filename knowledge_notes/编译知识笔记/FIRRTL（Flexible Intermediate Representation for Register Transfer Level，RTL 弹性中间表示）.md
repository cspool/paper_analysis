## FIRRTL（Flexible Intermediate Representation for Register Transfer Level，RTL 弹性中间表示）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FIRRTL 是 Chisel 生态的 RTL 中间表示（Li, Izraelevitz, Bachrach, UCB/EECS-2016-9）：介于 HCL 与 Verilog 之间的结构化 IR（.fir 文本），配套编译器支持 pass 式变换（展开 when、推断时钟/复位、位宽推断、降级），并提供 Annotation 系统把元数据从 Chisel 一路携带到后端 pass——这是"元数据驱动电路改写"的基础设施。本论文中 FIRRTL 是自动识别与改写的载体：AutoFHE 在 FIRRTL 图上做加密组件识别（BFS 标记），并注入自定义 transformation（--custom-transforms）生成 FHE 加速器。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 流程：HCL →（Chisel Convert phase）→ FIRRTL 电路 + Annotation 文件（.anno.json）→ 标准 passes（uniquify/flatten 等）→ AutoFHE 自定义 pass（读 SecureAnnotation → 图遍历标密 → 算子-CPE 映射 → CPE 虚拟化调度固化）→ 降级 → Verilog。
- 例子（论文图 5）：io.vec1 标 Secure 后，res(i) := io.vec1(i)*io.vec2(i)+const.U 的乘法器与 res 在 FIRRTL 图中被隐式标记为加密组件（算子级），整体被替换/绑定到 ciphertext arithmetic CPE。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源实现：chipsalliance/firrtl（Scala）与 CIRCT 的 FIRRTL dialect（MLIR）。注解以 JSON 序列化（.anno.json），每条含 Target（Named 定位信号，如 Module|Module>io.sig）；自定义 Transform 继承 firrtl.passes.Transform 注册后经 --custom-transforms 调用，且 transform 须消费并移除所用注解（CIRCT FIRRTLAnnotations 文档）。AutoFHE 组件（识别/DSE/调度/模拟器）用 Python，仅生成部分走 Chisel/FIRRTL。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
