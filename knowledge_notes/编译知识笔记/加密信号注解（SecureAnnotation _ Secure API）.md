## 加密信号注解（SecureAnnotation / Secure API）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AutoFHE 提出的声明式加密意图表达机制：设计者在 Chisel 里对携带密文的输入端口调用 Secure(io.vec1)，声明该信号为密文，不改动任何底层电路逻辑。实现：Secure 创建自定义 SecureAnnotation（继承 chisel3.experimental.ChiselAnnotation），复用 Chisel 原生注解基础设施（与 dontTouch 同源）把元数据传到后端，注解记录 Target 字符串定位 FIRRTL 信号。设计目标：既保留加密意图灵活性（哪些信号加密、哪些明文），又不牺牲开发流程透明性（Challenge 1）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 三步编译栈流程（论文 III-B）：① Chisel Elaboration 阶段——调用 Secure() 时 Chisel Builder 创建 SecureAnnotation 存元数据；② 序列化阶段——后端产出 .fir（电路）+ .anno.json（含 Target 的注解元数据）；③ FIRRTL Transformation 阶段——自定义 pass 读注解列表、按 Target 定位信号、修改电路支持密文计算；此后注解成为 BFS 识别加密组件的种子。
- 例子：加密 NPU 只对 io.vec1/io.vec2 加 Secure（权重保持明文）；加密查找引擎对 index 与 features 都加 Secure。不同加密策略通过注解位置表达，无需改架构。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与 ChiselAnnotation + RunFirrtlTransform 标准模式一致（见 Chisel 条目）；dontTouch 是同类注解机制。使用注意：注解只是提示（hint），实际加密组件集合由 FIRRTL 图分析自动补全，避免手工全标注的漏标错标；Chisel 7 移除 ChiselAnnotation API 后需换新注解 API（rocket-chip issue #3722）。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
