## Chisel（Constructing Hardware in a Scala Embedded Language，Scala 嵌入硬件构造语言）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Chisel 是 UC Berkeley 开发的开源硬件构造语言（HCL）：以 Scala 嵌入 DSL 描述 RTL，经 elaboration 生成 FIRRTL IR，再经 FIRRTL 编译器产出 Verilog。相对直接写 Verilog，它提供参数化、面向对象、类型安全的电路构造；关键价值在于"可程序化变换"——生成流程中间有结构化 IR（FIRRTL），编译器 pass 可可靠地分析/改写电路，这是 Verilog 文本改写难以做到的（本论文 Discussion 明确对比：Verilog 下做同类变换需脆弱代码改写或自建工具链）。生态：Rocket Chip、riscv-mini、chiseltest；官方库 chisel-lang.org。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 本论文 AutoFHE 全流程的 Chisel 部分：设计者写明文语义的 Chisel（eNPU 案例起点为开源 Vector MulAdd Accelerator）→ 对密文输入调用 Secure(io.vec1)（自定义 API，类比 dontTouch）→ Chisel elaboration 阶段 Builder 创建 SecureAnnotation 记录元数据 → 序列化阶段 FIRRTL 后端产出 .fir（电路）+ .anno.json（注解 Target 字符串，如 VecUnit|VecUnit>io.vec1）→ 下游自定义 FIRRTL pass 按 Target 定位并改写电路。
- 标准注解-变换绑定模式（chipsalliance/chisel PR #393、BoringUtils/WiringTransform 同款）：
```
chisel3.experimental.annotate(
  new ChiselAnnotation with RunFirrtlTransform {
    def toFirrtl = SourceAnnotation(component.toNamed, id)
    def transformClass = classOf[MyTransform]
  })
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 自定义 pass 参数传递完全走注解（firrtl issue #552：不推荐命令行参数）；FIRRTL 驱动经 --custom-transforms 注入自定义 Transform。注意 Chisel 7 移除 ChiselAnnotation API（rocket-chip issue #3722），迁移到 InlineInstance 等新 API——AutoFHE 若随迁需换注解 API。dontTouch 是同类注解机制（标记信号为优化屏障）。AutoFHE 还使用 Chisel diagrammer 把电路可视化/图化供 FIRRTL 图分析。

FlashTFHE 补充视角（ISCA'26，Chisel 作为全芯片 RTL 实现与验证流）：整个 FlashTFHE 加速器（4 core ×（BRU + LPU + 内存子系统 + NoC））用 Chisel HDL 实现，功能单元 RTL 正确性在 Xilinx Virtex VU47P FPGA 上用 Beethoven 框架（Duke 的异构多核加速器系统组合器，ISPASS'25）验证；验证后的 Chisel 编译为 Verilog，再用 Synopsys Design Compiler 在 TSMC N16 综合至 1GHz（所有 worst-case negative slack 0.00ns）；scratchpad 内存用 Arm Artisan physical IP compiler 建模。Chisel 的参数化与可生成特性使设计空间探索（核数 4/8、round-robin ciphertext 数、buffer 大小）可快速重生成/重综合，这是直接写 Verilog 难以支持的。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
