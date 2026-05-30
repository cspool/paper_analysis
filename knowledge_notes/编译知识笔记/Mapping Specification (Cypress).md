## Mapping Specification (Cypress)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mapping Specification是Cypress编程模型中与Logical Description并列的第二组件，定义了task-based程序到物理机器的绑定。其关键属性：(1) instance名——标识该task-mapping对象（可被其他instance的calls字段引用）；(2) variant——指定使用的task variant；(3) proc——指定执行的processor级别（HOST/BLOCK/WARPGROUP/WARP/THREAD）；(4) mems数组——每个tensor argument应物化在哪种memory（GLOBAL/SHARED/REGISTER/NONE）；(5) tunables——为task variant中的tunable变量提供具体值；(6) 可选控制——warpspecialize（是否warp specialize）、pipeline（pipeline深度）等。

核心设计理念：mapping decisions仅影响性能，不影响正确性——Cypress编译器保证任何有效mapping下生成的代码都保持了顺序语义的等价性。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Mapping Specification在Hopper GEMM编译中的作用：

```
Mapping控制的计算分解路径（Figure 5b）:

gemm_host (HOST):
  U=256, V=256  → C被分解为256×256 output tiles
  proc=HOST     → 执行在CPU host (kernel launch)
  调用 gemm_block (BLOCK):
    W=64        → K-reduction tile宽度64
    warpspec=T  → 启用warp specialization (DMA warp + compute wg)
    pipeline=3  → 3-deep software pipeline
    mems=[GLOBAL,GLOBAL,GLOBAL] → A,B,C tiles在global memory
    调用 gemm_tile (BLOCK):
      WGS=2     → split M维度到2个warpgroups
      mems=[NONE,SHARED,SHARED] → accumulator不在此级物化(NONE),
                                  A,B在shared memory
      调用 gemm_warpgroup (WARPGROUP):
        PIECES=4, PROC=WARP → 4 warp-level pieces
        mems=[NONE,SHARED,SHARED]
        调用 gemm_warp (WARP):
          PIECES=32, PROC=THREAD → 32 thread-level pieces
          调用 gemm_thread (THREAD):
            mems=[REGISTER,SHARED,SHARED] → accumulator在寄存器

NONE memory的作用:
  gemm_tile@BLOCK: mems[0]=NONE → accumulator不在block级别以完整形式物化，
  而是以partitioned form存在于各thread的寄存器中。
  若编译器无法满足NONE约束（如partition size超出寄存器容量），
  则report error → 用户需改变partition或mapping策略。
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mapping Specification的使用方式：
- 与Logical Description完全分离——改变mapping不需要修改算法代码
- 参数化设计允许系统性探索性能空间——改变tile sizes、pipeline depth、memory placement无需代码重写
- 可作为模板继承——类似task trees的mapping可被复用（但论文承认当前版本有冗余，未来可改进）
- Cypress编译器将mapping作为编译时配置，所有决策静态解析，无运行时开销

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs
