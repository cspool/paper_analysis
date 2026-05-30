## Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

- 属于编译框架的实现是什么？实验比较什么？
  实现是Tilus，一个面向低精度GPGPU计算的tile级领域特定语言（DSL），由五个核心组件构成：(1) Python DSL——允许开发者用Python编写Tilus程序；(2) Intermediate Representation (IR)——VM指令集的中间表示；(3) 优化passes——消除冗余、简化算术表达式；(4) 代码生成器——将优化后的IR翻译为Hidet IR（类CUDA C的中间表示），应用低精度到标准精度的转换规则（使用PRMT、LOP3、bitwise指令做vectorized casting），生成CUDA C代码，由nvcc编译为硬件二进制；(5) 运行时系统——管理动态加载的二进制、workspace memory、execution context和kernel缓存。整个系统约35K行Python和C++代码。

  实验比较了Tilus与：vendor库cuBLAS、编译器Triton v3.1.0和Ladder（bitblas v0.0.1.dev15）、手写kernel QuantLLM（commit 9802c5a）和Marlin v0.1.1。operator实验覆盖uint8、f6e3m2、int4、uint4、uint2、uint1六种低精度类型，end-to-end实验将Tilus kernel集成至vLLM v0.5.3（artifact使用vLLM 0.7.3）与vLLM和Ladder对比。跨硬件实验在A100、L40S、H100上验证。性能提升：vs Triton 1.75×、vs Ladder 2.61×、vs QuantLLM 1.29×、vs Marlin 1.03×。

- 硬件平台是什么，配置是什么。
  主实验平台：NVIDIA L40S GPU (48 GiB)，GPU driver 565.57.01，CUDA Toolkit 12.6.3。跨硬件验证：NVIDIA A100 (Ampere)、NVIDIA H100 (Hopper)。软件依赖：PyTorch v2.5.1、Triton v3.1.0、BitBLAS v0.0.1.dev15、Marlin v0.1.1。

- 开源编译框架是什么。修改了什么。
  Tilus是自研的全新编译框架，非在现有编译器上修改。Tilus后端使用Hidet IR [14] 表示低级GPU代码，通过Hidet生成CUDA C后再用nvcc编译。核心创新在于：(1) 代数layout系统——通过参数化primitive layouts（local, spatial）和Kronecker product组合构建复杂tensor layouts，统一表示register tensor的跨线程元素分布；(2) thread-block级编程模型（SIMB）——显式暴露global memory、shared memory、registers三级内存层次；(3) 原生支持1-8 bit任意位宽的低精度数据类型（signed/unsigned int、float含任意exponent/mantissa分布）；(4) 编译流程中shared memory planner和global memory planner自动管理内存分配；(5) 自动向量化（cp.async.v4, lds128, ldg128）和指令选择（ldmatrix vs lds选择）。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  开源地址：https://github.com/NVIDIA/tilus（主仓库），artifact仓库：https://github.com/yaoyaoding/tilus-artifacts，DOI: 10.5281/zenodo.16756859，License: Apache 2.0。

  作用：Tilus的核心价值是以单一参数化程序模板高效支持全谱任意位宽（1-8 bit）低精度矩阵乘法，用户只需用Python编写tile级程序，编译器自动处理layout转换、指令选择、向量化和低精度转换。

  全过程（以FP16 × INT6矩阵乘法为例）：
  ```
  输入：Python Tilus程序（图2）
    ├── 定义: C[M,N] = A[M,K] × B[K,N]，A=f16, B=i6
    ├── Grid: (M/BM, N/BN) thread blocks
    ├── Layout: 使用Kronecker product构建register tensor layout
    │    例如: local(2,1) ⊗ spatial(8,4) ⊗ local(1,2)
    └── Body: LoadGlobal → View(r)einterpret → Cast → Dot(mma) → StoreGlobal

  Tilus编译流程:
  Step 1 — 全局/共享内存规划 (Global and Shared Memory Planning):
    分析程序中AllocateShared/AllocateGlobal指令
    计算总共享内存需求，映射shared tensor到kernel的shared memory区域
    运行时系统分配global memory workspace供AllocateGlobal使用

  Step 2 — 指令级代码生成 (Code Emitting):
    按顺序为每条Tilus指令生成Hidet IR代码
    指令选择: 若register tensor的layout与spatial(8,4).repeat(1,4)兼容 → 使用PTX ldmatrix
    否则使用PTX lds从shared memory加载
    自动向量化: LoadGlobal/StoreGlobal使用ldg128/stg128, 异步拷贝使用cp.async.v4

  Step 3 — 低精度类型降低 (Lowering Low-Precision Data Types):
    将低精度操作转换为硬件友好类型的对应操作
    由于layout系统和register tensor reinterpretation（§7.2），
    memory loading已由标准类型替代（u8 loading → View reinterpret to i6）
    仅需应用vectorized casting：使用PRMT（permute bytes in 32-bit register）、
    LOP3（任意三输入逻辑操作）、bitwise指令在registers内完成低精度→float16转换

  Step 4 — 运行时加载 (Runtime System):
    nvcc编译Hidet IR → 硬件二进制（.cubin）
    运行时动态加载编译后的二进制
    缓存已编译kernel避免重复编译
    维护workspace memory、execution context (CUDA stream)

  预处理步骤（低精度权重layout转换，图9）:
    在kernel启动前，将权重tensor从i6[K, N]变换为u8[K/BK, N/BN, BK*BN*6/8]
    消除非连续内存访问和bitwise extraction overhead
    使低精度权重加载与标准类型一样高效

  程序模板参数化（auto-tuning）:
    每种算子约200个配置（tile大小变体），编译时间约1分钟
    所有低精度类型共用同一个程序模板，仅改变tile大小参数
  ```
