# GPU WMMA和Tensor Core

ref：Modeling Deep Learning Accelerator Enabled GPUs

## wmma api功能

通过WMMA API，开发者可将D = A × B + C当作warp操作，其中的A、B、C、D都是更大矩阵的tile。通过WMMA API，**warp的所有线程可以合作完成在这些tile上的矩阵乘加操作。CUDA 9.0 WMMA API的tile大小有限制为16×16×16**。tile的大小用M×N×K表示，A的维度是M×K，B的维度是K×N，C和D的维度是M×N。

每个tile可以进一步分割为fragment（小块reg阵列），每个fragment是映射到线程寄存器的一组tile元素。因此，输入矩阵的分布是跨线程的，每个线程只包含一部分tile。

一个16×16的tile包含256个元素。warp（包括32个线程）中的每个线程在8个GPR（General-Purpose Register）中保存一个8（256/32=8）元素的fragment。

**注意：Tile只是任务分块大小，不代表硬件并行度。**

**每个warp的tile为16*16*16，而每个TensorCore每周期完成一个4*4的矩阵乘和加(MACC)，直观看每个tile需要64个tensor core的MACC操作.若tile等于并行度，则每个warp需要64个TensorCore。**

**但每个warp只有2个TensorCore，并且每个TensorCore的MAC并行度是64（FP32）或128（FP16），每周期完成的4*4和4*4的矩阵乘法（FP16）对应的MAC操作数就是128。**

> **[图片提取文字 (image.png)]:**
> ## L1 指令缓存
> 
> ![](_page_0_Figure_2.jpeg)
> 
> | 线程束调度器 (32 线程/clk) |        |                  |              |                  |              |
> |--------------------|--------|------------------|--------------|------------------|--------------|
> | 分配单元(32 线程/clk)    |        |                  |              |                  |              |
> | FP64               |        |                  | FP32         |                  |              |
> | FP64               |        |                  | FP32         |                  |              |
> | FP64               | INT II | NT FP32          | FP32         |                  |              |
> | FP64               | INT II | NT FP32          | FP32         | TENSOR<br>核心     | TENSOR<br>核心 |
> | FP64               |        | # Total Total    | FP32         | 核心               | 核心           |
> | FP64<br>FP64       |        | -                | FP32<br>FP32 |                  |              |
> | FP64               | INT II | NT FP32          | FP32         |                  |              |
> | LD/ LD/<br>ST ST   |        | LD/ LD/<br>ST ST | LD/<br>ST    | LD/ LD/<br>ST ST | SFU          |
> 
> L0 指令缓存
> 
> 线程束调度器(32线程/clk)
> 
> 分配单元(32 线程/clk)
> 
> 寄存器堆(16,384 x 32 位)
> 
> FP32 FP32
> 
> INT
> 
> FP64
> 
> ED64
> 
> INT
> 
> L0 指令缓存
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image.png)

## API函数

CUDA WMMA API提供三个新方法：load_matrix_sync,、store_matrix_sync 和mma_sync。

这三个方法在计算出结果前会执行一个隐含的warp barrier同步。

load_matrix_sync,、store_matrix_sync方法用于载入和保存线程可访问GPR中的一部分输入矩阵。

mma_sync方法执行warp同步矩阵乘加操作，在GPR中产生一个M×N（如16×16）的结果D矩阵。

为了在PTX级别执行操作Tensor Core，在PTX 6.0引入了三个PTX指令，如下所示：
wmma.load.a.sync.layout.shape.type ra, [pa] {stride};
wmma.load.b.sync.layout.shape.type rb, [pb] {stride};
wmma.load.c.sync.layout.shape.type rc, [pc] {stride};
wmma.mma.sync.alayout.blayout.shape.dtype.ctype rd, ra, rb, rc;
wmma.store.d.sync.layout.shape.type rd, [pd] {stride};

其中，“sync”标识符表示指令等待warp中所有线程同步后才开始执行。PTX手册中将tile称为“操作数矩阵”。“layout”标识符标识操作数矩阵是以行主序或列主序的形式保存在内存中。“shape”标识符表示操作数矩阵的fragment大小（如，16×16×16表示为m16n16k16）。“type”标识符表示操作数矩阵的精度，如FP16或FP32。在Volta架构中，矩阵A和B必须是FP16，但C可以是FP16或FP32。

在矩阵乘操作之前，操作数矩阵A、B、C必须从内存加载到寄存器文件中，这由三个PTX指令wmma.load.a、wmma.load.b、wmma.load.c完成。wmma.load.a将矩阵A加载到寄存器ra中，wmma.load.b将矩阵B加载到寄存器rb中，wmma.load.c将矩阵C加载到寄存器rc中。ra、rb、rc表示GPR集合，这些GPR集合分布跨warp（对应fragment，每个warp线程持有一个fragment）线程。PTX指令中的pa、pb、pc代表保存操作数矩阵A、B、C的内存地址。

从内存载入的输入tile是一个更大矩阵的一部分。为了帮助访问tile，wmma.load和wmma.store支持stride内存访问。PTX指令中的“stride”操作数指定了每行/列的起始位置。

wmma.mma指令执行warp级别的矩阵乘累加操作。这个指令使用寄存器a、b、c分别保存矩阵A、B、C，计算结果保存在寄存器d中。

**每个时钟周期，每个Tensor Core可完成一个4×4矩阵乘累加（MACC，Matrix multiply and Accumulation）计算。**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3: Tensor cores complete one  $4 \times 4$  MACC operation per cycle (D = A \* B + C). Reproduces Figure 8 in [26].
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image%201.png)

在Tensor Core的MACC操作中，矩阵A的fragment包括8个FP16**2元素（即16个FP16元素），矩阵B的fragment包括另一8个FP16**2元素，以及针对FP16累加的4个FP16*2元素，或针对FP32累加的8个FP32元素。**（涉及tensorCore如何存取每个fragment进行子块计算，需要进一步研究）**

## WMMA中矩阵乘的加载、存储过程

下图中大矩形1表示操作数矩阵A、B，其中较小的方形表示操作数矩阵中的元素，位于同一行的元素在内存中位置连续。每个threadgroup（warp的32个线程分为8个threadgroup，每个threadgroup包含4个线程）加载一个4×16子矩阵，这个子矩阵称为segment（对应大矩阵1中的4个不同色块）。4个segment组成了一个操作数矩阵。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> | (a) Operand | matrices | A | and | В. |
> |-------------|----------|---|-----|----|
> |-------------|----------|---|-----|----|
> 
> Figure 7: Distribution of operand matrix elements to threads for Tensor Cores in the Titan V (Volta).
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Matrix C Distribution within a Warp (FP32 and FP16)
> 
> (b) Operand matrix C.
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image%202.png)

上图中的矩阵2、3显示了segment中的元素在threadgroup中的各线程间的分布。对于Volta，每个segment由两个不同threadgroup加载，即，A、B矩阵的元素由同一warp中的两个不同线程加载。例如，A矩阵的前四行由threadgroup 0和threadgroup 2加载。

行主序布局保存的A矩阵元素的线程分布和列主序布局保存的B矩阵元素的线程分布相同。对于行主序布局的矩阵A ，threadgroup中的每个线程使用2个合并的（coalesced）128位宽load指令加载16个连续元素（图中标记2）。对于列主序布局，threadgroup中的每个线程使用4个合并的64位宽load指令加载4块4个连续元素（图中标记3），每个load指令的stride是64个元素。对于矩阵C，每个threadgroup加载1个8×4的segment。

**wmma.load和wmma.store PTX指令拆分为一组SASS load（LD.E.64，LD.E.128，LD.E.SYS）和store（ST.E.SYS）指令。**这意味着Tensor Core是直接从GPU寄存器文件访问操作数矩阵。wmma.load.c拆分为一组LD.E.SYS指令。

对于矩阵A、B， **wmma.load拆分为4个64位宽load指令（LD.E.64）或2个128位宽load指令（LD.E.128），视矩阵布局是行主序还是列主序而定。**

## WMMA中MMA的计算

wmma.mma PTX指令通过HMMA SASS指令实现。

下图展示**一条PTX.mma指令对应的HMMA指令**.每个HMMA指令有4个操作数，每个操作数使用一对相邻寄存器，但在HMMA指令中只用一个寄存器的标识符表示。例如，在指令“HMMA.884.F32.F32.STEP0 R8, R24.reuse.COL, R22.reuse.ROW, R8”中的“R8”表示**warp寄存器**对< R8, R7>。类似地，剩余寄存器标识符表示3对源操作数寄存器< R24, R23>、< R22, R21>、< R8, R7>。4对寄存器对对应矩阵A、B、C、D。

**注意：warp寄存器每个32bit，是warp内线程共享，理解为专为Tensor Core服务；通用寄存器每个一般32bit，是线程独立持有。**

> **[图片提取文字 (image.jpeg)]:**
> | 0     | 1  | 2  | 3<br>7<br>11<br>15<br>19<br>23 |  |
> |-------|----|----|--------------------------------|--|
> | 4     | 5  | 6  |                                |  |
> | 8     | 9  | 10 |                                |  |
> | 12    | 13 | 14 |                                |  |
> | 16    | 17 | 18 |                                |  |
> | 20    | 21 | 22 |                                |  |
> | 24 25 |    | 26 | 27                             |  |
> | 28    | 29 | 30 | 31                             |  |
> 
> | 0 | 4 | 8  | 12 | 16 | 20 | 24 | 28 |
> |---|---|----|----|----|----|----|----|
> | 1 | 5 | 9  | 13 | 17 | 21 | 25 | 29 |
> | 2 | 6 | 10 | 14 | 18 | 22 | 26 | 30 |
> | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 |
> 
> Fig. 1: Row-major order (left) and column-major order (right) to store one  $8 \times 8$  half precision matrix. Number in the cell is the lane id. One 32-bit thread register stores two half elements. 32 threads within a warp can store 64 elements of the matrix.
![image.jpeg](GPU%20WMMA%E5%92%8CTensor%20Core/image.jpeg)

> **[图片提取文字 (image.jpeg)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 2: Distribution of matrix elements in resigers for HMMA.1688.F16 R0, R2, R6, R4. Each colored  $8 \times 8$  square is an  $8 \times 8$  matrix. We label the register index next to the coresponding matrix. The second source matrix (R6) needed to be stored in column-major order.
![image.jpeg](GPU%20WMMA%E5%92%8CTensor%20Core/image%201.jpeg)

指令中的“reuse”表示相关操作数在下一步中会被重用，因此缓存在操作数重用cache中，可避免一次寄存器获取（register fetch）并降低GRF bank conflict可能性。

> **[图片提取文字 (image.png)]:**
> ```
> 10
>        HMMA.884.F32.F32.STEPO R8, R24.reuse.COL, R22.reuse.ROW, R8;
>                                                                         12
>        HMMA.884.F32.F32.STEP1 R10, R24.reuse.COL, R22.reuse.ROW, R10;
> SET1.
>                                                                         14
>        HMMA.884.F32.F32.STEP2 R4, R24.reuse.COL, R22.reuse.ROW, R4;
>                                                                         18
>       _HMMA.884.F32.F32.STEP3 R6, R24.COL, R22.ROW, R6;
>        HMMA.884.F32.F32.STEP0 R8, R20.reuse.COL, R18.reuse.ROW, R8;
>                                                                         20
>        HMMA.884.F32.F32.STEP1 R10, R20.reuse.COL, R18.reuse.ROW, R10;
>                                                                         22
>        HMMA.884.F32.F32.STEP2 R4, R20.reuse.COL, R18.reuse.ROW, R4;
>                                                                         2.4
>        HMMA.884.F32.F32.STEP3 R6, R20.COL, R18.ROW, R6;
>                                                                         28
>        HMMA.884.F32.F32.STEP0 R8, R14.reuse.COL, R12.reuse.ROW, R8;
>                                                                         30
>        HMMA.884.F32.F32.STEP1 R10, R14.reuse.COL, R12.reuse.ROW, R10;
>                                                                         32
>        HMMA.884.F32.F32.STEP2 R4, R14.reuse.COL, R12.reuse.ROW, R4;
>                                                                         34
>        HMMA.884.F32.F32.STEP3 R6, R14.COL, R12.ROW, R6;
>                                                                         38
>        HMMA.884.F32.F32.STEP0 R8, R16.reuse.COL, R2.reuse.ROW, R8;
>                                                                         40
>        HMMA.884.F32.F32.STEP1 R10, R16.reuse.COL, R2.reuse.ROW, R10;
>                                                                         42
>        HMMA.884.F32.F32.STEP2 R4, R16.reuse.COL, R2.reuse.ROW, R4;
>                                                                         44
>       HMMA.884.F32.F32.STEP3 R6, R16.COL, R2.ROW, R6;
>                                                                         54
> ```
> 
> ## (a) Disassembled SASS instructions for Mixed precision mode
> 
> ```
> Cumulative
>                                                               Clock Cycles
> HMMA.884.F16.F16.STEP0 R4, R22.reuse.T, R12.reuse.T, R4;
>                                                                    12
> HMMA.884.F16.F16.STEP1 R6, R22.T, R12.T, R6;
>                                                                    2.1
> HMMA.884.F16.F16.STEP0 R4, R16.reuse.T, R14.reuse.T, R4;
>                                                                    25
> HMMA.884.F16.F16.STEP1 R6, R16.T, R14.T, R6;
>                                                                    34
> HMMA.884.F16.F16.STEP0 R4, R18.reuse.T, R8.reuse.T, R4;
>                                                                    38
> HMMA.884.F16.F16.STEP1 R6, R18.T, R8.T, R6;
>                                                                    47
> HMMA.884.F16.F16.STEP0 R4, R2.reuse.T, R10.reuse.T, R4;
>                                                                    51
> HMMA.884.F16.F16.STEP1 R6, R2.T, R10.T, R6;
>                                                                    64
> ```
> 
> ## (b) Disassembled SASS instructions for FP16 mode
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image%203.png)

对于混合精度的Volta，每条wmma.mma指令拆成4组共16条HMMA指令，每组4条HMMA指令。每条HMMA指令都有“STEP<n>”标记，n从1到3。对于FP16，每条wmma.mma指令拆成4组共8条HMMA指令，每组2条HMMA指令。

当执行HMMA指令时，每个threadgroup将A矩阵中的一个4×4子块（sub-tile）与B矩阵中的一个4×8子块相乘，然后将乘积与C矩阵累加。如下图所示：

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Elements accessed in each "Set"
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image%204.png)

更具体地说，当threadgroup 0执行Set 0的HMMA指令（如下所示）时，将包含矩阵A的前4行和前4列的子块与包含矩阵B的前4行和前8列的子块相乘，其乘积与矩阵C的4×8子块累加，得到的和保存在矩阵D的4×8子块中，即上图第一行计算过程。

HMMA.884.F32.F32.STEP0 R8, R24.reuse.COL, R22.reuse.ROW, R8;
HMMA.884.F32.F32.STEP1 R10, R24.reuse.COL, R22.reuse.ROW, R10;
HMMA.884.F32.F32.STEP2 R4, R24.reuse.COL, R22.reuse.ROW, R4;
HMMA.884.F32.F32.STEP3 R6, R24.COL, R22.ROW, R6; HMMA

下图显示了在混合精度模式下，threadgroup 0的一组指令中的每一HMMA step的操作，每一组指令有4个step，如上例中的STEP0~3。在每一个step中，矩阵A的一个2×4子块与矩阵B的一个4×4子块相乘，其乘积与矩阵C的2×4子块累加。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (b) Elements accessed in each "Step" (mixed-precision mode).
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image%205.png)

类似地，下图显示了在FP16精度模式下，threadgroup 0的一组指令中的每一HMMA step的操作，每一组指令有2个step，而不是混合精度下的4个step。在每一个step中，矩阵A的一个4×4子块与矩阵B的一个4×4子块相乘，其乘积与矩阵C的4×4子块累加。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (c) Elements accessed in each Step (FF10 mode).
> 
> Figure 10: HMMA instruction analysis for Volta (Titan V).
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image%206.png)

## TensorCore计算时的数据访问过程

为了确定线程0如何加载操作数矩阵元素，可以改变元素值并观察对结果的影响。可以发现，threadgroup是成对工作并计算得到8×8子块结果。这样的一对threadgroup称为octet，一个warp中有4个octect（1 warp = 32线程 = 8 threadgroup = 4 octet）。

下表显示了构成每个octet的threadgroup对，可用如下公式表示octet的构成方式

Octet X = threadgroup X ∪ threadgroup X+4 X∈[0, 3]

下表的第三、四列表示每个octet中的线程访问的矩阵A、B的子块。

下表中显示，矩阵A、B的每个元素被不同threadgroup中的线程加载两次，即，每个octet读入矩阵A的一个8×16子块、矩阵B的一个16×8子块，以及矩阵C的8×8子块。

> **[图片提取文字 (image.png)]:**
> | Octet | Threadgroup | Matrix A    | Matrix B    |
> |-------|-------------|-------------|-------------|
> | 0     | 0 and 4     | [0:7,0:15]  | [0:15,0:7]  |
> | 1     | 1 and 5     | [8:15,0:15] | [0:15,0:7]  |
> | 2     | 2 and 6     | [0:7,0:15]  | [0:15,8:15] |
> | 3     | 3 and 7     | [8:15,0:15] | [0:15,8:15] |
> 
> Table II: Octet composition and elements accessed
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image%207.png)

为了更好地理解octet中线程的组织方式，以下分析了octet在不同set和step执行的计算，如下图所示。在set1中，子块[a]、[e]和[A]、[E]之间的乘积需要生成部分结果[aA]、[aE]、[eA]、[eE]。以计算[aE]为例，threadgroup 0需要用到矩阵B的子块[E]，[E]由threadgroup 4加载。类似地，为了计算[eA]，threadgroup 4需要用到矩阵B的子块[A]，[A]由threadgroup 0加载。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Elements of operand matrices accessed by each octet
> 
> (b) Outer product formulation during sets and steps in an octet
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image%208.png)

下表展示上图b中的计算全过程：

> **[图片提取文字 (image.png)]:**
> | SET | STEP | Threadgroup X     | Threadgroup X+4   |
> |-----|------|-------------------|-------------------|
> | 1   | 0    | $a[0:1] \times A$ | $e[0:1] \times A$ |
> |     | 1    | $a[2:3] \times A$ | $e[2:3] \times A$ |
> |     | 2    | $a[0:1] \times E$ | $e[0:1] \times E$ |
> |     | 3    | $a[2:3] \times E$ | $e[2:3] \times E$ |
> | 2   | 0    | $b[0:1] \times B$ | $f[0:1] \times B$ |
> |     | 1    | $b[2:3] \times B$ | $f[2:3] \times B$ |
> |     | 2    | $b[0:1] \times F$ | $f[0:1] \times F$ |
> |     | 3    | $b[2:3] \times F$ | $f[2:3] \times F$ |
> | 3   | 0    | $c[0:1] \times C$ | $g[0:1] \times C$ |
> |     | 1    | $c[2:3] \times C$ | $g[2:3] \times C$ |
> |     | 2    | $c[0:1] \times G$ | $g[0:1] \times G$ |
> |     | 3    | $c[2:3] \times G$ | $g[2:3] \times G$ |
> | 4   | 0    | $d[0:1] \times D$ | $h[0:1] \times D$ |
> |     | 1    | $d[2:3] \times D$ | $h[2:3] \times D$ |
> |     | 2    | $d[0:1] \times H$ | $h[0:1] \times H$ |
> |     | 3    | $d[2:3] \times H$ | $h[2:3] \times H$ |
> 
> Table III: Octet computation details
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image%209.png)

> **[图片提取文字 (image.png)]:**
> ```
> HMMA.884.F32.F32.STEP0 R8, R24.reuse.COL, R22.reuse.ROW, R8;
> HMMA.884.F32.F32.STEP1 R10, R24.reuse.COL, R22.reuse.ROW, R10;
>                                                                  14
> HMMA.884.F32.F32.STEP2 R4, R24.reuse.COL, R22.reuse.ROW, R4;
>                                                                  18
> HMMA.884.F32.F32.STEP3 R6, R24.COL, R22.ROW, R6;
>                                                                  20
> HMMA.884.F32.F32.STEP0 R8, R20.reuse.COL, R18.reuse.ROW, R8;
>                                                                  2.2
> HMMA.884.F32.F32.STEP1 R10, R20.reuse.COL, R18.reuse.ROW, R10;
>                                                                  24
> HMMA.884.F32.F32.STEP2 R4, R20.reuse.COL, R18.reuse.ROW, R4;
>                                                                  28
> HMMA.884.F32.F32.STEP3 R6, R20.COL, R18.ROW, R6;
> HMMA.884.F32.F32.STEP0 R8, R14.reuse.COL, R12.reuse.ROW, R8;
>                                                                  30
> HMMA.884.F32.F32.STEP1 R10, R14.reuse.COL, R12.reuse.ROW, R10;
>                                                                  32
> HMMA.884.F32.F32.STEP2 R4, R14.reuse.COL, R12.reuse.ROW, R4;
>                                                                  34
> HMMA.884.F32.F32.STEP3 R6, R14.COL, R12.ROW, R6;
>                                                                  38
> HMMA.884.F32.F32.STEP0 R8, R16.reuse.COL, R2.reuse.ROW, R8;
>                                                                  40
> HMMA.884.F32.F32.STEP1 R10, R16.reuse.COL, R2.reuse.ROW, R10;
>                                                                  42
> HMMA.884.F32.F32.STEP2 R4, R16.reuse.COL, R2.reuse.ROW, R4;
>                                                                  44
> HMMA.884.F32.F32.STEP3 R6, R16.COL, R2.ROW, R6;
>                                                                  54
> ```
> 
> (a) Disassembled SASS instructions for Mixed precision mode
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (b) Elements accessed in each "Step" (mixed-precision mode).
![image.png](GPU%20WMMA%E5%92%8CTensor%20Core/image%205.png)

一个warp中所有线程执行的指令相同,都是HMMA.884.F32.F32.STEPi.

虽然相同set的每个step的**寄存器编号都一样**,但**每个线程每一步使用的操作数根据指令和线程ID动态获取**.**每个编号代表2个32bit寄存器,warp内线程共享一组warp寄存器。**

**每个线程的操作数不同,即读取和写入的目的寄存器各不相同，即计算不同位置的矩阵元素的部分和。**

同一set中两个thread group共同完成1个8*8的MACC计算,这是一种操作数/寄存器的排布方式.

**寄存器堆中对每个操作数矩阵只存储一份数据,矩阵乘法的数据复用通过HMMA指令中寄存器中的reuse将数据保存到operand collentor中,从而不需要重新从寄存器加载数据.**

混合精度下,一个WMMA指令映射到16个HMMA指令(step0-3共4种指令共16组操作数),每个**线程串行执行4个set中的16条指令**.

> **[图片提取文字 (image.jpeg)]:**
> | 0     | 1  | 2  | 3<br>7<br>11<br>15<br>19<br>23 |  |
> |-------|----|----|--------------------------------|--|
> | 4     | 5  | 6  |                                |  |
> | 8     | 9  | 10 |                                |  |
> | 12    | 13 | 14 |                                |  |
> | 16    | 17 | 18 |                                |  |
> | 20    | 21 | 22 |                                |  |
> | 24 25 |    | 26 | 27                             |  |
> | 28    | 29 | 30 | 31                             |  |
> 
> | 0 | 4 | 8  | 12 | 16 | 20 | 24 | 28 |
> |---|---|----|----|----|----|----|----|
> | 1 | 5 | 9  | 13 | 17 | 21 | 25 | 29 |
> | 2 | 6 | 10 | 14 | 18 | 22 | 26 | 30 |
> | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 |
> 
> Fig. 1: Row-major order (left) and column-major order (right) to store one  $8 \times 8$  half precision matrix. Number in the cell is the lane id. One 32-bit thread register stores two half elements. 32 threads within a warp can store 64 elements of the matrix.
![image.jpeg](GPU%20WMMA%E5%92%8CTensor%20Core/image.jpeg)

> **[图片提取文字 (image.jpeg)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 2: Distribution of matrix elements in resigers for HMMA.1688.F16 R0, R2, R6, R4. Each colored  $8 \times 8$  square is an  $8 \times 8$  matrix. We label the register index next to the coresponding matrix. The second source matrix (R6) needed to be stored in column-major order.
![image.jpeg](GPU%20WMMA%E5%92%8CTensor%20Core/image%201.jpeg)

线程组X中第n号线程串行执行set1中4个指令/step的操作数R24分别对应a[0:1,n]，a[2:3,n]，a[0:1,n]，a[2:3,n]，数据按列主序排列；操作数R22分别对应A[n,0:3]，A[n,0:3]，E[n,0:3]，E[n,0:3]，数据按行主序排列。A和a均为4个数的向量。

观察到指令的发射周期和获取数据的长度，每个线程需要从warp寄存器中获取矩阵1的长度为2的列主序数据（如R24、R20、R14、R16）时，发射间隔是2个cycle；需要从warp寄存器中获取矩阵2的长度为4的行主序数据时，发射间隔是4个cycle。