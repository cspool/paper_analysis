# CUDA程序在GPU的执行过程

## 现实中GPU

> **[图片提取文字 (GPU2018_01.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> Background image: Nvidia Tesla P100 with Pascal architecture. Slides by Zhenyu Ye & Gert-Jan van den Braak
![GPU2018_01.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_01.png)

> **[图片提取文字 (GPU2018_05.png)]:**
> ## CPU vs. GPU – chip area
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
![GPU2018_05.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_05.png)

> **[图片提取文字 (GPU2018_07.png)]:**
> ## GPU in Graphics Card
> 
> Image: Nvidia GTX 980
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
![GPU2018_07.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_07.png)

> **[图片提取文字 (GPU2018_09.png)]:**
> GPU in High-Performance Computers
> 
> Image: Nvidia P100 (Pascal Architecture)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
> 
> Chip-on-Wafer-on-Substrate
> 
> ![](_page_0_Picture_5.jpeg)
![GPU2018_09.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_09.png)

> **[图片提取文字 (GPU2018_11.png)]:**
> ## **Transistor Count**
> 
> ref: <a href="http://en.wikipedia.org/wiki/Transistor count">http://en.wikipedia.org/wiki/Transistor count</a>
> 
> ## Multicore Manycore SOC
> 
> | Processor                 | Transistor      | Tech. Node |
> |---------------------------|-----------------|------------|
> | 61-Core Xeon Phi          | 5,000,000,000   | 22nm       |
> | 22-core Xeon Broadwell-E5 | 7,200,000,000   | 14nm       |
> | 32-Core Sparc M7          | 10,000,000,000+ | 20nm       |
> 
> ## **GPU**
> 
> | Processor           | Transistor     | Tech. Node |
> |---------------------|----------------|------------|
> | Nvidia GP100 Pascal | 15,300,000,000 | 16nm       |
> 
> **FPGA** 
> 
> | FPGA                      | Transistor      | Tech. Node |
> |---------------------------|-----------------|------------|
> | Virtex-Ultrascale XCVU440 | 20,000,000,000+ | 20nm       |
> | Stratix 10 10GX5500       | 30,000,000,000+ | 14nm 11    |
![GPU2018_11.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_11.png)

> **[图片提取文字 (GPU2018_08.png)]:**
> ## GPU in Mobile Processors
> 
> Image: Nvidia Jetson TX1 (Tegra X1 SOC)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
![GPU2018_08.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_08.png)

> **[图片提取文字 (GPU2018_10.png)]:**
> ## NVIDIA (Fermi Arch.)
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Fermi Streaming Multiprocessor (SM)
> 
> ref: http://www.nvidia.com/content/PDF/fermi white papers/NVIDIA Fermi Compute Architecture Whitepaper.pdf
![GPU2018_10.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_10.png)

## 图形pipeline

> **[图片提取文字 (GPU2018_12.png)]:**
> ## What Can 15bn Transistors Do?
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Render triangles.
> Billions of triangles per second.
> 
> ![](_page_0_Picture_3.jpeg)
![GPU2018_12.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_12.png)

> **[图片提取文字 (GPU2018_14.png)]:**
> ## The Graphics Pipeline
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Polygon mesh in world space
> 
> ![](_page_0_Figure_3.jpeg)
> 
> Polygons mesh in eye space. Vertices are attached with lighting, color, etc
> 
> ![](_page_0_Picture_5.jpeg)
![GPU2018_14.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_14.png)

> **[图片提取文字 (GPU2018_16.png)]:**
> ## The Graphics Pipeline
> 
> Polygon mesh Vertex Processing Rasterization Fragment Processing Raster Operation Pixels
> 
> Pixel fragments without color, lighting, etc.
> 
> ![](_page_0_Picture_3.jpeg)
> 
> Pixel fragments with color, lighting, etc.
> 
> ![](_page_0_Picture_5.jpeg)
> 
> Interpolate depth, color, etc. for pixels in the fragment. Texture filtering and mapping.
> 
> ![](_page_0_Picture_7.jpeg)
> 
> ![](_page_0_Picture_8.jpeg)
![GPU2018_16.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_16.png)

> **[图片提取文字 (GPU2018_18.png)]:**
> ## Graphics Pipeline on GPU
> 
> ![](_page_0_Figure_1.jpeg)
![GPU2018_18.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_18.png)

> **[图片提取文字 (GPU2018_20.png)]:**
> ## Graphics Pipeline on GPU
> 
> ![](_page_0_Figure_1.jpeg)
![GPU2018_20.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_20.png)

> **[图片提取文字 (GPU2018_22.png)]:**
> ## How Do GPUs Spend Their Die Area?
> 
> GPUs are designed to match the workload of 3D graphics.
> 
> Die photo of GeForce GTX 280 (source: NVIDIA)
> 
> ## Texture:
> 
> For fragment processing.
> 
> ## **ROP & Frame Buffer:**
> 
> For raster operation.
> 
> ![](_page_0_Figure_7.jpeg)
> 
> J. Roca, et al. "Workload Characterization of 3D Games", IISWC 2006, link
> 
> T. Mitra, et al. "Dynamic 3D Graphics Workload Characterization and the Architectural Implications", Micro 1999, link
![GPU2018_22.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_22.png)

> **[图片提取文字 (GPU2018_13.png)]:**
> ## The Graphics Pipeline
> 
> Polygon mesh **Vertex Processing** Rasterization Fragment Processing Raster Operation Pixels
![GPU2018_13.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_13.png)

> **[图片提取文字 (GPU2018_15.png)]:**
> ## The Graphics Pipeline
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Triangle primative Rasterization Pixel fragments covered
> 
> by the triangle.
> 
> ![](_page_0_Picture_4.jpeg)
> 
> From geometry primatives to pixel fragments.
> 
> ![](_page_0_Figure_6.jpeg)
![GPU2018_15.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_15.png)

> **[图片提取文字 (GPU2018_17.png)]:**
> ## The Graphics Pipeline
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> Visible pixels from
![GPU2018_17.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_17.png)

> **[图片提取文字 (GPU2018_19.png)]:**
> ## Graphics Pipeline on GPU
> 
> ![](_page_0_Figure_1.jpeg)
![GPU2018_19.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_19.png)

> **[图片提取文字 (GPU2018_21.png)]:**
> ## Graphics Pipeline on GPU
> 
> ![](_page_0_Figure_1.jpeg)
![GPU2018_21.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_21.png)

> **[图片提取文字 (GPU2018_23.png)]:**
> ## GPUs Besides Graphics
> 
> TOP500 supercomputer list in Nov. 2018. (<a href="http://www.top500.org">http://www.top500.org</a>)
> 
> | Rank | System                                                                                                                                                                 | Cores      | Rmax<br>(TFlop/s) | Rpeak<br>(TFlop/s) | Power (kW) |
> |------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------|--------------------|------------|
> | 1    | Summit - IBM Power System AC922, IBM POWER9 22C 3.07GHz, NVIDIA Volta GV100 Dual-rail Mellanox EDR Infiniband , IBM DOE/SC/Oak Ridge National Laboratory United States | 2,397,824  | 143,500.0         | 200,794.9          | 9,783      |
> | 2    | Sierra - IBM Power System S922LC, IBM POWER9 22C 3.1GHz,  NVIDIA Volta GV100 Dual-rail Mellanox EDR Infiniband, IBM /  NVIDIA / Mellanox  DOE/NNSA/LLNL  United States | 1,572,480  | 94,640.0          | 125,712.0          | 7,438      |
> | 3    | Sunway TaihuLight - Sunway MPP, Sunway SW26010 260C<br>1.45GHz, Sunway , NRCPC<br>National Supercomputing Center in Wuxi<br>China                                      | 10,649,600 | 93,014.6          | 125,435.9          | 15,371     |
![GPU2018_23.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_23.png)

## CPU的标量指令和向量指令 vs. GPU CUDA指令

SM的SIMT执行提供分支发散-收束管理机制，Shared-Mem提供线程间协作，因此CUDA是多线程编程模型，线程编程的语义范围比向量指令更广。

> **[图片提取文字 (GPU2018_24.png)]:**
> ## Let's Start with Examples
> 
> ![](_page_0_Figure_1.jpeg)
> 
> We will start from C and RISC.
![GPU2018_24.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_24.png)

> **[图片提取文字 (GPU2018_26.png)]:**
> ## Most CPUs Have Vector SIMD Units
> 
> Programmer's view of a vector SIMD, e.g. SSE.
> 
> ![](_page_0_Figure_2.jpeg)
![GPU2018_26.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_26.png)

> **[图片提取文字 (GPU2018_28.png)]:**
> ## How Do Vector Programs Run?
> 
> ```
> int A[2][4];
> for(i=0;i<2;i++){
>     movups xmm0,
> ```
> 
> ![](_page_0_Picture_2.jpeg)
![GPU2018_28.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_28.png)

> **[图片提取文字 (GPU2018_30.png)]:**
> ## CUDA Programmer's View of GPUs
> 
> A GPU contains multiple SIMD Units. All of them can access global memory.
> 
> ![](_page_0_Figure_2.jpeg)
![GPU2018_30.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_30.png)

> **[图片提取文字 (GPU2018_25.png)]:**
> ## Let's Start with C and RISC
> 
> ```
> int A[2][4];
> for(i=0;i<2;i++){
>     for(j=0;j<4;j++){
>         A[i][j]++;
>     }
> }</pre>
> ```
> 
> Assembly code of inner-loop
> 
> ![](_page_0_Picture_3.jpeg)
> 
> lw r0, 4(r1) addi r0, r0, 1 sw r0, 4(r1)
> 
> Programmer's view of RISC
![GPU2018_25.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_25.png)

> **[图片提取文字 (GPU2018_27.png)]:**
> ## Let's Program the Vector SIMD
> 
> Unroll inner-loop to vector operation.
> 
> ```
> int A[2][4];
> int A[2][4];
> for(i=0;i<2;i++){
> for(j=0;j<4;j++){
>                       for(i=0;i<2;i++){
>                         addps xmm0, xmm1
>                                                          // add 1
>                         movups [ &A[i][0] ], xmm0
>                                                          // store
> ```
> 
> Looks like the previous example, but each SSE instruction executes on 4 ALUs.
![GPU2018_27.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_27.png)

> **[图片提取文字 (GPU2018_29.png)]:**
> ## CUDA Programmer's View of GPUs
> 
> A GPU contains multiple SIMD Units.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
![GPU2018_29.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_29.png)

> **[图片提取文字 (GPU2018_31.png)]:**
> ## What Are the Differences?
> 
> SSE
> 
> Reg Reg Reg Vector LD/ST
> 
> Vector LD/ST
> 
> Global
> Memory
> 
> 4
> 
> GPU
> 
> ![](_page_0_Figure_4.jpeg)
> 
> Let's start with two important differences:
> 
> - 1. GPUs use threads instead of vectors
> - 2. GPUs have the "Shared Memory" spaces
![GPU2018_31.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_31.png)

## GPU执行模型

定义grid为gridSz和blkSz，线程block到SM（SM接受负载的单位），block在SM内按warp执行（SIMD Cores执行指令的单位），SM内不同线程Ctx独立。

> **[图片提取文字 (GPU2018_32.png)]:**
> ## Thread Hierarchy in CUDA
> 
> Grid
> contains
> Thread Blocks
> 
> Thread Block contains
> Threads
> 
> ![](_page_0_Figure_3.jpeg)
![GPU2018_32.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_32.png)

> **[图片提取文字 (GPU2018_34.png)]:**
> ## Thread Hierarchy
> 
> ```
> Example:
> thread 3 of block 1 operates
> on element A[1][3]
> int A[2][4];
> kernelF << < (2,1), (4,1) >>> (A); // define 2x4=8 threads
>                kernelF(A){
>   device___
>   i = blockldx.x;
>   j = threadldx.x;
>   A[i][j]++;
> ```
> 
> ```
> kernelF contains 2 x 1)thread blocks
>    block 0,0 block 0,1
>                     Grid
>        Thread Block
>                          Thread
>               thread
>    thread
>   Each thead block contains 4 x 1 threads
> // all threads run same kernel
>  // each thread block has its id
>  // each thread has its id
>  // each thread has different i and j
> ```
![GPU2018_34.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_34.png)

> **[图片提取文字 (GPU2018_36.png)]:**
> ## Blocks Are Dynamically Scheduled
> 
> ![](_page_0_Figure_1.jpeg)
![GPU2018_36.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_36.png)

> **[图片提取文字 (GPU2018_38.png)]:**
> ## Utilizing Memory Hierarchy
> 
> Memory access latency
> 
> several cycles
> 
> 100+ cycles
> 
> ![](_page_0_Figure_4.jpeg)
![GPU2018_38.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_38.png)

> **[图片提取文字 (GPU2018_40.png)]:**
> ## Utilizing the Shared Memory
> 
> Average over a 3x3 window for a 16x16 array
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ```
> kernelF<<<(1,1),(16,16)>>>(A);
> _device kernelF(A){
>      shared int smem[16][16];
>   i = threadIdx.y;
>  j = threadldx.x;
>   smem[i][j] = A[i][j]; // load to smem[i][i][j] = A[i][i][i][i][i][i][i][i][i][i][i][i][i][
>   A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
>              ... + smem[i+1][i+1]) / 9;
> ```
![GPU2018_40.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_40.png)

> **[图片提取文字 (GPU2018_33.png)]:**
> ## Let's Start Again from C
> 
> ```
> int A[2][4];
>                     for(i=0;i<2;i++)
>                        for(j=0;j<4;j++)
>   convert into CUDA
>                          A[i][i]++;
> int A[2][4];
> kernelF <<<(2,1),(4,1)>>>(A); // define 2x4=8 threads
>   device kernelF(A){
>                                 // all threads run same kernel
>   i = blockldx.x;
>                                 // each thread block has its id
>   j = threadldx.x;
>                                 // each thread has its id
>   A[i][j]++;
>                                 // each thread has different i and j
> ```
![GPU2018_33.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_33.png)

> **[图片提取文字 (GPU2018_35.png)]:**
> ## How Are Threads Scheduled?
> 
> kernelF contains 2 x 1 thread blocks
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Each thead block contains 4 x 1 threads
> 
> ![](_page_0_Figure_4.jpeg)
![GPU2018_35.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_35.png)

> **[图片提取文字 (GPU2018_37.png)]:**
> ## **How Are Threads Executed?**
> 
> ```
> int A[2][4];
> kernelF<<<(2,1),(4,1)>>>(A);
> ```
> 
> ```
> mv.u32 %r0, %ctaid.x
> mv.u32 %r1, %ntid.x
> mv.u32 %r2, %tid.x
> mad.u32 %r3, %r2, %r1, %r0
> ld.global.s32 %r4, [%r3]
> add.s32 %r4, %r4, 1
> st.global.s32 [%r3], %r4
> ```
> 
> ```
> // r0 = i = blockldx.x
> // r1 = "threads-per-block"
> // r2 = j = threadldx.x
> // r3 = i * "threads-per-block" + j
> // r4 = A[i][j]
> // r4 = r4 + 1
> // A[i][j] = r4
> 37
> ```
![GPU2018_37.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_37.png)

> **[图片提取文字 (GPU2018_39.png)]:**
> ## Example: Average Filters
> 
> Average over a 3x3 window for a 16x16 array
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ```
> kernelF <<<(1,1),(16,16)>>>(A);
> device kernelF(A){
> i = threadIdx.y;
> j = threadIdx.x;
> tmp = (A[i-1][j-1] + A[i-1][j] +
>         ... + A[i+1][i+1]) / 9;
>  A[i][j] = tmp;
>                Each thread loads 9 elements
>                from global memory.
> ```
> 
> It takes hundreds of cycles.
![GPU2018_39.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_39.png)

> **[图片提取文字 (GPU2018_41.png)]:**
> ## Utilizing the Shared Memory
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ```
> kernelF<<<(1,1),(16,16)>>>(A);
>            kernelF(A){
> device
>                                allocate
>                                shared
>   shared smem[16][16];
> i = threadIdx.y;
>                  Each thread loads one
> j = threadIdx.x;
>                  element from global memory.
> A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
>           ... + smem[i+1][i+1]) / 9;
> ```
![GPU2018_41.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_41.png)

> **[图片提取文字 (GPU2018_42.png)]:**
> ## However, the Program Is Incorrect
> 
> ```
> kernelF<<<(1,1),(16,16)>>>(A);
> device kernelF(A){
>   shared_smem[16][16];
>   i = threadIdx.y;
>   j = threadldx.x;
>   smem[i][j] = A[i][j]; // load to smem
>   A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
>            ... + smem[i+1][i+1]) / 9;
> ```
> 
> Hazards!
![GPU2018_42.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_42.png)

> **[图片提取文字 (GPU2018_44.png)]:**
> ## Let's See What's Wrong
> 
> scheduled on 8 PEs.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Some threads finish the load earlier than others.
> 
> ```
> Assume 256 threads are kernelF <<<(1,1),(16,16)>>>(A);
>           device kernelF(A){
>            shared__smem[16][16];
>           i = threadIdx.y;
>           j = threadIdx.x;
>           smem[i][j] = A[i][j]; // load to smem
>           A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
>                     ... + smem[i+1][i+1]) / 9;
> ```
![GPU2018_44.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_44.png)

> **[图片提取文字 (GPU2018_46.png)]:**
> ## **How To Solve It?**
> 
> scheduled on 8 PEs.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ```
> Assume 256 threads are kernelF <<<(1,1),(16,16)>>>(A);
>           device kernelF(A){
>            shared smem[16][16];
>           i = threadIdx.y;
>           j = threadldx.x;
>           smem[i][j] = A[i][j]; // load to smem
>           A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
>                     ... + smem[i+1][i+1]) / 9;
> ```
![GPU2018_46.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_46.png)

> **[图片提取文字 (GPU2018_48.png)]:**
> ## Use a "SYNC" barrier kernelF<<<(1,1),(16,16)>>>(A);
> 
> Assume 256 threads are scheduled on 8 PEs.
> 
> ```
> PES
> Shared Mem
> 
> DRAM
> ```
> 
> ```
> device
>            kernelF(A){
>   shared smem[16][16];
>                                 Wait until all
> i = threadIdx.y;
>                                   threads
> j = threadIdx.x;
>                                   hit barrier
> smem[i][j] = A[i][j]; // load to smem
>   SYNC();
> A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
>           ... + smem[i+1][i+1]) / 9;
> ```
![GPU2018_48.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_48.png)

> **[图片提取文字 (GPU2018_43.png)]:**
> ## Let's See What's Wrong
> 
> scheduled on 8 PEs.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ```
> Assume 256 threads are kernelF <<<(1,1),(16,16)>>>(A);
>            device kernelF(A){
>             _shared__smem[16][16];
>            i = threadIdx.y;
>           j = threadldx.x;
>                              Before load instruction
>            smem[i][j] = A[i][j]; // load to smem
>            A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
>                      ... + smem[i+1][i+1]) / 9;
> ```
![GPU2018_43.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_43.png)

> **[图片提取文字 (GPU2018_45.png)]:**
> ## Let's See What's Wrong
> 
> scheduled on 8 PEs.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Some elements in the window are not yet loaded by other threads. Error!
> 
> ```
> Assume 256 threads are kernelF <<<(1,1),(16,16)>>>(A);
>           device kernelF(A){
>            _shared__smem[16][16];
>           i = threadIdx.y;
>           j = threadldx.x;
>           smem[i][j] = A[i][j]; // load to smem
>           A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
>                     ... + smem[i+1][i+1]) / 9;
> ```
![GPU2018_45.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_45.png)

> **[图片提取文字 (GPU2018_47.png)]:**
> ## Use a "SYNC" barrier kernelF<<<(1,1),(16,16)>>>(A);
> 
> Assume 256 threads are scheduled on 8 PEs.
> 
> ```
> PEs
> Shared Mem
> 
> DRAM
> ```
> 
> ```
> device
>           kernelF(A){
>   shared smem[16][16];
> i = threadIdx.y;
> j = threadldx.x;
> smem[i][j] = A[i][j]; // load to smem
>   SYNC();
> A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
> ```
> 
> ... + smem[i+1][i+1]) / 9;
![GPU2018_47.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_47.png)

> **[图片提取文字 (GPU2018_49.png)]:**
> ## Use a "SYNC" barrier kernelF<<<(1,1),(16,16)>>>(A);
> 
> Assume 256 threads are scheduled on 8 PEs.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> All elements in the window are loaded when each thread starts averaging.
> 
> ```
> device
>           kernelF(A){
>   shared smem[16][16];
> i = threadIdx.y;
> j = threadldx.x;
> smem[i][j] = A[i][j]; // load to smem
>   SYNC();
> A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
>          ... + smem[i+1][i+1]) / 9;
> ```
![GPU2018_49.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_49.png)

## vector指令编程模型 vs. SIMT CUDA

vector是数据级并行DLP，HW SIMD宽度对编程者可见（vector宽度等于SIMD宽度），vector指令的数据同步执行。

CUDA是线程级并行TLP，HW SIMD宽度对编程者透明（HW通过pipeline并行提高吞吐），不同warp的线程异步，warp（线程组）内的线程同步执行。

> **[图片提取文字 (GPU2018_50.png)]:**
> ## Review What We Have Learned
> 
> - 1. Single Instruction Multiple Thread (SIMT)
> - 2. Shared memory
> 
> ![](_page_0_Picture_3.jpeg)
> 
> Q: What are the pros and cons of explicitly managed memory?
> 
> Q: What are the fundamental differences between SIMT and vector SIMD programming model?
![GPU2018_50.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_50.png)

> **[图片提取文字 (GPU2018_52.png)]:**
> ## Vector SIMD v.s. SIMT
> 
> ```
> int A[16][16]; // A in global memory
> __shared__ int B[16][16]; // B in shared mem
> for(i=0;i<16;i++){
>   for(j=0;j<16;j+=4){
>      movups xmm0, [ &A[i][j] ]
>      movups [ &B[i][j] ], xmm0 }}
> for(i=0;i<16;i++)
>   for(j=0;j<16;j+=4){
>      addps xmm1, [ &B[i-1][j-1] ]
>      addps xmm1, [ &B[i-1][j] ]
>      ... divps xmm1, 9 }}
> for(i=0;i<16;i++){
>   for(j=0;j<16;j+=4){
>      addps [ &A[i][j] ], xmm1 }}
> ```
> 
> ```
> kernelF <<<(1,1),(16,16)>>>(A);
>  device__
>              kernelF(A){
>   __shared__ int smem[16][16];
>   i = threadIdx.y;
>   j = threadldx.x;
>   smem[i][j] = A[i][j]; // load to smem (1)
>     sync(); // threads wait at barrier
>   A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
>            ... + smem[i+1][i+1]) / 9;
> ```
> 
> - (1) load to shared mem
> - (2) compute
> - (3) store to global mem
![GPU2018_52.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_52.png)

> **[图片提取文字 (GPU2018_54.png)]:**
> ## Review What We Have Learned
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Programmers convert data level parallelism (DLP) into thread level parallelism (TLP).
![GPU2018_54.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_54.png)

> **[图片提取文字 (GPU2018_51.png)]:**
> ## Take the Same Example Again
> 
> Average over a 3x3 window for a 16x16 array
> 
> ![](_page_0_Picture_2.jpeg)
> 
> Assume vector SIMD and SIMT both have shared memory.
> 
> What are the differences?
> 
> ![](_page_0_Picture_5.jpeg)
![GPU2018_51.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_51.png)

> **[图片提取文字 (GPU2018_53.png)]:**
> ## Vector SIMD v.s. SIMT
> 
> ```
> int A[16][16]; // A in global memory
> __shared__ int B[16][16]; // B in shared mem
> for(i=0;i<16;i++){
>   for(j=0; j<16; j+=4){ \leftarrow (a)
>    movups xmm0, [ &A[i][j] ]
>     movups [ &B[i][j] ], xmm0 }}
> for(i=0;i<16;i++){
>   for(j=0;j<16;j+=4){
>      addps xmm1, [ &B[i-1][j-1] ]
>      addps xmm1, [ &B[i-1][j] ]
>      ... divps xmm1, 9 }}
> for(i=0;i<16;i++){
>   for(j=0;j<16;j+=4){
> ```
> 
> addps [ &A[i][j] ], xmm1 }}
> 
> ```
> kernelF <<<(1,1),(16,16)>>>(A);
>  _device__ kernelF(A){
>   __shared__ smem[16][16];
>  i = threadIdx.y;
> j = threadIdx.x;
> (b)
>   smem[i][j] = A[i][j]; // load to smem
>     _sync(); // threads wait at barrier (d)
>   A[i][j] = (smem[i-1][j-1] + smem[i-1][j] +
>             ... + smem[i+1][i+1] ) / 9;
> ```
> 
> - (a) HW vector width **explicit** to programmer
> - (b) HW vector width transparent to programmers
> - (c) each vector executed by all PEs in lock step
> - (d) threads executed <u>out of order</u>, need explicit syng
![GPU2018_53.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_53.png)

> **[图片提取文字 (GPU2018_55.png)]:**
> ## **HW Groups Threads Into Warps**
> 
> Group Threads into Warps
> 
> Processing Elements (PEs)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Example: 32 threads per wasp
![GPU2018_55.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_55.png)

## SM中warp的执行

**scheduler**调度不同warp准备发射，可能包含不同线程块的warp，一般交错调度不同warp的指令来避免指令间interlock。

通过**计分板**SB来管理warp内或warp间可能存在的后序指令依赖前序指令，不依赖前序指令的后序指令不必等待前序指令完成即可发射。

warp中线程的**PC**相同/共享，基于Barrier的分支发散和执行中每个线程可能保有自己的PC，同分支线程组并发，而不必等待warp同步。

warp的线程从**寄存器Banks**中**收集操作数**（Operator Collector），可能存在Bank Conflict。操作数收集完成的指令通过发射器**issuer发射**到SIMD Cores执行。

**SIMD Cores**设计pipeline并行，执行warp的SIMD指令需要多周期，不同指令延迟不同。不同SIMD Cores并行执行不同指令。

> **[图片提取文字 (GPU2018_56.png)]:**
> ## **Execution of Threads and Warps**
> 
> Let's start with an example.
![GPU2018_56.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_56.png)

> **[图片提取文字 (GPU2018_58.png)]:**
> ## **Example of Register Allocation**
> 
> Assumption: register file has 32 lanes each warp has 32 threads each thread uses 8 registers
> 
> Warp 0
> 
> Acronym:
> 
> "T": thread number
> 
> "R": register number
> 
> Warp 1
> 
> Note: NVIDIA may use a more complicated allocation method. See patent: US 7634621 B1, "Register file allocation".
> 
> Lane 0 Lane 31 Lane 1 T0,R0 T31,R0 T1,R0 T0,R1 T32,R0 T33,R0 T63,R0 T33,R1 T32,R1 T32,R7 T33,R7 T63,R7
![GPU2018_58.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_58.png)

> **[图片提取文字 (GPU2018_60.png)]:**
> ## Read Src Op 1
> 
> Address: Instruction
> 
> 0x0004 : add r0, r1, r2
> 
> 0x0008 : sub r3, r4, r5
> 
> Read source operands:
> 
> r1 for warp 0
> 
> r4 for warp 1
> 
> ![](_page_0_Figure_7.jpeg)
![GPU2018_60.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_60.png)

> **[图片提取文字 (GPU2018_62.png)]:**
> ## Read Src Op 2
> 
> Address: Instruction
> 
> 0x0004 : add r0, r1, r2
> 
> 0x0008 : sub r3, r4, r5
> 
> Read source operands:
> 
> r2 for warp 0
> 
> r5 for warp 1
> 
> ![](_page_0_Figure_7.jpeg)
![GPU2018_62.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_62.png)

> **[图片提取文字 (GPU2018_64.png)]:**
> ## **Execute Stage 1**
> 
> Address: Instruction
> 
> 0x0004 : add r0, r1, r2
> 
> 0x0008 : sub r3, r4, r5
> 
> Compute the first 16 threads in the warp.
> 
> ![](_page_0_Figure_5.jpeg)
![GPU2018_64.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_64.png)

> **[图片提取文字 (GPU2018_66.png)]:**
> ## Write Back
> 
> Address: Instruction
> 
> 0x0004 : add r0, r1, r2
> 
> 0x0008 : sub r3, r4, r5
> 
> Write back: r0 for warp 0 r3 for warp 1
> 
> ![](_page_0_Figure_5.jpeg)
![GPU2018_66.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_66.png)

> **[图片提取文字 (GPU2018_57.png)]:**
> **Example of Implementation** 
> 
> Note: NVIDIA may use a more complicated implementation. See patent: US 8555035 B1
> 
> SM
> 
> Xbar
> 
> Shared Mem / L
> 
> tex tex tex tex
> 
> PolyMorph Engine
> 
> texture cache
> 
> ![](_page_0_Figure_2.jpeg)
![GPU2018_57.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_57.png)

> **[图片提取文字 (GPU2018_59.png)]:**
> ## Example
> 
> Address: Instruction
> 
> 0x0004 : add r0, r1, r2
> 
> 0x0008 : sub r3, r4, r5
> 
> Assume: two data paths warp 0 on left data path warp 1 on right data path
> 
> ## Acronyms:
> 
> "AGU": address generation unit
> 
> "r": register in a thread
> 
> "w": warp number
> 
> ![](_page_0_Figure_9.jpeg)
![GPU2018_59.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_59.png)

> **[图片提取文字 (GPU2018_61.png)]:**
> ## Buffer Src Op 1
> 
> Address: Instruction
> 
> 0x0004 : add r0, r1, r2
> 
> 0x0008 : sub r3, r4, r5
> 
> Push ops to op collector:
> 
> r1 for warp 0
> 
> r4 for warp 1
> 
> ![](_page_0_Figure_7.jpeg)
![GPU2018_61.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_61.png)

> **[图片提取文字 (GPU2018_63.png)]:**
> ## Buffer Src Op 2
> 
> Address: Instruction
> 
> 0x0004 : add r0, r1, r2
> 
> 0x0008 : sub r3, r4, r5
> 
> Push ops to op collector: r2 for warp 0 r5 for warp 1
> 
> ![](_page_0_Figure_5.jpeg)
![GPU2018_63.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_63.png)

> **[图片提取文字 (GPU2018_65.png)]:**
> ## **Execute Stage 2**
> 
> Address: Instruction
> 
> 0x0004 : add r0, r1, r2
> 
> 0x0008 : sub r3, r4, r5
> 
> Compute the last 16 threads in the warp.
> 
> ![](_page_0_Figure_5.jpeg)
![GPU2018_65.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_65.png)

> **[图片提取文字 (GPU2018_67.png)]:**
> ## Recap
> 
> What we have learned so far:
> 
> - Pros and cons of massive threading
> - How threads are executed on GPUs
> 
> Next: variations of GPU multiprocessor cluster.
![GPU2018_67.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_67.png)

## Fermi、Kepler、Maxwell、Pascal、Volta、Turing

> **[图片提取文字 (GPU2018_68.png)]:**
> ## NVIDIA Fermi (2009
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Fermi Streaming Multiprocessor (SM)
> 
> ref: http://www.nvidia.com/content/PDF/fermi white papers/NVIDIA Fermi Compute Architecture Whitepaper.pdf
![GPU2018_68.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_68.png)

> **[图片提取文字 (GPU2018_70.png)]:**
> ## NVIDIA Maxwell (2014)
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
![GPU2018_70.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_70.png)

> **[图片提取文字 (GPU2018_72.png)]:**
> ## NVIDIA Pascal (GP100, 2016)
> 
> ![](_page_0_Figure_1.jpeg)
> 
> -archit
> 
> df/tesla/whitepaper/pa
![GPU2018_72.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_72.png)

> **[图片提取文字 (GPU2018_74.png)]:**
> NVIDIA Turing (2018)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> NVLink - Two x8 Links
> 
> ![](_page_0_Figure_3.jpeg)
![GPU2018_74.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_74.png)

> **[图片提取文字 (GPU2018_69.png)]:**
> NVIDIA Kepler (2012)
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
![GPU2018_69.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_69.png)

> **[图片提取文字 (GPU2018_71.png)]:**
> ![](_page_0_Figure_1.jpeg)
> 
> 11
> 
> Raster Engine
> 
> 14
> 
> 11
> 
> 14
> 
> Raster Engine
> 
> 11
> 
> 14
> 
> GPC
> 
> |                          |                                 |                                                     |                          |                                                   | Instructi                       | on Cache            |                                 |                                                     |                          |                                                   |                                 |
> |--------------------------|---------------------------------|-----------------------------------------------------|--------------------------|---------------------------------------------------|---------------------------------|---------------------|---------------------------------|-----------------------------------------------------|--------------------------|---------------------------------------------------|---------------------------------|
> |                          | - 1                             | nstructio                                           | on Buffe                 | II.                                               |                                 |                     | - 30                            | nstructio                                           | on Buffe                 | r                                                 |                                 |
> |                          |                                 | Warp \$c                                            | heduler                  |                                                   | Warp Scheduler                  |                     |                                 |                                                     |                          |                                                   |                                 |
> | Di                       | ispatch Uni                     | it                                                  | Dispetch Unit            |                                                   |                                 | Dispatch Unit       |                                 |                                                     | Dispetch Unit            |                                                   |                                 |
> |                          |                                 | er File (*                                          | 16,384 x                 |                                                   |                                 |                     |                                 | er File (1                                          | 16,384 x                 |                                                   |                                 |
> | Core                     | Core                            | Core                                                | Core                     | LD/ST                                             | SFU                             | Core                | Core                            | Core                                                | Core                     | LD/ST                                             | SFL                             |
> | Core                     | Core                            | Core                                                | Core                     | LD/ST                                             | SFU                             | Core                | Core                            | Core                                                | Core                     | LD/ST                                             | SFL                             |
> | Core                     | Core                            | Core                                                | Core                     | LD/ST                                             | SFU                             | Core                | Core                            | Core                                                | Core                     | LD/ST                                             | SFL                             |
> | Core                     | Core                            | Core                                                | Core                     | LD/ST                                             | SFU                             | Core                | Core                            | Core                                                | Core                     | LD/ST                                             | SFL                             |
> | Core                     | Core                            | Core                                                | Core                     | LD/ST                                             | SFU                             | Core                | Core                            | Core                                                | Core                     | LD/ST                                             | SFL                             |
> | Core                     | Core                            | Core                                                | Core                     | LD/ST                                             | SFU                             | Core                | Core                            | Core                                                | Core                     | LD/ST                                             | SFL                             |
> | Core                     | Core                            | Core                                                | Core                     | LD/ST                                             | SFU                             | Core                | Core                            | Core                                                | Core                     | LDIST                                             | SFL                             |
> | Core                     | Core                            | Core                                                | Core                     | LD/ST                                             | SFU                             | Core                | Core                            | Core                                                | Core                     | LD/ST                                             | SFL                             |
> |                          |                                 |                                                     |                          |                                                   | Texture /                       | L1 Cache            |                                 |                                                     |                          |                                                   |                                 |
> | Tex Tex                  |                                 |                                                     |                          |                                                   |                                 |                     | Tex                             |                                                     |                          | Tex                                               |                                 |
> |                          |                                 |                                                     |                          |                                                   |                                 |                     |                                 |                                                     |                          |                                                   |                                 |
> |                          |                                 | nstructi                                            | on Buffe                 | er -                                              |                                 |                     |                                 | nstructi                                            | on Buffe                 | ) (                                               |                                 |
> |                          | e company                       | Warp Se                                             | heduler                  | 15 N-77 75                                        | 17.                             |                     |                                 | Warp S                                              | heduler                  | er verringe                                       |                                 |
> | D                        | ispatch Un                      | Warp Se                                             | heduler                  | Dispatch U                                        | nit                             | D                   | ispatch Un                      | Warp S                                              | heduler                  | Dispatch U                                        | nit                             |
> | D                        | ispatch Un                      | Warp Se                                             | heduler                  | Dispatch U                                        | nit                             | D                   | ispatch Un                      | Warp S                                              | cheduler                 | Dispatch U                                        | nit                             |
> | Core                     | ispatch Un Regist               | Warp Se                                             | theduler<br>16,384 x     | Dispatch U                                        | SFU                             | Core                | ispatch Un                      | Warp Si<br>it<br>er File (                          | cheduler                 | Dispatch Un                                       |                                 |
> |                          | ispatch Un Regist               | Warp Se<br>it<br>er File (                          | theduler<br>16,384 x     | Dispatch U  32-bit)                               |                                 |                     | ispatch Un<br>•<br>Regist       | Warp Si<br>it<br>er File (                          | theduler<br>16,384 x     | Dispatch Un                                       | SFI                             |
> | Core                     | Regist                          | Warp Seit<br>er File (<br>Core                      | 16,384 x                 | 32-bit)                                           | SFU                             | Core                | Regist                          | Warp Si<br>it<br>er File (<br>Core                  | 16,384 x                 | 32-bit)                                           | SFI                             |
> | Core                     | Regist Core                     | Warp Set                                            | 16,384 x<br>Core         | 32-bit)  LD/ST                                    | SFU<br>SFU                      | Core                | Regist Core                     | Warp Sit \ner File (  Core                          | 16,384 x<br>Core         | 32-bit)  LD/ST                                    | SFI<br>SFI                      |
> | Core<br>Core             | Regist Cere Core                | Warp Stit \ner File (  Core  Core                   | Core                     | 32-bit) LD/ST LD/ST LD/ST                         | SFU<br>SFU<br>SFU               | Core<br>Core        | Regist Core Core                | Warp Stite of File (**  Core  Core  Core            | Core                     | 32-bit)  LD/ST  LD/ST                             | SFI<br>SFI<br>SFI               |
> | Core Core Core           | Regist Core Core Core           | Warp Sit\ner File (  Core  Core  Core               | Core Core Core           | 32-bit)  LD/ST  LD/ST  LD/ST  LD/ST               | SFU<br>SFU<br>SFU               | Core Core Core      | Regist Core Core Core           | Warp Stite (** Core Core Core                       | Core Core Core           | 32-bit)  LD/ST  LD/ST  LD/ST                      | SFI<br>SFI<br>SFI<br>SFI        |
> | Core Core Core           | Regist Core Core Core           | Warp Sit\ner File ( Core Core Core                  | Core Core Core           | 32-bit)  LD/ST  LD/ST  LD/ST  LD/ST  LD/ST        | SFU<br>SFU<br>SFU<br>SFU        | Core Core Core      | Regist Core Core Core           | Warp Stite of File (** Core Core Core Core          | Core Core Core           | 32-bit)  LD/ST  LD/ST  LD/ST  LD/ST               | SFI SFI SFI SFI                 |
> | Core Core Core Core      | Regist Core Core Core Core      | Warp Stit \ner File (  Core  Core  Core  Core       | Core Core Core Core      | 32-bit)  LD/ST  LD/ST  LD/ST  LD/ST  LD/ST  LD/ST | SFU<br>SFU<br>SFU<br>SFU<br>SFU | Core Core Core Core | Regist Core Core Core Core      | Warp Stite  Core  Core  Core  Core  Core  Core      | Core Core Core Core      | 32-bit)  LD/ST  LD/ST  LD/ST  LD/ST  LD/ST        | SFU<br>SFU<br>SFU<br>SFU<br>SFU |
> | Core Core Core Core Core | Regist Core Core Core Core Core | Warp Stit \ner File (  Core  Core  Core  Core  Core | Core Core Core Core Core | 32-bit)  LD/ST  LD/ST  LD/ST  LD/ST  LD/ST  LD/ST | SFU SFU SFU SFU SFU SFU         | Core Core Core Core | Regist Core Core Core Core Core | Warp Stite  Er File (  Core  Core  Core  Core  Core | Core Core Core Core Core | 32-bit)  LD/ST  LD/ST  LD/ST  LD/ST  LD/ST  LD/ST | SFL SFL SFL SFL                 |
![GPU2018_71.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_71.png)

> **[图片提取文字 (GPU2018_73.png)]:**
> **NVLink** 
> 
> **NVLink** 
> 
> **NVLink** 
> 
> **NVLink** 
> 
> **NVLink** 
> 
> **NVLink**
![GPU2018_73.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_73.png)

> **[图片提取文字 (GPU2018_75.png)]:**
> ## Variations of Multiprocessor Cluster
> 
> Fermi (2009) I Cache I Cache Crossbar Network Reg File Reg File
> 
> Kepler (2012) Warp Crossbar Network
> 
> Maxwell (2014) I Cache
> 
> From Fermi to Kepler: More cores per cluster.
> 
> From Kepler to Maxwell: Partition cluster into sub-clusters.
> 
> Question: What are the Pros and Cons of these variations?
![GPU2018_75.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_75.png)

## SM中的冒险

结构冒险：Shared Mem和Global Mem的访问，GRF Banks的访问。

数据冒险：warp指令乱序发射基于计分板SB管理依赖。

控制冒险：CPU分支指令要么等待完成，要么预测和重置。

warp内线程分支发散后，两个分支都要执行，在未来收束：SIMT栈、Barrier、动态warp整合。

> **[图片提取文字 (GPU2018_76.png)]:**
> ## What Are the Possible Hazards?
> 
> Three types of hazards we have learned earlier:
> 
> - Structural hazards
> - Data hazards
> - Control hazards
![GPU2018_76.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_76.png)

> **[图片提取文字 (GPU2018_78.png)]:**
> ## Data Hazards
> 
> ![](_page_0_Figure_1.jpeg)
> 
> In practice, we have a much longer pipeline.
![GPU2018_78.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_78.png)

> **[图片提取文字 (GPU2018_80.png)]:**
> ## Additional Hazard: Mem Bank Conflicts
> 
> It is similar to structural hazard.
> 
> Example: 32 threads tries to perform load or store simultaneously to the 32-bank memory.
> 
> ![](_page_0_Figure_3.jpeg)
![GPU2018_80.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_80.png)

> **[图片提取文字 (GPU2018_82.png)]:**
> ## Global Memory Access Coalescing
> 
> Example: 32 thread access the global memory (off-chip DRAM).
> 
> The memory transaction will load/store consecutive data at 32/64/128-Byte boundary.
> 
> Non-aligned access will cause additional memory transactions.
> 
> ![](_page_0_Figure_4.jpeg)
> 
> More examples in backup slides.
![GPU2018_82.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_82.png)

> **[图片提取文字 (GPU2018_84.png)]:**
> ## Let's Start Again from a simple CPU
> 
> Let's start from MIPS.
> 
> ![](_page_0_Figure_2.jpeg)
![GPU2018_84.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_84.png)

> **[图片提取文字 (GPU2018_86.png)]:**
> ## Handling Branch In GPU
> 
> Threads within a warp are free to branch.
> 
> ```
> if( $r17 > $r19 ){
>     $r16 = $r20 + $r31
> }\nelse{
>     $r16 = $r21 - $r32
> }
> $r18 = $r15 + $r16
> ```
> 
> ```
> PC: 0x0010
>         join.label label11
> PC: 0x0011
> set.gt.s32 $p2 | $o127, $r17, $r19
> PC: 0x0012
>     @$p2.ne bra.label labe10
> PC: 0x0013
>    add.u32 $r16, $r20, $r31
> PC: 0x0014
>        bra.label label11
> PC: 0x0015
> label10: sub.u32 $r16, $r21, $r32
> PC: 0x0016
>       label11: nop.join
> PC: 0x0017
>    add.u32 $r18, $r15, $r16
> ```
![GPU2018_86.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_86.png)

> **[图片提取文字 (GPU2018_88.png)]:**
> ## Branch Divergence within a Warp
> 
> - If threads within a warp diverge, both paths have to be executed.
> - Masks are set to filter out threads not executing on current path.
> 
> ![](_page_0_Figure_3.jpeg)
![GPU2018_88.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_88.png)

> **[图片提取文字 (GPU2018_90.png)]:**
> ## Recap SIMT Architecture
> 
> - If no divergent branch, SIMT acts like interleaved threading SIMD. A vector of 32 data lanes has single program state.
> - At divergent branch, data lanes are assigned program states in a branch stack. SIMD becomes SIMT.
> 
> ## Classification of SIMT by Hennessy and Patterson
> 
> |     | Static | Dynamic     |
> |-----|--------|-------------|
> | ILP | VLIW   | Superscalar |
> | DLP | SIMD   | SIMT        |
![GPU2018_90.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_90.png)

> **[图片提取文字 (GPU2018_77.png)]:**
> ## Structural Hazards
> 
> ![](_page_0_Figure_1.jpeg)
![GPU2018_77.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_77.png)

> **[图片提取文字 (GPU2018_79.png)]:**
> ## **Control Hazards**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> We cannot travel back in time. What should we do?
![GPU2018_79.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_79.png)

> **[图片提取文字 (GPU2018_81.png)]:**
> ## Shared Memory (compute capability 2.x)
> 
> **Banks:** 
> 
> Threads:
> 
> Without bank conflict:
> 
> **Banks:** Threads: 10 11-12 13-147 16 **17**-18~ 19/ 22 23/ 24~ 26 27 28-28 29 29 30-
> 
> 31′
> 
> ![](_page_0_Figure_4.jpeg)
> 
> With bank conflict:
> 
> ![](_page_0_Figure_6.jpeg)
> 
> Programmers take the responsibility to eliminate bank conflicts.
![GPU2018_81.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_81.png)

> **[图片提取文字 (GPU2018_83.png)]:**
> ## Additional Hazards: Branch Divergence
> 
> GPUs (and SIMDs in general) have additional challenges: branch divergence.
![GPU2018_83.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_83.png)

> **[图片提取文字 (GPU2018_85.png)]:**
> ## A Naive SIMD Processor
> 
> ![](_page_0_Figure_1.jpeg)
> 
> How to handle branch divergence within a vector?
> 
> WB
![GPU2018_85.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_85.png)

> **[图片提取文字 (GPU2018_87.png)]:**
> ## If No Branch Divergence in a Warp
> 
> • If threads within a warp take the same path, the other path will not be executed.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
![GPU2018_87.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_87.png)

> **[图片提取文字 (GPU2018_89.png)]:**
> ## Dynamic Warp Formation Example: merge two divergent warps into a new warp if possible.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ref: Dynamic warp formation: Efficient MIMD control flow on SIMD graphics hardware. http://dx.doi.org/10.1145/1543753.1543756
![GPU2018_89.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_89.png)

> **[图片提取文字 (GPU2018_91.png)]:**
> ## **Elaborated Microarchitecture**
> 
> Previous slides show a simplified architecture. Below is an elaborated example.
> 
> ![](_page_0_Figure_2.jpeg)
![GPU2018_91.png](CUDA%E7%A8%8B%E5%BA%8F%E5%9C%A8GPU%E7%9A%84%E6%89%A7%E8%A1%8C%E8%BF%87%E7%A8%8B/GPU2018_91.png)