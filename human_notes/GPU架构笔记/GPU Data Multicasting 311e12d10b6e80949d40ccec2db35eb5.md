# GPU Data Multicasting

ref：PROGRAMMATICALLY CONTROLLED DATA MULTICASTING ACROSS MULTIPLE COMPUTE ENGINES

## Fig1

Strong Scaling是增多SM的数量来增大并行，weak Scaling是增大SM的tile来增加并发。

scaling描述负载的特点，strong scaling是负载有**硬性的计算或存储需求**，通常来源于单输入的算法复杂度，weak scaling是负载有**弹性的计算或存储需求**，通常来源于动态请求或输入。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 1A
![image.png](GPU%20Fast%20Sync%20HW%20SW/image.png)

> **[图片提取文字 (image.png)]:**
> ## Weak scaling DL. Output Activations FIG. 1B
> 
> ## Strong scaling
> 
> ![](_page_0_Picture_2.jpeg)
![image.png](GPU%20Fast%20Sync%20HW%20SW/image%201.png)

tileSz二次方变小（tile并发），tileNum二次方变大，但带宽需求一次方减小（带宽并行），所有tile的带宽要求变高。

读一次Cache、广播复用、计算

> **[图片提取文字 (image.png)]:**
> three different GPUs, from left to right representing increasing numbers of processors (in this example, streaming multiprocessors (SM)) and L2 slices per GPU, as the tile size is made smaller. The pattern of sharp increase in the L2 bandwidth demand in larger more powerful GPUs as the tile size is made smaller is clear in FIG. 1D. [0044] Example embodiments of this disclosure leverage data fetch redundancy to reduce the bandwidth and power required to move the same amount of data and better scale. Example embodiments increase effective L2 bandwidth by multicast of response data from one L2 read to multiple cooperative thread arrays (CTA) on multiple SMs. In some embodiments, the CTAs receiving the multicast response all belong to the same corporative group array (CGA). Example applications that can benefit significantly from this disclosure include generic matrix multiply (GEMM) and GEMM-
> 
> like kernels.
> 
> [0043] FIG. 1D illustrates the L2 bandwidth demand in
![image.png](GPU%20Data%20Multicasting/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 1D
![image.png](GPU%20Data%20Multicasting/image%201.png)

> **[图片提取文字 (image.png)]:**
> In previous NVidia GPUs, the A1 element was separately fetched from the memory (typically from the L2 cache) for calculating each element in the first row of the result matrix. Likewise, the B1 element was separately fetched for calculating each element in the first column of the result matrix. The fetching of the same element (e.g. A1, B1) several times from the L2 cache is an example data fetch redundancy that results in bandwidth (e.g. memory bandwidth, crossbar (Network on Chip-NOC) bandwidth) and power both being wasted. [0047] Example embodiments of this disclosure leverage this data fetch redundancy to reduce the bandwidth and power required to move the same amount of data and better scale.
> 
> [0046] As can be seen in FIG. 1E, each cell (element) of
> 
> the first row of the result matrix utilizes element A1 from the
> 
> input matrix A, and each cell of the first column of the result
> 
> matrix utilizes element B1 from the input matrix B. In
> 
> general, each cell in the result matrix utilizes some input
> 
> data that is the same as in the input data of its adjacent cells.
![image.png](GPU%20Data%20Multicasting/image%202.png)

> **[图片提取文字 (image.png)]:**
> |   | B1    | B2    | B3    | B4    |
> |---|-------|-------|-------|-------|
> | - | A1*B1 | A1*B2 | A1*B3 | A1*B4 |
> | - | A2*B1 | A2*B2 | A2*B3 | A2*B4 |
> | - | A3*B1 | A3*B2 | A3*B3 | A3*B4 |
> |   | A4*B1 | A4*B2 | A4*B3 | A4*B4 |
> 
> A2
> 
> A3
> 
> A4
> 
> ## FIG. 1E
![image.png](GPU%20Data%20Multicasting/image%203.png)

## Fig2、3、4、5、6

L2-Cache data multcast

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 2A
![image.png](GPU%20Data%20Multicasting/image%204.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ## FIG. 2B
![image.png](GPU%20Data%20Multicasting/image%205.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3
![image.png](GPU%20Data%20Multicasting/image%206.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 4
![image.png](GPU%20Data%20Multicasting/image%207.png)

CGA Mem Map、MIG情景

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 5
![image.png](GPU%20Data%20Multicasting/image%208.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 6
![image.png](GPU%20Data%20Multicasting/image%209.png)

特殊load：读一次Cache，广播到多个SM。

特殊load支持IM2COL模式的加载。

> **[图片提取文字 (image.png)]:**
> [0086] In an example embodiment, a multicast load instruction implemented on TMA may have a format such as the following: [0087] UTMALDG.dim{.IM2COL}{.MULTICAST} [URb], [URa], URc . . . ; .dim: {0.1D, 0.2D, 0.3D, 0.4D, 0.5D}-Tensor dimensionality
![image.png](GPU%20Data%20Multicasting/image%2010.png)

> **[图片提取文字 (image.png)]:**
> - [0088] .IM2COL: Enables image-to-column load mode. The image-to-column mode support of the TMA is described in the U.S. application Ser. No. 17/691,276 which was incorporated above.
> - [0089] URb: Source B uniform register. Packed destination address, shared memory barrier address and tensor coordinates.
> - [0090] {URb, URb+1} specifies the destination data/barrier distributed shared memory address as follows:
> 
> ![](_page_0_Figure_3.jpeg)
> 
> - [0091] URa: Source A uniform register. Specifies the global memory address of the tensor descriptor.
> - [0092] .MULTICAST: Enables multicast mode.
> - [0093] URc: Source C uniform register. Multicast CTA ids (or SM ids), and optionally also one or more (e.g., up to three) tensor coordinate offsets for .IM2COL.
![image.png](GPU%20Data%20Multicasting/image%2011.png)

> **[图片提取文字 (image.png)]:**
> [0094] In some embodiments, the CTA ID mask may be encoded in the following format:
> 
> ```
> 31 16 15 0
> +-----+
> | Multicast CTA_ID mask | im2col offsets|
> +-----+
> ```
> 
> [0095] In some embodiments, all CTAs in the CGA may be multicast destinations. However, in some embodiments up to 16 destination CTA IDs are encoded in a 16-bit mask included in URc, where each bit corresponds to a CTA ID. In some embodiments, although a CGA may have up to 32 CTAs only the first 16 CTAs from [0:15] range may be able participate in the multicast mode.
> 
> [0096] Destination SMs may not have metadata that describe how the received data is to be processed (e.g., such as, received data should be written in image-to-column format, etc.), unlike source SMs. Therefore, all metadata necessary to handle the responses must be sent with the packet. "Metadata" transported from source SM to destination SMs may include, for example:
> 
> - [0097] SM ID mask corresponding to the destination SMs (CTA IDs),
> - [0098] CGA ID,
> - [0099] Data SMEM (shared memory) Offset,
> - [0100] Barrier address Offset,
> - [0101] Source SM ID for responses,
> - [0102] ACK phase ID (two possible phases, part of the MEMBAR protocol), and
> - [0103] implementation specific result data processing parameters.
![image.png](GPU%20Data%20Multicasting/image%2012.png)

## Fig7、8、9

multicast example in GPU

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Data%20Multicasting/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 8
![image.png](GPU%20Data%20Multicasting/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 9A
![image.png](GPU%20Data%20Multicasting/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 9B
![image.png](GPU%20Data%20Multicasting/image%2016.png)