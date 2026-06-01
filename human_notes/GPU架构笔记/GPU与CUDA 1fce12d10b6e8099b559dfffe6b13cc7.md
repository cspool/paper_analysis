# GPU与CUDA

# 1、CUDA初识

## 1、CUDA作用

使用 CUDA C/C++语言编写并行程序，通过调用 CUDA API 将计算任务发送到 GPU 执行。

CUDA 编程模型包括主机（CPU）和设备（GPU）之间的协作，此外还提供了对其它编程语言的支持，比如 C/C++，Python，Fortran 等语言，支持 OpenCL 和 DirectCompute 等应用程序接口。

因此CUDA框架也是CPU将计算任务卸载到GPU执行.

## 2、CUDA组成

CUDA 从下到上由Driver、编程工具包(编程接口实现)、运行库（Runtime）、加速算法库组成。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Picture_1.jpeg)
> 
> OS PLATFORMS
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Picture_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
> 
> ![](_page_0_Picture_6.jpeg)
> 
> ![](_page_0_Picture_7.jpeg)
![image.png](GPU%E4%B8%8ECUDA/image.png)

- CUDA DRIVER 驱动 GPU 负责内存和图像管理。
    - 基于C语言实现
    - 协同操作系统内核模块,提供对GPU内存和计算核的抽象和管理.
- CUDA Runtime:提供GPU**运行时所需功能**的函数库，但这些函数调用对应用开发者透明（无需关注）
- CUDA TOOLKIT包括nvcc编译器和 C++ Core,**为开发者提供功能的库和工具.**
    - CUDA C++ Core是一组库，提供GPU上进行**C++编程**所需特性的实现，作用类似标准库
    - nvcc将开发者编写的CUDA代码分离出host和device代码,分别编译出host和device的目标代码.
- CUDA-X LIBRARIES 主要提供了机器学习（Meachine Learning）、深度学习（Deep Learning）和高性能（High Performance Computing）计算方面的加速库.
- APPS & FRAMEWORKS 主要对接 TensorFlow 和 Pytorch 等框架。

# 2、CUDA编程

## 1、执行模式

CUDA 引入主机端（host）和设备（device）概念，CUDA 程序中既包含主机（host）程序也包含设备（device）程序，host 和 device 之间可以进行通信，以此来实现数据拷贝，主机负责管理数据和控制程序流程，设备负责执行并行计算任务。

**在 CUDA 编程中，Kernel 是在 GPU 上并行执行的函数，开发人员编写 Kernel 来描述并行计算任务，然后在主机上调用 Kernel 来在 GPU 上执行计算。**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Iterative
> 
> ## CUDA Kernel 1
> 
> **Horizontal Processing** 
> 
> - 1. Update APP values
> - 2. Hard decisions
> 
> ## CUDA Kernel 2
> 
> ## Serial code
> 
> 1. Transfer data from Device to Host
> 
> Host
> 
> (CPU)
> 
> 2. Finish decoding
> 
> ![](_page_0_Figure_13.jpeg)
> 
> ![](_page_0_Picture_14.jpeg)
![image.png](GPU%E4%B8%8ECUDA/image%201.png)

## 2、代码示例

在 CUDA 程序架构中，host 代码部分在 CPU 上执行，是普通的 C 代码。当遇到数据并行处理的部分，CUDA 会将程序编译成 GPU 能执行的程序，并传送到 GPU，这个程序在 CUDA 里称做核(Kernel)。device 代码部分在 GPU 上执行，此代码部分在 Kernel 上编写(.cu 文件)。

- CPU执行张量加法代码cuda_host.cpp
    
    ```cpp
    #include <iostream>
    #include <math.h>
    #include <sys/time.h>
    
    // function to add the elements of two arrays
    void add(int n, float *x, float *y)
    {
        for (int i = 0; i < n; i++)
            y[i] = x[i] + y[i];
    }
    
    int main(void)
    {
        int N = 1<<25; // 30M elements
    
        float *x = new float[N];
        float *y = new float[N];
    
        // initialize x and y arrays on the host
        for (int i = 0; i < N; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
    
        struct timeval t1,t2;
        double timeuse;
        gettimeofday(&t1,NULL);
    
        // Run Kernel on 30M elements on the CPU
        add(N, x, y);
    
        // Free memory
        delete [] x;
        delete [] y;
    
        return 0;
    }
    ```
    
- 代码 `cuda_device.cu` 是使用 CUDA 编程实现 GPU 计算张量加法.
    
    Kernel 用 `__global__` 符号声明，在调用时需要用 `<<<grid, block>>>` 来指定 Kernel 要执行及结构。
    
    ```cpp
    #include <iostream>
    #include <math.h>
    
    // Kernel function to add the elements of two arrays
    // __global__ 变量声明符，作用是将 add 函数变成可以在 GPU 上运行的函数
    // __global__ 函数被称为 Kernel
    __global__
    void add(int n, float *x, float *y)
    {
      for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
    }
    
    int main(void)
    {
      int N = 1<<25;
      float *x, *y;
    
      // Allocate Unified Memory – accessible from CPU or GPU
      // 内存分配，在 GPU 或者 CPU 上统一分配内存
      cudaMallocManaged(&x, N*sizeof(float));
      cudaMallocManaged(&y, N*sizeof(float));
    
      // initialize x and y arrays on the host
      for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
      }
    
      // Run Kernel on 1M elements on the GPU
      // execution configuration, 执行配置
      add<<<1, 1>>>(N, x, y);
    
      // Wait for GPU to finish before accessing on host
      // CPU 需要等待 cuda 上的代码运行完毕，才能对数据进行读取
      cudaDeviceSynchronize();
    
      // Free memory
      cudaFree(x);
      cudaFree(y);
      
      return 0;
    }
    ```
    

## 3、编程流程

- 编写 Kernel 函数描述并行计算任务。
- 在主机上配置线程块和网格，将 Kernel 发送到 GPU 执行。
- 在主机上处理数据传输和结果处理，以及控制程序流程。

## 4、线程模型

- 线程层次结构Ⅰ-Grid：Kernel 在 device 上执行时，实际上是启动很多线程，**一个 Kernel 所启动的所有线程称为一个网格（grid），同一个网格上的线程共享相同的全局内存空间**，grid 是线程结构的第一层次。
    - **同一个kernel的任意调用都共享一个global memory?**
- 线程层次结构Ⅱ-Block：**Grid 分为多个线程块（block），一个 block 里面包含很多线程，Block 之间并行执行，并且无法通信，也没有执行顺序，每个 block 包含共享内存（shared memory）**，被里面的 Thread共享。
- 线程层次结Ⅲ-Thread：一个线程 block包含多个threads，**同一个 block 中 threads 可以同步，也可以通过 shared memory 通信**。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%E4%B8%8ECUDA/image%202.png)

**每次kernel调用需要指明grid和blk,为什么?**

- CUDA 和英伟达硬件架构有以下对应关系:
    - 从软件侧看到的是线程的执行，对应于硬件上的 CUDA Core，每个线程对应于 CUDA Core，软件方面线程数量是超配的，硬件上 CUDA Core 是固定数量的。
    - Block 线程块只在一个 SM 上通过 Warp 进行调度，一旦在 SM 上调用了 Block 线程块，就会一直保留到执行完 Kernel，SM 可以同时保存多个 Block 线程块.
    - 多个 SM 组成的 TPC (thread process cluster)和 GPC (graphic process cluster)硬件实现了 GPU 并行计算。

> **[图片提取文字 (image.png)]:**
> ## **Software** Hardware **CUDA Core** Thread Thread Block
> 
> Grid
> 
> ![](_page_0_Picture_1.jpeg)
![image.png](GPU%E4%B8%8ECUDA/image%203.png)

## 5、GPU算力

公式如下:$Peak FLOPS=𝐹clk×𝑁SM×𝐹req$,其中Fclk是每周期的指令吞吐数

- 𝐹clk：GPU 时钟周期内指令执行数 (FLOPS/Cycle)
- 𝑁SM：SM（Streaming Multiprocessor）数量
- 𝐹req：Tensor Core 核心运行频率（GHz）

以英伟达 A100 为例，其中 **Tensor Core的FP32**精度指令吞吐 64 FLOPS/Cycle ，核心运行频率为 1.41GHz ，SM 数量为 108 ，因此 GPU 的FP32 Tensor Core的算力峰值是:

$𝑃𝑒𝑎𝑘𝐹𝐿𝑂𝑃𝑆=1.41∗108∗64∗2=19,491𝐺𝐹𝐿𝑂𝑃𝑆$

(乘累加融合)