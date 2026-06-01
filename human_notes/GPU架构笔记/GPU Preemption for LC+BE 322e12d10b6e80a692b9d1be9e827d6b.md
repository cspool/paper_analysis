# GPU Preemption for LC+BE

## wait-based

**wait-based**：等待1个kernel grid后抢占，或者1个block后抢占。

### Effisha

### Globally scheduled real-time

> **[图片提取文字 (image.png)]:**
> ```
> # device codes
>   global___ void conv_relu(in, weight, out):
> 1 sum = 0;
> 2 for i in range(0,3)
> 3
>       for j in range(0,3)
> 4
>           sum += in[...] \times weight[...]
> 5
>    out[...] = ReLU(sum)
>   global__ void dense(in, weight, bias, out):
>    sum = 0;
> 6
> 7 for i in range(0,512)
>        sum += in[...] \times weight[...]
> 8
>     out[..] = sum + bias[..]
> 9
> # host codes
> void inference(...):
> 10
>    memcpyH2D(in, in_host, in_sz) # copy in to GPU
> 11 conv_relu <<<dim(32), ..>>> (in, .., buf_conv)
> 12 ... # launch other kernels
> 13 pooling <<<dim(64), ..>>> (.., buf_pool)
> 14 dense <<<dim(10), ...>>> (buf_pool, ..., buf_dense)
> 15 softmax <<<dim(1), ..>>> (buf_dense, .., out)
> 16
>    memcpyD2H(out_host, out, out_sz) # copy out to CPU
> ```
> 
> **Fig. 2:** An example of DNN inference using a model like ResNet.
![image.png](GPU%20Preemption%20for%20LC+BE/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **Fig. 3:** (a) The CDF of execution time of several typical kernels in VGG, and (b) the timeline of CU usage during VGG execution on a GPU with 60 CUs. Note that the execution time of GPU kernels in VGG covers a fairly wide range from 10 µs to 255 µs (see Fig. 10).
![image.png](GPU%20Preemption%20for%20LC+BE/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **Fig. 1:** (a) The overall throughput of DNN inferences (both real-time and best-effort tasks) and (b) the end-to-end latency of real-time tasks when using concurrent GPU scheduling (i.e., multiple GPU streams [46, 49, 60]), (c) the end-to-end latency of real-time tasks when using preemptive GPU scheduling (i.e., wait-based preemption [12, 77, 90]), and (d) the throughput of best-effort tasks as the frequency of real-time tasks increases. **Workload**: VGG [68] (real-time) and ResNet [30] (best-effort). **Testbed**: one AMD Radeon Instinct MI50 GPU with 16 GB of memory (see §7 for details).
![image.png](GPU%20Preemption%20for%20LC+BE/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **Fig. 4:** An example of GPU task scheduling with different kernel preemption and parallelism schemes for a hybrid workload, which contains two best-effort and one real-time DNN inference tasks. The GPU has four compute units (CUs).
![image.png](GPU%20Preemption%20for%20LC+BE/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **Fig. 5:** Architecture of REEF. Modules in boxes with dashed border are on the critical path of serving DNN inference requests. Other modules do not directly impact serving latency and throughput.
![image.png](GPU%20Preemption%20for%20LC+BE/image%204.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **Fig. 6:** An example of timeline in REEF. The DNN inference tasks here are the same as that in Fig. 4.
![image.png](GPU%20Preemption%20for%20LC+BE/image%205.png)

动态kernel padding

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **Fig. 7:** Extended GPU runtime in REEF for instant preemption.
![image.png](GPU%20Preemption%20for%20LC+BE/image%206.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **Fig. 8:** An example of serving multiple kernels in parallel with different approaches.
![image.png](GPU%20Preemption%20for%20LC+BE/image%207.png)

> **[图片提取文字 (image.png)]:**
> ```
> # device codes
>  _device__ void dense(in, weight, bias, out): ...
>   global void dkp(rt_kern, rt_args,
>                     be kerns, be argss):
>     ncus = rt_kern.ncus # number of CUs
> 1
> 2
>     if (cu_id() < ncus) then</pre>
> 3
>        rt kern(rt args) # run RT/kernel
>     else
> 4
> 5
>        ncus += be_kerns[i=0].ncus
>        while (cu_id() >= ncus)
> 6
>           ncus += be_kerns[++i].ncus
> 7
>        be_kerns[i](be_argss[i]) # run BE/kernel
> 8
> # host codes
> void inference(...):
>     # set the real-time kernel w/ its args (e.g., dense)
> 9 rt_kern, rt_args = ...
>     # select a set of best-effort kernels w/ their args
>    be_kerns, be_argss = kern_select(rt_kern)
> 10
> 11 dkp <<<...>>> (rt_kern, rt_args, be_kerns, be_argss)
> 12 ... # launch other dynamic padded kernels
> ```
> 
> **Fig. 9:** Pseudocode for dynamic kernel padding in REEF.
![image.png](GPU%20Preemption%20for%20LC+BE/image%208.png)

### GPES

## reset-based

0抢占延迟，但要求幂等kernel？

### Microsecond-scale preemption

### Chimera

## switch-based

所谓的switch？和指令level抢占有什么区别？

### **GPreempt**