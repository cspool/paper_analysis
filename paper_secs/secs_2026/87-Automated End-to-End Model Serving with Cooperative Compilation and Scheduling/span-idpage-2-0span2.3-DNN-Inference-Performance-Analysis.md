# <span id="page-2-0"></span>2.3 DNN Inference Performance Analysis

Before designing our system, we first go through some stateof-the-practice DNN inference works, identify their limitations, and analyze the inference performance. Modern

<span id="page-3-0"></span>![](_page_3_Figure_2.jpeg)

Figure 4: GEMM kernel performance influenced by register file level (RF-level) intensity, ILP, and TLP. The is implemented via RF-level tile configuration [61].

DL frameworks follow a compile-then-schedule paradigm, where operators [22] are instantiated as GPU kernels [28, 32, 61] and subsequently managed by their schedulers for execution on GPUs.

**Compilation.** The goal of compilation is to generate high-performance kernels for operators by applying various optimization techniques including data layout transformation [56], operator fusion [60], and loop optimization [59].

Existing approaches suffer from two critical limitations. First, compiling operators as monolithic units neglects fine-grained inter-operator data dependencies, thus missing plenty of parallelization opportunities. The computations within a compiled operator are forced to happen simultaneously because kernels are the minimum scheduling units. For example, an image tensor tile can propagate through successive convolution operators without needing to sequentially compute each operator in full. Second, most kernel generation practices either require substantial domain expertise for labor-intensive manual optimization [1, 10] or involve computationally expensive compilation processes [28, 59], which prevents their large-scale real-world application.

Next, we analyze the performance of highly-optimized DNN kernels. Given that memory and math instructions comprise over 90% of high-performance kernels' makeup [34], we can calculate the number of instructions by

<span id="page-3-1"></span>#inst 
$$\approx$$
 #inst<sub>mem</sub> + #inst<sub>math</sub>

$$= a \cdot #bytes + b \cdot #ops = #ops \cdot \left(\frac{a}{I} + b\right), \tag{3}$$

where a and b are constants, #ops is a constant for a given DL workload, and I denotes the ratio of the number of arithmetic operations to the number of bytes accessed (i.e., the algorithm's arithmetic intensity). This equation is applicable to any two memory levels with data transfer. It reveals

<span id="page-3-2"></span>

| Compilation | Scheduling Strategy |         |         |
|-------------|---------------------|---------|---------|
| Strategy    | unfused             | b-fused | w-fused |
| unfused     | 2.55 ms             | 2.91 ms | 3.57 ms |
| b-fused     | $2.73\mathrm{ms}$   | 2.48 ms | 2.94 ms |
| w-fused     | 2.83 ms             | 2.72 ms | 2.27 ms |

Table 1: Toy experiment colocating op MatMul and op Add. "unfused", "b-fused", "w-fused" refer to stream parallelism, block-level horizontal kernel fusion, and warp-level horizontal kernel fusion respectively [34].

that higher intensity leads to a reduced instruction count, which accelerates execution as per Eq. (1). This yields a new optimization metric, intensity. However, integrating ILP into Eq. (2) proposes challenges as ILP and intensity likewise form a trade-off competing for registers and shared memory. Fig. 4 demonstrates the trade-off by showing that the peak performance appears in the green box where ILP, TLP, and intensity are balanced. By optimizing Eq. (1) with the combination of Eqs. (2) and (3), we can qualitatively formalize the trade-off as

<span id="page-3-3"></span>
$$\max ILP \cdot Intensity$$
, s.t.  $TLP \ge 4$ . (4)

**Scheduling.** Most DL inference systems' schedulers [32, 42, 44, 49] run kernels generated at the compilation phase. However, these schedulers have significant design and implementation limitations.

For independent kernels, the schedulers struggle to achieve proper parallelization. For example, the inner scheduler of NVIDIA GPUs dispatches thread blocks from one kernel at a time, leading to implicit synchronization at the kernel launch point ("operator-based kernel direct" in Fig. 2). Therefore, kernel parallelism is only realized during the brief overlap of two kernels. Kernels monopolize GPUs sequentially, and their monotonous instruction pattern brings an imbalanced load to the hardware pipeline units, ultimately degrading GPU performance.

For dependent kernels, the schedulers trigger tail effect and cold start. When two consecutive dependent kernels are added to the same CUstream queue, the latter has to wait for the former to finish before being launched. The tail effect arises from insufficient threads and epilogue pipeline bubble during the former kernel's ending phase. Similarly, the cold start occurs at the start of subsequent kernels, which stems from the kernel preamble section including thread block dispatching, resource allocation, and prologue pipeline bubbles. Our measurements on an A100 show GPU idle intervals are within 1–3 microseconds, with overhead becoming more pronounced as GPUs incorporate more cores [2].

**Coupled compile-schedule.** To illustrate the interplay between compilation and scheduling in GPU performance,

we conduct an operator colocation experiment (Table 1). We run two operators MatMul and Add in parallel with different compilation and scheduling strategies. For instance, the scheduling strategy "b-fused" alternates thread block emissions from two kernels during inference, while the compilation strategy "w-fused" creates a large kernel with two operators fused at warp level. The result shows a diagonal principle that optimal performance appears only when the compilation strategy aligns with the scheduling strategy.

This is expected as tuning for a specific scheduling strategy yields the best results [32, 42]. Consequently, achieving optimal performance requires co-designing compilation and scheduling strategies. However, it is a difficult joint optimization problem. Moreover, the inference-time environment is complex and unpredictable, which makes traditional deterministic compilation strategies [32, 48, 61] unfeasible.

### <span id="page-4-2"></span>2.4 Our Scheme

We propose our solution with a two-phase design: generate multiple kernel versions at compile time, and schedule them dynamically at inference time. This design successfully addresses the tight coupling between compilation and scheduling by transforming them from dynamic coupling to static coupling. It elevates Halide's decoupling of algorithms and schedules [46] to a new level with scheduling at inference time rather than compile time [32]. It is able to solve all above concerns as follows.

Compilation (§ 4). At compile time, the compiler partitions large operators into smaller ones (micro operators) and generates multiple micro-kernel versions for them. Micro operators extend data asynchrony to sub-operator level, while different kernel versions provide various trade-offs between ILP, TLP, and intensity. Unlike previous methods that compile, profile, and discard a large amount of kernels, our zero-tuning compilation strategy directly produces high-quality kernels based on static analysis with minimal overhead. Additionally, zero-tuning allows for hardware-agnostic compilation and program portability.

**Scheduling (§ 5).** At inference time, aimed at Eq. (4), the scheduler dynamically assigns the micro kernels for DNN execution based on kernel property, task/GPU status, etc. Dynamic scheduling is able to eliminate overhead like cold start and tail effect. In addition, our sophisticated and comprehensive scheduling system effectively manages heterogeneous user inference jobs with diverse requirements.

#### <span id="page-4-1"></span>3 Overview

We describe Infera by using an example to walk through its components. Fig. 5 summarizes the inference steps in Infera. A user application first compiles a DNN model with the compiler  $(\mathfrak{I})$ , an offline static program. The application

<span id="page-4-3"></span>![](_page_4_Figure_10.jpeg)

Figure 5: Overview of Infera system. The solid lines represent data (③④) and signals (⑤) involved in the critical inference process, while the dashed lines indicate the preparatory works before inference, including compilation (①) and registration of models (②).

can then upload the compiled modules and the parameters to the inference model pool (②). After that, a user application can submit DNN inference jobs (③), which are then executed end-to-end by the inference server (④). Finally, the inference server notifies the user application of job completion, at which point the application can access the output data (⑤).

**End-user example.** In a few lines of code, a user can take a model of ONNX format [22] and call the Infera API to get a compiled module.

```
import infera as inf
raw_model = inf.import_model(onnx_model)
target = inf.device.gpu(gpu_id)
# compile a model with zero tuning
rt_model = inf.compile(raw_model, target)
```

The runtime model rt\_model is mainly composed of a model structure and compile-generated compute kernels. It can be uploaded to the online model pool of Infera inference server.

```
import infera.runtime as infrt
# upload the model with weights to the online pool
model_tpl_id = infrt.register_model(rt_model)
model_id = infrt.register_param(model_tpl_id, params)
```

With all prerequisite work completed, a user can easily submit inference jobs specifying the model and the input, then waiting for the output.

```
# submit a job and wait for asynchronous execution
job_id, data_out = infrt.submit(model_id, data_in)\ninfrt.wait(job_id)
```

## <span id="page-4-0"></span>4 Compiling Models

The Infera compiler (Fig. 6) automates DNN model compilation with the following steps. First, it converts an ONNX model [22] into a computation graph in the TVM Relay format [4]. The computation graph can be modified with the help of TVMScript [4]. Then, the tailored tile-based TVM compiler [5] generates tile-based tensor programs TensorIR [4] for the graphs's operators (§ 4.1), performing graph optimizations such as operator fusion and layout transformation. After the tensor programs have been compiled into basic CUDA programs framework with the TVM code generator, the code optimizer is used to reconstruct and modify the

CUDA kernel at various code levels, including CUDA, PTX, and SASS (§ [4.2\)](#page-5-2). Finally, the compiler consolidates all the generated code and data into static libraries.

