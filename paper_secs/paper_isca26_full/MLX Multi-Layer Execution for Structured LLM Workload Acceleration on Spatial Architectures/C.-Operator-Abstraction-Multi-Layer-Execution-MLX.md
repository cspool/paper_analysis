# C. Operator Abstraction: Multi-Layer Execution (MLX)

Chunked FFT and hierarchical BSMM can be expressed as sequences of stages with layer-aligned, forward-only dependencies. More broadly, other structured-sparse operators with staged, stride-regular dependencies also fit this abstraction. Because each stage has a bounded array-resident footprint, execution can be folded over time: only a subset of stages

![](_page_4_Figure_9.jpeg)

Fig. 8: Pipeline computations across multiple butterfly-sparse matrix multiplications (BSMMs).

resides on the array at once, while the others are timemultiplexed in dependency order. We call this execution abstraction MLX, which decouples logical stage depth from physical array size, enabling deep pipelining through folded execution on a compact PE array. A detailed formalization of MLX is provided in Sec. V-B.

#### IV. MLX ARCHITECTURE

This section describes how to realize MLX paradigm in hardware. As overviewed in Fig. 9, the architecture consists of a host controller, scratchpad memory and a mesh of processing elements (PEs) connected by a hop-encoded network.

#### A. BSMM as the Motivating Case for MLX Design

Among the operators in our hybrid model, BSMM provides the clearest illustration of why MLX is required. As shown in Fig. 8(b), each BSMM layer consumes the immediate output of the previous one, forming a deep and strictly layered dataflow with entirely predictable pipelined dependency [30, 31, 32]. In principle, consecutive BSMM layers could overlap on a spatial array to expose substantial but fine-grained parallelism. However, the full BSMM dataflow graph is far too large and too deeply layered to map onto a fixed-size mesh at once. Once compute units are shared across BSMM layers, the accelerator must introduce additional specialization to sustain high throughput: (1) schedule instructions so that different BSMM layers can execute in a staggered highly utilized fashion among FUs and (2) route intermediate results through short, predictable paths to their explicit downstream.

Our goal is to build such an accelerator that is specialized for large structured dataflow graphs with moderate arithmetic intensity, capable of orchestrating overlapped layer execution and explicit low-latency data transfer, as presented in Fig. 9(a).

## B. Skip-Hop NoC Topology for Layer-Folded Execution

Layer folding turns cross-layer dependencies into bounded, regular communication patterns. In BSMM and FFT, each folded layer accesses deterministic stride- $2^k$  neighbors, which are poorly served by global-memory traffic but naturally match a topology-aware mesh NoC. MLX therefore adopts a *skip-hop mesh* (Fig. 9(b)), extending each PE with fixed-distance links in addition to local-neighbor forwarding. These links directly span the folded dependency radius and reduce most cross-layer transfers to one or two hops.

To realize these transfers with minimal hardware state, MLX uses a hop-encoded data-movement primitive. Each xfer instruction carries only a residual hop count, routing direction, and destination register. Routers are stateless: when the hop count reaches zero, the value is written locally. Otherwise, the router consumes the largest admissible step—unit or skip—and forwards the data packet. This converts structured MLX dependencies into deterministic bounded-hop transfers, avoiding routing tables, virtual channels, and dynamic route computation. The same primitive naturally covers butterfly strides, FFT pairings, dense-MM systolic motion, and bounded window interactions (Sec. V-C), providing a unified spatial substrate for folded execution.

