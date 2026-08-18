# CODO: An Automated Compiler for Comprehensive Dataflow Optimization

Weichuang Zhang School of Computer Science Shanghai Jiao Tong University Shanghai, China 1064080006@sjtu.edu.cn

Chi Zhang School of Computer Science Shanghai Jiao Tong University Shanghai, China zhang-chi@sjtu.edu.cn

Chao Li School of Computer Science Shanghai Jiao Tong University Shanghai, China lichao@cs.sjtu.edu.cn

Yiquan Wang School of Computer Science Shanghai Jiao Tong University Shanghai, China abcdfehg@sjtu.edu.cn

Yu Feng School of Computer Science Shanghai Jiao Tong University Shanghai, China y-feng@sjtu.edu.cn

Jieru Zhao<sup>∗</sup> School of Computer Science Shanghai Jiao Tong University Shanghai, China zhao-jieru@sjtu.edu.cn

Xinzhou Zhang School of Computer Science Shanghai Jiao Tong University Shanghai, China xz zhang@sjtu.edu.cn

Xiaofeng Hou School of Computer Science Shanghai Jiao Tong University Shanghai, China hou-xf@cs.sjtu.edu.cn

Minyi Guo Guizhou Provincial Laboratory of Big Data College of Computer Science and Technology, Guizhou University School of Computer Science, SJTU guo-my@cs.sjtu.edu.cn

*Abstract*—FPGAs are well-suited for dataflow architectures that process data in a streaming or pipelined manner, thus satisfying the high computational and communication demands of emerging applications. However, manually implementing an efficient dataflow architecture for large-scale applications is still challenging, even for specialists who use high-level synthesis (HLS) to simplify FPGA programming.

To address this, we introduce CODO, an automated compiler that generates feasible and efficient dataflow accelerators on FPGAs. CODO features a systematic method for detecting and eliminating both coarse-grained and fine-grained dataflow violations. Building on this, CODO performs both on- and off-chip data movement optimizations to maximize transfer efficiency. To guarantee a higher design quality, CODO performs automatic scheduling to generate high-performance dataflow accelerators, ensuring a balanced performance-resource tradeoff. Synthesis results show that CODO delivers 1.45× to 4.52× latency speedups on typical computation kernels and 3.7× to 33.8× speedups on DNN models compared to SOTA frameworks. In on-board evaluations, CODO achieves 7.3× average speedup on CNN models and 2.07× average speedup on the GPT-2 model over SOTA frameworks. The compiler is open-sourced at [https://github.com/sjtu-zhao-lab/codo-artifact.](https://github.com/sjtu-zhao-lab/codo-artifact)

# CODO: An Automated Compiler for Comprehensive Dataflow Optimization

Weichuang Zhang School of Computer Science Shanghai Jiao Tong University Shanghai, China 1064080006@sjtu.edu.cn

Chi Zhang School of Computer Science Shanghai Jiao Tong University Shanghai, China zhang-chi@sjtu.edu.cn

Chao Li School of Computer Science Shanghai Jiao Tong University Shanghai, China lichao@cs.sjtu.edu.cn

Yiquan Wang School of Computer Science Shanghai Jiao Tong University Shanghai, China abcdfehg@sjtu.edu.cn

Yu Feng School of Computer Science Shanghai Jiao Tong University Shanghai, China y-feng@sjtu.edu.cn

Jieru Zhao<sup>∗</sup> School of Computer Science Shanghai Jiao Tong University Shanghai, China zhao-jieru@sjtu.edu.cn

Xinzhou Zhang School of Computer Science Shanghai Jiao Tong University Shanghai, China xz zhang@sjtu.edu.cn

Xiaofeng Hou School of Computer Science Shanghai Jiao Tong University Shanghai, China hou-xf@cs.sjtu.edu.cn

Minyi Guo Guizhou Provincial Laboratory of Big Data College of Computer Science and Technology, Guizhou University School of Computer Science, SJTU guo-my@cs.sjtu.edu.cn

*Abstract*—FPGAs are well-suited for dataflow architectures that process data in a streaming or pipelined manner, thus satisfying the high computational and communication demands of emerging applications. However, manually implementing an efficient dataflow architecture for large-scale applications is still challenging, even for specialists who use high-level synthesis (HLS) to simplify FPGA programming.

To address this, we introduce CODO, an automated compiler that generates feasible and efficient dataflow accelerators on FPGAs. CODO features a systematic method for detecting and eliminating both coarse-grained and fine-grained dataflow violations. Building on this, CODO performs both on- and off-chip data movement optimizations to maximize transfer efficiency. To guarantee a higher design quality, CODO performs automatic scheduling to generate high-performance dataflow accelerators, ensuring a balanced performance-resource tradeoff. Synthesis results show that CODO delivers 1.45× to 4.52× latency speedups on typical computation kernels and 3.7× to 33.8× speedups on DNN models compared to SOTA frameworks. In on-board evaluations, CODO achieves 7.3× average speedup on CNN models and 2.07× average speedup on the GPT-2 model over SOTA frameworks. The compiler is open-sourced at [https://github.com/sjtu-zhao-lab/codo-artifact.](https://github.com/sjtu-zhao-lab/codo-artifact)

