# C. ASADI Complexity

**Time complexity:** We analyze the time complexity of one En-PE of ASADI for processing a batch of *n* embeddings, each with dimension d. In the  $\bigcirc$  phase of Figure 17 (b), the analog module generates matrices Q, K, and V for one embedding in one cycle. The latency of writing these matrices to the digital module is also one cycle, allowing the digital module to write matrices of embedding $_{i-1}$  while the analog module processes embedding<sub>i</sub>. As a result, the analog module will take O(n)cycles to process all embeddings and write the output to the digital module. In the **2** phase, the digital module processes the multi-head attention in constant cycle c with high parallel in-situ computing. Each element of matrices Q, K, V, and S are stored and processed in parallel by one ReRAM row. Finally, the  $\mathbf{3}$  phase takes n cycles, similar to the  $\mathbf{1}$  phase. These phases operate sequentially due to data dependencies, and there is no possibility of forming a pipeline. Thus, the overall time complexity of one En-PE is  $\mathbf{O}(n) + c + \mathbf{O}(n)$ .

**Memory complexity:** Suppose the DIA format of S matrix has a diagonal window size  $\omega$ . The analog module requires  $d \times d$  ReRAM capacity to store the weight matrix  $W_Q$ . The metrices  $W_Q$ ,  $W_K$ , and  $W_V$  require  $d \times d \times 3$  ReRAM capacity, which is  $\mathbf{O}(d^2)$  memory complexity. The digital module needs

TABLE I ASADI CONFIGURATIONS

| Component                 | Area (mm <sup>2</sup> ) | Power (mW) | Params.      | Spec.     |
|---------------------------|-------------------------|------------|--------------|-----------|
| Analog module properties  |                         |            |              |           |
| ReRAM                     |                         |            | Bit per Cell | 1         |
| Arrays                    | 0.0013                  | 2.45       | Size         | 64×64     |
| Allays                    |                         |            | Total        | 96        |
| IR                        | 0.0002                  | 0.057      | Size         | 64B       |
| OR                        | 0.0002                  | 0.057      | Size         | 64B       |
| ADC                       | 0.0047                  | 8          | resolution   | 6-bit     |
|                           |                         |            | Total        | 16        |
| DRV                       | 0.0005                  | 6          | resolution   | 1-bit     |
|                           |                         |            | Total        | 64×96     |
| S&A                       | 0.001                   | 0.8        | Total        | 16        |
| AM total                  | 0.0079                  | 18.43      | Size         | 49KB      |
| Digital module properties |                         |            |              |           |
| ReRAM                     | 1.76                    | 3143.6     | Bit per Cell | 1         |
| Arrays                    |                         |            | Size         | 1024×1024 |
| Allays                    |                         |            | Total        | 512       |
| IR                        | 0.0608                  | 20.89      | Size         | 64KB      |
| OR                        | 0.0322                  | 11.47      | Size         | 32KB      |
| DRV                       | 0.0423                  | 506.4      | resolution   | 1-bit     |
|                           |                         |            | Total        | 1024×512  |
| S&A                       | 0.032                   | 25.62      | Total        | 512       |
| DM total                  | 1.927                   | 3708       | Size         | 67.2MB    |
| En-PE properties          |                         |            |              |           |
| AM                        | 0.19                    | 442.32     | Total        | 24        |
|                           |                         |            | Size         | 1176KB    |
| DM                        | 23.12                   | 44.5K      | Total        | 12        |
|                           |                         |            | Size         | 806.6MB   |
| Controller                | 0.0048                  | 7.8        | Total        | 12        |
| En-PE total               | 23.31                   | 44.9K      | Size         | 807.8MB   |
| ASADI properties          |                         |            |              |           |
| En/De-PE                  | 279.8                   | 538.9K     | Total        | 12        |
|                           |                         |            | Size         | 9.7GB     |

memory capacity of  $\mathbf{O}(nd)$  to store matrices Q, K, and V. The digital module requires  $\mathbf{O}(n\omega)$  memory capacity to store matrix S and softmax matrix  $\tilde{S}$ , and  $\mathbf{O}(nk)$  memory capacity to store intermediate results during in-situ computing. Here, k is a constant number. The total memory capacity required for storing all original and intermediate matrices is  $\mathbf{O}(d^2) + \mathbf{O}(nd) + \mathbf{O}(n\omega) + \mathbf{O}(nk)$ . Since both d and k are constants, and  $\omega = \frac{n}{8}$ , the memory complexity of ASADI is  $\mathbf{O}(n) + \mathbf{O}(n\omega) + \mathbf{O}(n)$ . When n exceeds the memory capacity of ASADI, we set  $\omega$  to a constant. In this case, the memory complexity becomes  $\mathbf{O}(n)$ . To support longer sequences, we use this method with some loss of accuracy, and in this work, n is set to a maximum of 8192.

## VI. METHODOLOGY

**Benchmarks.** We evaluate the performance of ASADI using BERT-Base (BERT), BART, GPT-2-Small (GPT2) models for NLP (natural-language processing) tasks, and ViL-Medium-Wide (ViL) model for CV (computer vision) tasks. To achieve dynamic sparsity, we adopt the quantize-and-pruning method of Sanger [22] for all models. For NLP models, we select nine datasets from the *General Language Understanding Evaluation* (GLUE) [36], including cola, mnli, mrpc, qnli,

qqp, rte, sst-2, stsb, and wnli. The *maximal sequence length* (MSL) of all GLUE datasets is less than 384. Additionally, we evaluate the models on MSL 512 *Stanford Question Answering Dataset* (SQuAD v1.1) [27], MSL 1K WikiText-2 [24], and MSL 2K IMDB [23] datasets. For ViL model, we use MSL 1K ImageNet-1K [28] dataset. To measure the efficiency of processing long sequences, we synthesize MSL 4K Syn-4K and MSL 8K Syn-8K by repeating the same sentence of IMDB multiple times to generate longer sentences. Note that Syn-4K and Syn-8K are not used for accuracy evaluation, but only for latency and energy consumption. The data precision used in this paper is Float32, and we limit the maximum length of sequences to 8192. However, ASADI can theoretically process sequences of any length if the memory capacity is sufficient.

Baseline PIM platform. Previous research has demonstrated the significant performance and energy efficiency advantages of PIM-based Transformer accelerators over traditional von Neumann architectures, such as CPU, GPU, FPGA, and ASIC-based architectures, as revealed by numerous studies [18], [41], [43], [47]. TransPIM [47], ReTransformer [41], CPSAA [18], and SPRINT [43] have already explained the reasons why PIM architectures outperform traditional architectures. Therefore, to emphasize the benefits of in-situ computing, we establish a PIM-based baseline and do not compare ASADI with non-PIM traditional architectures. We also note that the differences between PIM and full-flow in-situ computing architectures has not been well-studied in the literature. The PIM baseline employs Samsung's novel function-in-memory DRAM (FIMDRAM) [16], which incorporates programmable computing units in the I/O circuits of the memory banks. We choose the standard configurations of FIMDRAM to support large-scale SpMM with high bandwidth and parallelism. We store the sparse S matrix in CSR format for baseline, which comprises 10GB HBM2 memory and 500 MHz on-chip logic units per bank. We use Ramulator-PIM [14] to obtain the baseline's latency and energy consumption.

**ASADI configurations.** Unlike previous PIM accelerators, ASADI is a full-flow in-situ Transformer accelerator that stores all original and intermediate data in the ReRAM arrays. The full-flow refers to the fact that we calculate all basic operations of Transformer in-situ. As a result, the memory capacity of ASADI is directly related to the input sequence length. We configure the ASADI accelerator to process a maximal sequence length (MSL) of 8192 and diagonal window size  $\omega = \frac{MSL}{8}$  for all datasets, with 12 En-PE and De-PE as shown in Table I. Each Encoder has 12 heads, and each En-PE has 12 Tiles, with each embedding having 64 dimensions for a single head. Both the analog and digital modules use one ReRAM cell to present only 1-bit to ensure accuracy and noise immunity. Thus, the analog module has  $32 \times 3$  $64 \times 64$  ReRAM arrays for Float32  $W_Q$ ,  $W_K$ , and  $W_V$  matrices, while the digital module needs  $64 \times \frac{8192}{1024}$   $1024 \times 1024$  ReRAM arrays for all intermediate matrices. We use 1000GB/s On-Chip Interconnect (OCI) [13] for inner-Encoder transfer and PCIe-6.0 [32] with 128GB/s for cross-Encoder transfer. The details of the ReRAM arrays are described below.

We use 1GHz 1-bit one transistor and one memristor (1T1M) ReRAM arrays for both analog and digital modules. The array-level area and power configurations of the 1T1M ReRAM array are obtained from [12]. ReRAM arrays are read/written column-parallelly, and the SET/RESET voltage for 1-bit ReRAM cell is 1.62/3.63V [11]. To serve as four 6-bit ADCs, we use one 8-bit ADC since the maximum value of 1-bit VMM operation of 64×64 ReRAM array is less than 64 (2<sup>6</sup>). We configure 16 ADC for 96 ReRAM arrays with six arrays sharing one ADC for area saving, using the 8-bit 1.2GS/s single-channel asynchronous SAR ADC from [15]. The area and power of the S&A unit and all on-chip SRAM buffers (IR and OR) are obtained from [31]. The DRV is obtained from 1-bit digital-analog converter (DAC) from [29]. We modify ZSim [30] to simulate the behaviors of ReRAM arrays. We further design an in-house cycle-accurate simulator to obtain the latency and energy consumption of ASADI, following the mathematical proof from [42].

Sisters of ASADI. We design two sister systems to evaluate the software and hardware efficiency of ASADI. To evaluate the hardware efficiency of ASADI, we configure DIA-PIM by using our DIA-wise computation paradigm for our baseline PIM platform. To evaluate the software efficiency of ASADI, we configure CSR-ASADI by using the CSR computation paradigm in Figure 9 (b) and Figure 11 (b) for ReRAM arrays.

Comparison with modern accelerators. We compare ASADI with one GPU platform and two modern sparse attention accelerators that utilize in-memory computing: NVIDIA RTX A6000 with 46GB memory, 300W TDP, CUDA v11.6, and PyTorch v2.0.0 [25], SPRINT [43] and CPSAA [18]. SPRINT prunes weak connections using ReRAM arrays (64KB) while using their ASIC accelerator with 10GB DRAM to process the multi-head attention. CPSAA stores part of the intermediate matrices (K and V) in ReRAM arrays (27.5MB) while storing other intermediate matrices (Q and S) in 10GB DRAM buffer. In contrast, ASADI has only ReRAM arrays of 9.7GB. We configure SPRINT and CPSAA with their algorithms, data flow, and hardware. All comparison platforms have two parts to area, i.e., (i) on-chip logic area and (ii) DRAM area. ASADI has only one area because computation and storage are both in memory (iii). It is unfair to ASADI if we keep (i) and (iii) the same. It's unfair to the comparison platforms if we keep (ii) the same as (iii) because ReRAM has higher memory density. Therefore, we keep the memory capacity the same, i.e., 10GB for all platforms.

**Pre-processing.** We conduct the following pre-processing phases. First, all models are fine-tuned from pre-trained checkpoints with the corresponding training datasets to get the weight matrices. Second, all weight matrices are pre-stored in the ReRAM memory. Finally, we perform the quantize-and-pruning sparse attention to get the sparse mask matrix of all datasets, and the sparse mask matrices are compressed to CSR and DIA format. For GLUE and SQuAD datasets, we set the learning rate and batch size the same as Sanger [22]. We set the learning rate 2e-5 and batch size of one for all other datasets. These pre-processing phases are implemented

![](_page_9_Figure_0.jpeg)

Fig. 18. Performace comparison between ASADI and PIM baseline

on our GPU server. Our code is modified from the GitHub project of Sanger [22]. All models and datasets are obtained from Hugging Face's models library [37] and datasets library.

