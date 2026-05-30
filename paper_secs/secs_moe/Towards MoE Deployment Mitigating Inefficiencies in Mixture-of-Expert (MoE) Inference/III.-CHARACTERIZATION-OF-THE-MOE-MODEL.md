# III. CHARACTERIZATION OF THE MOE MODEL

To characterize the workload of MoE Transformer models, we study two major use cases: Language Modeling and Machine Translation. Language modeling generates the probability an input sequence appears in natural text whereas machine translation maps the input from one language to another. Both tasks are core problems to natural language processing, and are currently major applications of MoE Transformers. We choose models in recent publications that achieved state-of-the-art as our testbed. The details of the datasets and models can be found in Table [I.](#page-3-0)

The MoE model's dense counterparts are selected to be FLOP-equivalent, so they share most of the hyperparameters with the MoE Transformers of interest including hidden dimensions, number of layers and attention heads. The only difference is that the MoE Transformer replaces the FFN layer with an MoE layer every MF layers. Capacity factor C, a parameter unique to the MoE Transformer, controls how many tokens can be processed by a single expert. Under the original design, no matter how many tokens are assigned to an expert, the expert will always process a number of tokens equal to C times the sequence length. When too many tokens are assigned to a single expert, excess tokens are dropped and not

| Task     | Type                                                                                                        | Size          | E            | MF           | CF               |
|----------|-------------------------------------------------------------------------------------------------------------|---------------|--------------|--------------|------------------|
| LM       | Dense<br>MoE                                                                                                | 355M<br>52B   | –<br>512     | –<br>2       | –<br>0.05        |
| MT       | Dense<br>MoE                                                                                                | 3.3B<br>54.5B | –<br>128     | –<br>4       | –<br>1           |
| Task     | Type                                                                                                        | Layers        | TD           | HD           | Vocab            |
| LM       | Dense<br>MoE                                                                                                | 24<br>24      | 1024<br>1024 | 4096<br>4096 | 51200<br>51200   |
| MT       | Dense<br>MoE                                                                                                | 48<br>48      | 2048<br>2048 | 8192<br>8192 | 256206<br>256206 |
| Platform | Specification                                                                                               |               |              |              |                  |
| CPU      | 2×Intel Xeon E5-2698 v4 at 2.2GHz<br>with 700GB memory                                                      |               |              |              |                  |
| CPU-GPU  | 16GB/s via PCIe 3.0                                                                                         |               |              |              |                  |
| GPU      | 8×NVIDIA Tesla V100, with 5120 CUDA<br>cores, 32GB HBM2 memory at 900GB/s<br>connected by NVLink at 300GB/s |               |              |              |                  |
|          |                                                                                                             |               |              |              |                  |

#### TABLE I

<span id="page-3-0"></span>EXPERIMENTAL SETUP. LM: LANGUAGE MODELING. MT: MACHINE TRANSLATION. E: NUMBER OF EXPERTS. MF: MOE LAYER FREQUENCY. CF: CAPACITY FACTOR. TD: TOKEN DIMENSION. HD: HIDDEN DIMENSION. VOCAB: VOCABULARY SIZE. E, MF AND CF DO NOT APPLY TO DENSE MODELS.

processed by any expert. When too few tokens are assigned, unused capacity will be filled by zeros. We utilize the capacity factor settings recommended by [\[2\]](#page-10-0), [\[22\]](#page-11-2). Table [I](#page-3-0) details the experimental setup.

