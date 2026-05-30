# *D. Installation*

*1) Download* Orojenesis *artifacts:* On a user machine, download the archived orojenesis code by running:

```
curl -Ls -w %{url_effective} -o a https://doi.org
    /10.5281/zenodo.10850531 > DL_url
wget $(cat DL_url)/files/orojenesis.zip
unzip orojenesis.zip
```

*2) Install* Orojenesis*:* We provide an installation script install.sh under the orojenesis repository to install Timeloop and other software dependencies.

Before proceeding with the installation, we offer two methods for setting up the system environment:

- Native Host: If you have sudo access on a Debian-based system with Python 3.10 or later installed, we recommend directly executing the installation script.
- Docker: Alternatively, if sudo access is not available, consider using a Docker container. You can find the Dockerfile at orojenesis/docker/Dockerfile. Please refer to orojenesis/README.md for detailed instructions on building and running the Docker container.

Once the system is properly set up, proceed with the following command to install *Orojenesis*:

```
cd orojenesis && ./install.sh
```

This command builds the Timeloop's *Orojenesis* code and adds its path to your TIMELOOP\_BASE\_PATH.

#### *E. Experiment workflow*

We provide Jupyter notebooks, orojenesis /orojenesis\_single.ipynb and orojenesis/ orojenesis\_multi.ipynb, to guide you through generating the key figures in the paper. Please launch the Jupyter GUI under orojenesis by running:

```
cd orojenesis && jupyter notebook
```

Follow the instructions displayed in the terminal output to navigate to the Jupyter interface in your web browser. The notebooks provide instructions and code to generate the *Orojenesis* bounds.

If a GUI is not accessible, you can run the following command to convert the notebooks to Python scripts.

```
jupyter nbconvert --to script <my-notebook.ipynb>
```

Running through the scripts will generate figures in the paper under orojenesis/figs.

#### *F. Evaluation and expected results*

Single-Einsum: Executing the cells in orojenesis/ orojenesis\_single.ipynb produces the following plots in the paper:

- Fig. [1:](#page-0-0) Bound for 16k 1k 1k GEMM.
- Fig. [10:](#page-5-0) Bounds for various GEMM shapes.
- Fig. [11:](#page-5-1) Maximal effectual buffer ratio over total operand size for various GEMMs.
- Fig. [12:](#page-5-2) Bounds for various convolution configurations.
- Fig. [13:](#page-6-1) Bounds for BMMs with different numbers of heads but identical OPs.
- Fig. [14:](#page-6-2) Bounds for Grouped BMMs with different numbers of groups but identical OPs.
- Fig. [24b:](#page-11-1) Validation of *Orojenesis* bounds on Simba accelerator.

Multi-Einsum: Running the cells in orojenesis/ orojenesis\_multi.ipynb generates the following plots:

- Fig. [18:](#page-9-1) Bounds for fusing 32k 4k 16k and 32k 16k 4k GEMMs.
- Fig. [20:](#page-10-1) Bounds for fused MHA.
- Fig. [21:](#page-10-1) Bounds for sliced fusion.
- Fig. [22:](#page-10-1) Bounds for a single fused LLM block.
- Fig. [23:](#page-11-0) Optimal hardware buffer area ratio for LLMs.

#### *G. Experiment customization*

To customize input workload shapes and constraints, please refer to instructions and examples in the Jupyter notebook orojenesis/orojenesis\_example.ipynb.

#### *H. Methodology*

Submission, reviewing and badging methodology:

- [https://www.acm.org/publications/policies/artifact](https://www.acm.org/publications/policies/artifact-review-and-badging-current)[review-and-badging-current](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
- <http://cTuning.org/ae/submission-20201122.html>
- <http://cTuning.org/ae/reviewing-20201122.html>

#### REFERENCES

- <span id="page-14-9"></span>[1] (2022) Snowcat. Accessed on November 14, 2023. [Online]. Available: <https://en.wikipedia.org/wiki/Snowcat>
- <span id="page-14-24"></span>[2] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard *et al.*, "TensorFlow: a system for Large-Scale machine learning," in *USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 2016.
- <span id="page-14-36"></span>[3] A. Aggarwal and S. Vitter, Jeffrey, "The input/output complexity of sorting and related problems," *Communications of the ACM*, vol. 31, no. 9, pp. 1116–1127, 1988.
- <span id="page-14-19"></span>[4] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebron, and ´ S. Sanghai, "Gqa: Training generalized multi-query transformer models from multi-head checkpoints," *arXiv preprint arXiv:2305.13245*, 2023.
- <span id="page-14-28"></span>[5] M. Alwani, H. Chen, M. Ferdman, and P. Milder, "Fused-layer cnn accelerators," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2016, pp. 1–12.
- <span id="page-14-29"></span>[6] A. Azizimazreah and L. Chen, "Shortcut mining: Exploiting cross-layer shortcut reuse in dcnn accelerators," in *Proceedings of the International Symposium on High-Performance Computer Architecture (HPCA)*, 2019, pp. 94–105.
- <span id="page-14-8"></span>[7] L. A. Belady, "A study of replacement algorithms for a virtual-storage computer," *IBM Systems journal*, vol. 5, no. 2, pp. 78–101, 1966.
- <span id="page-14-11"></span>[8] J. Cai, Y. Wei, Z. Wu, S. Peng, and K. Ma, "Inter-layer scheduling space definition and exploration for tiled accelerators," in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2023, pp. 1–17.
- <span id="page-14-37"></span>[9] A. Chen, J. Demmel, G. Dinh, M. Haberle, and O. Holtz, "Communication bounds for convolutional neural networks," in *Proceedings of the Platform for Advanced Scientific Computing Conference*, 2022, pp. 1–10.
- <span id="page-14-25"></span>[10] T. Chen, M. Li, Y. Li, M. Lin, N. Wang, M. Wang, T. Xiao, B. Xu, C. Zhang, and Z. Zhang, "Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems," *arXiv preprint arXiv:1512.01274*, 2015.
- <span id="page-14-26"></span>[11] T. Chen, T. Moreau, Z. Jiang, L. Zheng, E. Yan, H. Shen, M. Cowan, L. Wang, Y. Hu, L. Ceze *et al.*, "TVM: An automated end-to-end optimizing compiler for deep learning," in *USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 2018, pp. 578– 594.
- <span id="page-14-16"></span>[12] Y.-H. Chen, T. Krishna, J. S. Emer, and V. Sze, "Eyeriss: An energyefficient reconfigurable accelerator for deep convolutional neural networks," *IEEE journal of solid-state circuits*, vol. 52, no. 1, pp. 127–138, 2016.
- <span id="page-14-34"></span>[13] Y.-H. Chen, T.-J. Yang, J. Emer, and V. Sze, "Eyeriss v2: A flexible accelerator for emerging deep neural networks on mobile devices," *IEEE Journal on Emerging and Selected Topics in Circuits and Systems*, vol. 9, no. 2, pp. 292–308, 2019.
- <span id="page-14-30"></span>[14] J. Choi, H. Li, B. Kim, S. Hwang, and J. H. Ahn, "Accelerating transformer networks through recomposing softmax layers," in *International Symposium on Workload Characterization (IISWC)*, 2021.
- <span id="page-14-22"></span>[15] T. Dao, "FlashAttention-2: Faster attention with better parallelism and work partitioning," *arXiv preprint arXiv:2307.08691*, 2023.
- <span id="page-14-23"></span>[16] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, "FlashAttention: Fast ´ and memory-efficient exact attention with IO-awareness," in *Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)*, 2022.
- <span id="page-14-0"></span>[17] S. Dave, Y. Kim, S. Avancha, K. Lee, and A. Shrivastava, "Dmazerunner: Executing perfectly nested loops on dataflow accelerators," *ACM Transactions on Embedded Computing Systems (TECS)*, vol. 18, no. 5s, pp. 1–27, 2019.
- <span id="page-14-3"></span>[18] S. Dave, T. Nowatzki, and A. Shrivastava, "Explainable-dse: An agile and explainable exploration of efficient hw/sw codesigns of deep learning accelerators using bottleneck analysis," in *Proceedings of the International Conference on Architectural Support for Programming Languages and Operation Systems (ASPLOS)*, 2023, pp. 87–107.
- <span id="page-14-35"></span>[19] P. J. Denning, "The working set model for program behavior," *Communications of the ACM*, vol. 11, no. 5, pp. 323–333, 1968.
- <span id="page-14-13"></span>[20] A. Einstein *et al.*, "The foundation of the general theory of relativity," *Annalen Phys*, vol. 49, no. 7, pp. 769–822, 1916.
- <span id="page-14-31"></span>[21] M. Gao, X. Yang, J. Pu, M. Horowitz, and C. Kozyrakis, "Tangram: Optimized coarse-grained dataflow for scalable nn accelerators," in *Proceedings of the International Conference on Architectural Support*

- *for Programming Languages and Operation Systems (ASPLOS)*, 2019, pp. 807–820.
- <span id="page-14-12"></span>[22] M. Gilbert, Y. N. Wu, A. Parashar, V. Sze, and J. S. Emer, "Looptree: Enabling exploration of fused-layer dataflow accelerators," in *Proceedings of the International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2023, pp. 316–318.
- <span id="page-14-18"></span>[23] Google. (2022) Tpu v5e. Accessed on November 14, 2023. [Online]. Available: [https://cloud.google.com/tpu/docs/system-architecture-tpu](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)[vm](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- <span id="page-14-14"></span>[24] C. R. Harris, K. J. Millman, S. J. Van Der Walt, R. Gommers, P. Virtanen, D. Cournapeau, E. Wieser, J. Taylor, S. Berg, N. J. Smith *et al.*, "Array programming with numpy," *Nature*, vol. 585, no. 7825, pp. 357–362, 2020.
- <span id="page-14-10"></span>[25] M. Harris, "Mapping computational concepts to gpus," in *ACM SIG-GRAPH 2005 Courses*, 2005, pp. 50–es.
- <span id="page-14-15"></span>[26] K. Hegde, H. Asghari-Moghaddam, M. Pellauer, N. Crago, A. Jaleel, E. Solomonik, J. Emer, and C. W. Fletcher, "ExTensor: An accelerator for sparse tensor algebra," in *International Symposium on Microarchitecture (MICRO)*, Oct. 2019, pp. 319–333. [Online]. Available: <https://doi.org/10.1145/3352460.3358275>
- <span id="page-14-1"></span>[27] K. Hegde, P.-A. Tsai, S. Huang, V. Chandra, A. Parashar, and C. W. Fletcher, "Mind mappings: enabling efficient algorithm-accelerator mapping space search," in *Proceedings of the International Conference on Architectural Support for Programming Languages and Operation Systems (ASPLOS)*, 2021.
- <span id="page-14-6"></span>[28] C. Hong, Q. Huang, G. Dinh, M. Subedar, and Y. S. Shao, "Dosa: Differentiable model-based one-loop search for dnn accelerators," in *Proceedings of the International Symposium on Microarchitecture (MI-CRO)*, 2023.
- <span id="page-14-20"></span>[29] M. Horeni, P. Taheri, P.-A. Tsai, A. Parashar, J. Emer, and S. Joshi, "Ruby: Improving hardware efficiency for tensor algebra accelerators through imperfect factorization," in *Proceedings of the International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2022.
- <span id="page-14-4"></span>[30] Q. Huang, C. Hong, J. Wawrzynek, M. Subedar, and Y. S. Shao, "Learning a continuous and reconstructible latent space for hardware accelerator design," in *Proceedings of the International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2022.
- <span id="page-14-2"></span>[31] Q. Huang, M. Kang, G. Dinh, T. Norell, A. Kalaiah, J. Demmel, J. Wawrzynek, and Y. S. Shao, "Cosa: Scheduling by constrained optimization for spatial accelerators," in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2021, pp. 554–566.
- <span id="page-14-33"></span>[32] G. Jeong, G. Kestor, P. Chatarasi, A. Parashar, P.-A. Tsai, S. Rajamanickam, R. Gioiosa, and T. Krishna, "Union: A unified hw-sw codesign ecosystem in mlir for evaluating tensor operations on spatial accelerators," in *Proceedings of the International Conference on Parallel Architectures and Compilation Techniques (PACT)*, 2021.
- <span id="page-14-38"></span>[33] H. Jia-Wei and H.-T. Kung, "I/o complexity: The red-blue pebble game," in *Proceedings of the thirteenth annual ACM symposium on Theory of computing*, 1981, pp. 326–333.
- <span id="page-14-32"></span>[34] S.-C. Kao, X. Huang, and T. Krishna, "Dnnfuser: Generative pre-trained transformer as a generalized mapper for layer fusion in dnn accelerators," *arXiv preprint arXiv:2201.11218*, 2022.
- <span id="page-14-5"></span>[35] S.-C. Kao, G. Jeong, and T. Krishna, "Confuciux: Autonomous hardware resource assignment for dnn accelerators using reinforcement learning," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2020, pp. 622–636.
- <span id="page-14-17"></span>[36] S.-C. Kao and T. Krishna, "GAMMA: Automating the HW Mapping of DNN Models on Accelerators via Genetic Algorithm," in *Proceedings of the International Conference on Computer-Aided Design (ICCAD)*, 2020.
- <span id="page-14-27"></span>[37] S.-C. Kao, A. Parashar, P.-A. Tsai, and T. Krishna, "Demystifying map space exploration for npus," in *International Symposium on Workload Characterization (IISWC)*, 2022.
- <span id="page-14-7"></span>[38] S.-C. Kao, M. Pellauer, A. Parashar, and T. Krishna, "Digamma: domainaware genetic algorithm for hw-mapping co-optimization for dnn accelerators," in *2022 Design, Automation & Test in Europe Conference & Exhibition (DATE)*. IEEE, 2022, pp. 232–237.
- <span id="page-14-21"></span>[39] S.-C. Kao, S. Subramanian, G. Agrawal, A. Yazdanbakhsh, and T. Krishna, "Flat: An optimized dataflow for mitigating attention bottlenecks," in *Proceedings of the International Conference on Architectural Support for Programming Languages and Operation Systems (ASPLOS)*, 2023, pp. 295–310.

- <span id="page-15-12"></span>[40] F. Kjolstad, S. Kamil, S. Chou, D. Lugato, and S. Amarasinghe, "The tensor algebra compiler," in *Proceedings of the International Conference on Object Oriented Programming Systems Languages and Applications (OOPSLA)*. ACM New York, NY, USA, 2017.
- <span id="page-15-3"></span>[41] A. Kumar, A. Yazdanbakhsh, M. Hashemi, K. Swersky, and S. Levine, "Data-driven offline optimization for architecting hardware accelerators," in *Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)*, 2021.
- <span id="page-15-35"></span>[42] H. Kwon, P. Chatarasi, M. Pellauer, A. Parashar, V. Sarkar, and T. Krishna, "Understanding reuse, performance, and hardware cost of dnn dataflow: A data-centric approach," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2019, pp. 754–768.
- <span id="page-15-1"></span>[43] R. Li, Y. Xu, A. Sukumaran-Rajam, A. Rountev, and P. Sadayappan, "Analytical characterization and design space exploration for optimization of cnns," in *Proceedings of the International Conference on Architectural Support for Programming Languages and Operation Systems (ASPLOS)*, 2021.
- <span id="page-15-31"></span>[44] Z. Li and M. Gao, "Kapla: Pragmatic representation and fast solving of scalable nn accelerator dataflow," *arXiv preprint arXiv:2306.15676*, 2023.
- <span id="page-15-37"></span>[45] R. L. Mattson, J. Gecsei, D. R. Slutz, and I. L. Traiger, "Evaluation techniques for storage hierarchies," *IBM Systems journal*, vol. 9, no. 2, pp. 78–117, 1970.
- <span id="page-15-9"></span>[46] L. Mei, P. Houshmand, V. Jain, S. Giraldo, and M. Verhelst, "Zigzag: Enlarging joint architecture-mapping design space exploration for dnn accelerators," *IEEE Transactions on Computers*, vol. 70, no. 8, 2021.
- <span id="page-15-11"></span>[47] N. Nayak, T. O. Odemuyiwa, S. Ugare, C. Fletcher, M. Pellauer, and J. Emer, "Teaal: A declarative framework for modeling sparse tensor accelerators," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2023.
- <span id="page-15-22"></span>[48] NVIDIA. (2010) Nvidia gf100 graphics processing unit (gpu). Accessed on November 14, 2023. [Online]. Available: [https://videocardz.net/gpu/](https://videocardz.net/gpu/nvidia-gf100) [nvidia-gf100](https://videocardz.net/gpu/nvidia-gf100)
- <span id="page-15-28"></span>[49] NVIDIA. (2018) TensorRT: https://developer.nvidia.com/tensorrt.
- <span id="page-15-27"></span>[50] NVIDIA. (2022) Nvidia ampere architecture. Accessed on November 14, 2023. [Online]. Available: [https://www.nvidia.com/en-us/data](https://www.nvidia.com/en-us/data-center/ampere-architecture/)[center/ampere-architecture/](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- <span id="page-15-20"></span>[51] NVIDIA. (2022) Nvidia h100 tensor core gpu. Accessed on November 14, 2023. [Online]. Available: [https://www.nvidia.com/en-us/data](https://www.nvidia.com/en-us/data-center/h100/)[center/h100/](https://www.nvidia.com/en-us/data-center/h100/)
- <span id="page-15-15"></span>[52] NVIDIA. (2022) Timeloop website. Accessed on November 14, 2023. [Online]. Available: [https://timeloop.csail.mit.edu/examples/full-design](https://timeloop.csail.mit.edu/examples/full-design-examples/eyeriss)[examples/eyeriss](https://timeloop.csail.mit.edu/examples/full-design-examples/eyeriss)
- <span id="page-15-13"></span>[53] T. O. Odemuyiwa, J. S. Emer, and J. D. Owens, "The edge language: Extended general einsums for graph algorithms," *arXiv preprint arXiv:2404.11591*, 2024.
- <span id="page-15-16"></span>[54] A. Olivry, G. Iooss, N. Tollenaere, A. Rountev, P. Sadayappan, and F. Rastello, "Ioopt: Automatic derivation of i/o complexity bounds for affine programs," in *Proceedings of the Conference on Programming Language Design and Implementation (PLDI)*, 2021, pp. 1187–1202.
- <span id="page-15-17"></span>[55] A. Olivry, J. Langou, L.-N. Pouchet, P. Sadayappan, and F. Rastello, "Automated derivation of parametric data movement lower bounds for affine programs," in *Proceedings of the Conference on Programming Language Design and Implementation (PLDI)*, 2020, pp. 808–822.
- <span id="page-15-38"></span>[56] F. Olken, "Efficient methods for calculating the success function of fixed-space replacement policies," Lawrence Berkeley National Lab.(LBNL), Berkeley, CA (United States), Tech. Rep., 1981.
- <span id="page-15-0"></span>[57] A. Parashar, P. Raina, Y. S. Shao, Y.-H. Chen, V. A. Ying, A. Mukkara, R. Venkatesan, B. Khailany, S. W. Keckler, and J. Emer, "Timeloop: A systematic approach to dnn accelerator evaluation," in *Proceedings of the International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2019, pp. 304–315.
- <span id="page-15-29"></span>[58] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga *et al.*, "Pytorch: An imperative style, high-performance deep learning library," *Proceedings of the Conference on Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.
- <span id="page-15-32"></span>[59] S. Pati, S. Aga, N. Jayasena, and M. D. Sinclair, "Demystifying bert: Implications for accelerator design," in *International Symposium on Workload Characterization (IISWC)*, 2021.
- <span id="page-15-24"></span>[60] M. Pellauer, J. Clemons, V. Balaji, N. Crago, A. Jaleel, D. Lee, M. O'Connor, A. Parashar, S. Treichler, P.-A. Tsai *et al.*, "Symphony: Orchestrating sparse and dense tensors with hierarchical heterogeneous

- processing," *ACM Transactions on Computer Systems*, vol. 41, no. 1-4, pp. 1–30, 2023.
- <span id="page-15-7"></span>[61] M. Pellauer, Y. S. Shao, J. Clemons, N. Crago, K. Hegde, R. Venkatesan, S. W. Keckler, C. W. Fletcher, and J. Emer, "Buffets: An efficient and composable storage idiom for explicit decoupled data orchestration," in *Proceedings of the International Conference on Architectural Support for Programming Languages and Operation Systems (ASPLOS)*, 2019, pp. 137–151.
- <span id="page-15-2"></span>[62] E. Russo, M. Palesi, S. Monteleone, D. Patti, G. Ascia, and V. Catania, "Medea: A multi-objective evolutionary approach to dnn hardware mapping," in *Design, Automation & Test in Europe Conference & Exhibition (DATE)*, 2022.
- <span id="page-15-30"></span>[63] A. Sabne, "Xla: Compiling machine learning for peak performance," 2020.
- <span id="page-15-4"></span>[64] C. Sakhuja, Z. Shi, and C. Lin, "Leveraging domain information for the efficient automated design of deep learning accelerators," in *Proceedings of the International Symposium on High-Performance Computer Architecture (HPCA)*, 2023.
- <span id="page-15-33"></span>[65] K. Sankaralingam, T. Nowatzki, V. Gangadhar, P. Shah, M. Davies, W. Galliher, Z. Guo, J. Khare, D. Vijay, P. Palamuttam *et al.*, "The mozart reuse exposed dataflow processor for ai and beyond," in *Proc. Int. Symp. Computer Architecture*, 2022.
- <span id="page-15-25"></span>[66] Y. S. Shao, J. Clemons, R. Venkatesan, B. Zimmer, M. Fojtik, N. Jiang, B. Keller, A. Klinefelter, N. Pinckney, P. Raina *et al.*, "Simba: Scaling deep-learning inference with multi-chip-module-based architecture," in *Proceedings of the International Symposium on Microarchitecture (MI-CRO)*, 2019, pp. 14–27.
- <span id="page-15-21"></span>[67] N. Shazeer, "Fast transformer decoding: One write-head is all you need," *arXiv preprint arXiv:1911.02150*, 2019.
- <span id="page-15-5"></span>[68] Z. Shi, C. Sakhuja, M. Hashemi, K. Swersky, and C. Lin, "Using bayesian optimization for hardware/software co-design of neural accelerators," in *Workshop on ML for Systems at the Conference on Neural Information Processing Systems (NeurIPS)*, 2020.
- <span id="page-15-39"></span>[69] T. M. Smith, B. Lowery, J. Langou, and R. A. van de Geijn, "A tight i/o lower bound for matrix multiplication," *arXiv preprint arXiv:1702.02017*, 2017.
- <span id="page-15-36"></span>[70] V. Sze, Y.-H. Chen, T.-J. Yang, and J. S. Emer, "Efficient processing of deep neural networks: A tutorial and survey," *Proceedings of the IEEE*, vol. 105, no. 12, pp. 2295–2329, 2017.
- <span id="page-15-14"></span>[71] V. Sze, Y.-H. Chen, T.-J. Yang, and J. S. Emer, *Efficient processing of deep neural networks*. Springer, 2020.
- <span id="page-15-26"></span>[72] V. Thakkar, P. Ramani, C. Cecka, A. Shivam, H. Lu, E. Yan, J. Kosaian, M. Hoemmen, H. Wu, A. Kerr, M. Nicely, D. Merrill, D. Blasig, F. Qiao, P. Majcher, P. Springer, M. Hohnerbach, J. Wang, and M. Gupta. (2023, Jan.) CUTLASS. [Online]. Available: <https://github.com/NVIDIA/cutlass>
- <span id="page-15-34"></span>[73] E. Valpreda, P. Mor`ı, N. Fasfous, M. R. Vemparala, A. Frickenstein, L. Frickenstein, W. Stechele, C. Passerone, G. Masera, and M. Martina, "Hw-flow-fusion: Inter-layer scheduling for convolutional neural network accelerators with dataflow architectures," *Electronics*, vol. 11, no. 18, p. 2933, 2022.
- <span id="page-15-19"></span>[74] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," *Advances in neural information processing systems*, vol. 30, 2017.
- <span id="page-15-6"></span>[75] R. Venkatesan, Y. S. Shao, M. Wang, J. Clemons, S. Dai, M. Fojtik, B. Keller, A. Klinefelter, N. Pinckney, P. Raina, Y. Zhang, B. Zimmer, W. J. Dally, J. Emer, S. W. Keckler, and B. Khailany, "Magnet: A modular accelerator generator for neural networks," in *Proceedings of the International Conference on Computer-Aided Design (ICCAD)*, 2019.
- <span id="page-15-8"></span>[76] S. Williams, A. Waterman, and D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65–76, 2009.
- <span id="page-15-18"></span>[77] J. Won, C. Mendis, J. S. Emer, and S. Amarasinghe, "Waco: learning workload-aware co-optimization of the format and schedule of a sparse tensor program," in *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2*, 2023, pp. 920–934.
- <span id="page-15-23"></span>[78] Y. N. Wu, J. S. Emer, and V. Sze, "Accelergy: An architecture-level energy estimation methodology for accelerator designs," in *Proceedings of the International Conference on Computer-Aided Design (ICCAD)*, 2019.
- <span id="page-15-10"></span>[79] Y. N. Wu, P.-A. Tsai, A. Parashar, V. Sze, and J. S. Emer, "Sparseloop: An analytical, energy-focused design space exploration methodology

- for sparse tensor accelerators," in *Proceedings of the International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2021, pp. 232–234.
- <span id="page-16-5"></span>[80] Y. N. Wu, P.-A. Tsai, A. Parashar, V. Sze, and J. S. Emer, "Sparseloop: An analytical approach to sparse tensor accelerator modeling," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2022, pp. 1377–1395.
- <span id="page-16-2"></span>[81] Q. Xiao, S. Zheng, B. Wu, P. Xu, X. Qian, and Y. Liang, "Hasco: Towards agile hardware and software co-design for tensor computation," in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2021.
- <span id="page-16-0"></span>[82] X. Yang, M. Gao, Q. Liu, J. Setter, J. Pu, A. Nayak, S. Bell, K. Cao, H. Ha, P. Raina *et al.*, "Interstellar: Using halide's scheduling language to analyze dnn accelerators," in *Proceedings of the International Conference on Architectural Support for Programming Languages and Operation Systems (ASPLOS)*, 2020, pp. 369–383.
- [83] A. Yazdanbakhsh, C. Angermueller, B. Akin, Y. Zhou, A. Jones, M. Hashemi, K. Swersky, S. Chatterjee, R. Narayanaswami, and J. Laudon, "Apollo: Transferable architecture exploration," *arXiv preprint arXiv:2102.01723*, 2021.
- <span id="page-16-3"></span>[84] D. Zhang, S. Huda, E. Songhori, K. Prabhu, Q. Le, A. Goldie, and A. Mirhoseini, "A full-stack search technique for domain optimized deep learning accelerators," in *Proceedings of the International Conference on Architectural Support for Programming Languages and Operation Systems (ASPLOS)*, 2022.
- <span id="page-16-1"></span>[85] S. Zheng, R. Chen, A. Wei, Y. Jin, Q. Han, L. Lu, B. Wu, X. Li, S. Yan, and Y. Liang, "Amos: enabling automatic mapping for tensor computations on spatial accelerators with hardware abstraction." in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2022.
- <span id="page-16-4"></span>[86] S. Zheng, S. Chen, S. Gao, L. Jia, G. Sun, R. Wang, and Y. Liang, "Tileflow: A framework for modeling fusion dataflow via tree-based analysis," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2023.