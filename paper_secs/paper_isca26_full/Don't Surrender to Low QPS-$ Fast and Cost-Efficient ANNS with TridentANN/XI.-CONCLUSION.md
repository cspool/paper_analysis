# XI. CONCLUSION

We propose TRIDENTANN, a high-performance yet lowcost system designed for billion-scale ANNS. We separate noise and clusters in the index structure and introduce the P2P GPU-SSD architecture to ANNS. By integrating multiple high-bandwidth SSDs with low-end GPUs and avoiding expensive GPU device memory, TRIDENTANN operates with costfriendly hardware. Coupled with parallel pipelines, TRIDEN-TANN achieves state-of-the-art performance and the highest cost effectiveness on mature commercial devices.

## ACKNOWLEDGMENT

We sincerely thank anonymous reviewers for their valuable feedback and guidance. This work was supported by the National Natural Science Foundation of China (Grant No.62272171). Chuliang Weng is the corresponding author.

## REFERENCES

- [1] "NYTimes dataset," 2008, https://archive.ics.uci.edu/da taset/164/bag+of+words.
- [2] "Basic linear algebra on nvidia gpus." NVIDIA Corporation, 2014, https://developer.nvidia.com/cublas.
- [3] "Storage Performance Development Kit (SPDK)." Intel, 2014, https://spdk.io/.
- [4] "Faiss: A library for efficient similarity search." Facebook, 2017, https://engineering.fb.com/2017/03/29/datainfrastructure/faiss-a-library-for-efficient-similaritysearch/.
- [5] "AVX used in SPTAG," 2018, https://github.com/micro soft/SPTAG/blob/main/AnnService/inc/Core/Common/ DistanceUtils.h.
- [6] "Distances in faiss wiki." Facebook Research, 2018, https://github.com/facebookresearch/faiss/wiki/MetricT ype-and-distances.
- [7] "NVIDIA Tesla V100-32GB GPU." NVIDIA Corporation, 2018, https://www.nvidia.com/en- gb/datacenter/tesla-v100.
- [8] "SPTAG: A library for fast approximate nearest neighbor search." Microsoft, 2018, https://github.com/microsoft /SPTAG.
- [9] "Faiss: Refine." Facebook, 2020, https://github.com/fac ebookresearch/faiss/wiki/Pre--and-post-processing.
- [10] "NVIDIA RTX A2000-6GB GPU." NVIDIA Corporation, 2020, https://www.nvidia.com/en-us/products/work stations/rtx-a2000.
- [11] "NVIDIA RTX A6000-48GB GPU." NVIDIA Corporation, 2020, https://www.nvidia.com/en-us/products/wo rkstations/rtx-a6000.
- [12] "SPACEV1B." Microsoft, 2020, https://github.com/mic rosoft/SPTAG.
- [13] "AMD EPYC 7453 CPU." AMD, 2021, https://ww w.amd.com/en/products/processors/server/epyc/7003 series/amd-epyc-7453.html.
- [14] "BIGANN Benchmarks," 2021, https://big-ann-benchm arks.com/neurips21.html.
- [15] "Elasticsearch: Open source, distributed, restful search engine." Elastic N.V., 2021, https://github.com/elastic /elasticsearch.
- [16] "Research talk: Approximate nearest neighbor search systems at scale." Microsoft Research, 2021, https: //www.youtube.com/watch?v=BnYNdSIKibQ&list=PLD 7HFcN7LXReJTWFKYqwMcCc1nZKIXBo9&index=9.
- [17] "NVIDIA Magnum IO." NVIDIA Corporation, 2022, https://www.nvidia.com/en-us/data-center/magnum-io/.
- [18] "Redis as a vector database quick start guide." Redis Ltd., 2022, https://redis.io/docs/latest/develop/get-starte d/vector-database.
- [19] "SAMSUNG 980 Pro PCIe 4.0 NVMe SSD 1TB." SAMSUNG Corporation, 2022, https://www.samsung. com/us/memory-storage/nvme-ssd/980-pro-pcie-4-0 nvme-ssd-1tb-sku-mz-v8p1t0b-am/.
- [20] "Hardware requirements of BaM." NVIDIA, 2023, ht

- tps://github.com/ZaidQureshi/bam#hardwaresystemrequirements.
- [21] "He, Who Can Pay Top Dollar For HBM Memory Controls AI Training." The Next Platform, 2024, https://www.nextplatform.com/2024/02/27/he-who-canpay-top-dollar-for-hbm-memory-controls-ai-training/.
- [22] "NVIDIA HGX H20 GPU." NVIDIA Corporation, 2024, https://viperatech.com/product/nvidia-hgx-h20.
- [23] "Scaling Semantic Search with FAISS: Challenges and Solutions for Billion-Scale Datasets." Medium, 2024, https://medium.com/@deveshbajaj59/scaling-semanticsearch-with-faiss-challenges-and-solutions-for-billionscale-datasets-1cacb6f87f95.
- [24] "AMD EPYC Server Processor." Advanced Micro Devices, Inc., 2025, https://www.amd.com/en/product s/specifications/server-processor.html.
- [25] "INTEL Xeon Server Processor." Intel Corporation, 2025, https://www.intel.com/content/www/us/en/prod ucts/details/processors/xeon.html.
- [26] "NVIDIA RTX PRO 5000 Blackwell GPU." NVIDIA Corporation, 2025, https://www.nvidia.com/en-us/pro ducts/workstations/professional-desktop-gpus/rtx-pro-5000/.
- [27] "NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU." NVIDIA Corporation, 2025, https://www. nvidia.com/rtx-pro-6000/.
- [28] L. Amsaleg, O. Chelly, T. Furon, S. Girard, M. E. Houle, K.-i. Kawarabayashi, and M. Nett, "Estimating local intrinsic dimensionality," in *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, ser. KDD, 2015.
- [29] I. Azizi, K. Echihabi, and T. Palpanas, "ELPIS: Graph-Based Similarity Search for Scalable Data Science," in *Proceedings of the VLDB Endowment, Volume 16, Issue 6*, ser. VLDB, 2023.
- [30] I. Azizi, K. Echihabi, and T. Palpanas, "Graph-Based Vector Search: An Experimental Evaluation of the Stateof-the-Art," in *Proceedings of the ACM on Management of Data, Volume 3, Issue 1*, ser. SIGMOD, 2025.
- [31] K. Chen, R. Nadig, M. Frouzakis, N. M. Ghiasi, Y. Liang, H. Mao, J. Park, M. Sadrosadati, and O. Mutlu, "REIS: A High-Performance and Energy-Efficient Retrieval System with In-Storage Processing," in *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, ser. ISCA, 2025.
- [32] Q. Chen, B. Zhao, H. Wang, M. Li, C. Liu, Z. Li, M. Yang, and J. Wang, "SPANN: highly-efficient billionscale approximate nearest neighbor search," in *Proceedings of the 35th International Conference on Neural Information Processing Systems*, ser. NeurIPS, 2021.
- [33] R. Chen, B. Liu, H. Zhu, Y. Wang, Q. Li, B. Ma, Q. Hua, J. Jiang, Y. Xu, H. Deng, and B. Zheng, "Approximate nearest neighbor search under neural similarity metric for large-scale recommendation," in *Proceedings of the 31st ACM International Conference on Information & Knowledge Management*, ser. CIKM, 2022.

- [34] Y. Chen, W. Ruys, and G. Biros, "KNN-DBSCAN: a DBSCAN in high dimensions," in *ACM Trans. Parallel Comput.*, 2025.
- [35] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, "A densitybased algorithm for discovering clusters in large spatial databases with noise," in *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining*, ser. KDD.
- [36] C. Fu, C. Xiang, C. Wang, and D. Cai, "Fast approximate nearest neighbor search with the navigating spreading-out graph," in *Proceedings of the VLDB Endowment, Volume 12, Issue 5*, ser. VLDB, 2019.
- [37] J. Gan and Y. Tao, "DBSCAN Revisited: Mis-Claim, Un-Fixability, and Approximation," in *Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data*, ser. SIGMOD, 2015.
- [38] J. Gao and C. Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search," in *Proceedings of the ACM on Management of Data, Volume 2, Issue 3*, ser. SIGMOD, 2024.
- [39] F. Groh, L. Ruppert, P. Wieschollek, and H. P. A. Lensch, "GGNN: Graph-Based GPU Nearest Neighbor Search," in *IEEE Transactions on Big Data*, 2023.
- [40] H. Guo and Y. Lu, "Achieving Low-Latency Graph-Based Vector Search via Aligning Best-First Search Algorithm with SSD," in *19th USENIX Symposium on Operating Systems Design and Implementation*, ser. OSDI, 2025.
- [41] M. Ha, E. Kim, and H. Kim, "H3: Hybrid Architecture Using High Bandwidth Memory and High Bandwidth Flash for Cost-Efficient LLM Inference," in *IEEE Computer Architecture Letters*, 2026.
- [42] G. Haas and V. Leis, "What Modern NVMe Storage Can Do, and How to Exploit it: High-Performance I/O for High-Performance Storage Engines," in *Proceedings of the VLDB Endowment, Volume 16, Issue 9*, ser. VLDB, 2023.
- [43] M. E. Houle, "Local intrinsic dimensionality i: An extreme-value-theoretic foundation for similarity applications," in *Similarity Search and Applications*, 2017.
- [44] Y. Huang, X. Fan, S. Yan, and C. Weng, "Neos: A NVMe-GPUs Direct Vector Service Buffer in User Space," in *2024 IEEE 40th International Conference on Data Engineering*, ser. ICDE, 2024.
- [45] J. Jang, H. Choi, H. Bae, S. Lee, M. Kwon, and M. Jung, "CXL-ANNS: Software-Hardware collaborative memory disaggregation and computation for Billion-Scale approximate nearest neighbor search," in *2023 USENIX Annual Technical Conference*, ser. ATC, 2023.
- [46] Y. Jun, S. Park, J.-U. Kang, S.-H. Kim, and E. Seo, "We ain't afraid of no file fragmentation: Causes and prevention of its performance impact on modern flash SSDs," in *22nd USENIX Conference on File and Storage Technologies*, ser. FAST, 2024.
- [47] H. Jegou, M. Douze, and C. Schmid, "Product quantiza- ´

- tion for nearest neighbor search," in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2011.
- [48] M. Kuschewski, J. Giceva, T. Neumann, and V. Leis, "High-Performance Query Processing with NVMe Arrays: Spilling without Killing Performance," in *Proceedings of the ACM on Management of Data, Volume 2, Issue 6*, ser. SIGMOD, 2024.
- [49] S. Li, J. Lin, F. Tu, Z. Wang, L. Liu, Y. Kang, Y. Ding, and Y. Xie, "ECSSD: Hardware/Data Layout Co-Designed In-Storage-Computing Architecture for Extreme Classification," in *Proceedings of the 50th Annual International Symposium on Computer Architecture*, ser. ISCA, 2023, pp. 814–827.
- [50] Y. Li, Y. Jin, B. Tian, H. Zhang, and M. Gao, "ANS-MET: Approximate Nearest Neighbor Search with Near-Memory Processing and Hybrid Early Termination," in *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, ser. ISCA, 2025.
- [51] D. G. Lowe, "Distinctive image features from scaleinvariant keypoints," in *International Journal of Computer Vision*, 2004.
- [52] Y. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs," in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2020.
- [53] M. D. Manohar, Z. Shen, G. Blelloch, L. Dhulipala, Y. Gu, H. V. Simhadri, and Y. Sun, "ParlayANN: Scalable and Deterministic Parallel Graph-Based Approximate Nearest Neighbor Search Algorithms," in *Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming*, ser. PPoPP, 2024.
- [54] J. Mohoney, D. Sarda, M. Tang, S. R. Chowdhury, A. Pacaci, I. F. Ilyas, T. Rekatsinas, and S. Venkataraman, "Quake: Adaptive Indexing for Vector Search," in *19th USENIX Symposium on Operating Systems Design and Implementation*, ser. OSDI, 2025.
- [55] H. Ootomo, A. Naruse, C. Nolet, R. Wang, T. Feher, and Y. Wang, "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs," in *2024 IEEE 40th International Conference on Data Engineering*, ser. ICDE, 2024.
- [56] J. J. Pan, J. Wang, and G. Li, "Survey of vector database management systems," in *The VLDB booktitle*, 2024.
- [57] J. Pennington, R. Socher, and C. Manning, "GloVe: Global vectors for word representation," in *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, ser. EMNLP, 2014.
- [58] A. Prokopenko, D. Lebrun-Grandie, and D. Arndt, "Fast tree-based algorithms for DBSCAN for low-dimensional data on GPUs," in *Proceedings of the 52nd International Conference on Parallel Processing*, ser. ICPP, 2023.
- [59] S. Qiu, W. Liu, Y. Hu, J. Yan, Z. Shen, X. Yao, R. Chen, G. Zhang, and Y. Zhang, "GeminiFS: A companion file system for GPUs," in *23rd USENIX Conference on File and Storage Technologies*, ser. FAST, 2025.

- [60] D. Quinn, M. Nouri, N. Patel, J. Salihu, A. Salemi, S. Lee, H. Zamani, and M. Alian, "Accelerating retrievalaugmented generation," in *Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1*, ser. ASPLOS, 2025.
- [61] Z. Qureshi, V. S. Mailthody, I. Gelado, S. Min, A. Masood, J. Park, J. Xiong, C. J. Newburn, D. Vainbrand, I.- H. Chung, M. Garland, W. Dally, and W.-m. Hwu, "Gpuinitiated on-demand high-throughput storage access in the bam system architecture," in *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2*, ser. ASPLOS, 2023.
- [62] J. Ren, M. Zhang, and D. Li, "HM-ANN: efficient billion-point nearest neighbor search on heterogeneous memory," in *Proceedings of the 34th International Conference on Neural Information Processing Systems*, ser. NeurIPS, 2020.
- [63] E. Schubert, J. Sander, M. Ester, H. P. Kriegel, and X. Xu, "DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN," in *ACM Trans. Database Syst.*, 2017.
- [64] B. Sim, Y. Kim, M. Kim, Y. Park, and J. W. Lee, "Instanns: Scalable approximate nearest neighbor search via cost-efficient in-storage processing," in *Proceedings of the 34th ACM International Conference on Information and Knowledge Management*, ser. CIKM, 2025.
- [65] Z. Song, J. Zhang, J. Sun, M. Sun, Z. Yang, Z. Zhang, X. Chen, F. Wu, H. Tang, and Z. Wang, "CAM: Asynchronous GPU-Initiated, CPU-Managed SSD Management for Batching Storage Access," in *IEEE 41st International Conference on Data Engineering*, ser. ICDE, 2025.
- [66] S. J. Subramanya, Devvrit, R. Kadekodi, R. Krishaswamy, and H. V. Simhadri, "DiskANN: fast accurate billion-point nearest neighbor search on a single node," in *Proceedings of the 33rd International Conference on Neural Information Processing Systems*, ser. NeurIPS, 2019.
- [67] J. Sun, G. Li, J. Pan, J. Wang, Y. Xie, R. Liu, and W. Nie, "GaussDB-Vector: A Large-Scale Persistent Real-Time Vector Database for LLM Applications," in *Proceedings of the VLDB Endowment, Volume 18, Issue 12*, ser. VLDB, 2025.
- [68] B. Tian, H. Liu, Z. Duan, X. Liao, H. Jin, and Y. Zhang, "Scalable billion-point approximate nearest neighbor search using SmartSSDs," in *2024 USENIX Annual Technical Conference*, ser. ATC, 2024.
- [69] B. Tian, H. Liu, Y. Tang, S. Xiao, Z. Duan, X. Liao, H. Jin, X. Zhang, J. Zhu, and Y. Zhang, "Towards Highthroughput and Low-latency Billion-scale Vector Search via CPU/GPU Collaborative Filtering and Re-ranking," in *23rd USENIX Conference on File and Storage Technologies*, ser. FAST, 2025.
- [70] G. T. Toussaint, "The relative neighbourhood graph of a

- finite planar set," in *Pattern Recognition*, 1980.
- [71] M. Wang, W. Xu, X. Yi, S. Wu, Z. Peng, X. Ke, Y. Gao, X. Xu, R. Guo, and C. Xie, "Starling: An i/oefficient disk-resident graph index framework for highdimensional vector similarity search on data segment," in *Proc. ACM Manag. Data*, ser. SIGMOD, 2024.
- [72] M. Wang, X. Xu, Q. Yue, and Y. Wang, "A comprehensive survey and experimental comparison of graph-based approximate nearest neighbor search," in *Proceedings of the VLDB Endowment, Volume 14, Issue 11*, ser. VLDB, 2021.
- [73] Y. Wang, S. Li, Q. Zheng, L. Song, Z. Li, A. Chang, H. H. Li, and Y. Chen, "NDSearch: Accelerating Graph-Traversal-Based Approximate Nearest Neighbor Search through Near Data Processing," in *Proceedings of the 51st Annual International Symposium on Computer Architecture*, ser. ISCA, 2024.
- [74] F. F. Xu, U. Alon, and G. Neubig, "Why do nearest neighbor language models work?" in *Proceedings of the 40th International Conference on Machine Learning*, ser. ICML, 2023.
- [75] Y. Xu, H. Liang, J. Li, S. Xu, Q. Chen, Q. Zhang, C. Li, Z. Yang, F. Yang, Y. Yang, P. Cheng, and M. Yang, "SPFresh: Incremental In-Place Update for Billion-Scale Vector Search," in *Proceedings of the 29th Symposium on Operating Systems Principles*, ser. SOSP, 2023.
- [76] J. Zhang, A. Naruse, X. Li, and Y. Wang, "Parallel topk algorithms on gpu: A comprehensive study and new methods," in *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, ser. SC, 2023.
- [77] Z. Zhang, F. Liu, G. Huang, X. Liu, and X. Jin, "Fast vector query processing for large datasets beyond GPU memory with reordered pipelining," in *21st USENIX Symposium on Networked Systems Design and Implementation*, ser. NSDI, 2024.
- [78] W. Zhao, S. Tan, and P. Li, "SONG: Approximate Nearest Neighbor Search on GPU," in *2020 IEEE 36th International Conference on Data Engineering*, ser. ICDE, 2020.
- [79] X. Zhong, H. Li, J. Jin, M. Yang, D. Chu, X. Wang, Z. Shen, W. Jia, G. Gu, Y. Xie, X. Lin, H. T. Shen, J. Song, and P. Cheng, "VSAG: An Optimized Search Framework for Graph-based Approximate Nearest Neighbor Search," in *Proceedings of the VLDB Endowment, Volume 18, Issue 12*, ser. VLDB, 2025.
- [80] C. Zou and A. A. Chien, "ASSASIN: Architecture Support for Stream Computing to Accelerate Computational Storage," in *Proceedings of the 55th Annual IEEE/ACM International Symposium on Microarchitecture*, ser. MI-CRO, 2022.# XI. CONCLUSION

We propose TRIDENTANN, a high-performance yet lowcost system designed for billion-scale ANNS. We separate noise and clusters in the index structure and introduce the P2P GPU-SSD architecture to ANNS. By integrating multiple high-bandwidth SSDs with low-end GPUs and avoiding expensive GPU device memory, TRIDENTANN operates with costfriendly hardware. Coupled with parallel pipelines, TRIDEN-TANN achieves state-of-the-art performance and the highest cost effectiveness on mature commercial devices.

## ACKNOWLEDGMENT

We sincerely thank anonymous reviewers for their valuable feedback and guidance. This work was supported by the National Natural Science Foundation of China (Grant No.62272171). Chuliang Weng is the corresponding author.

## REFERENCES

- [1] "NYTimes dataset," 2008, https://archive.ics.uci.edu/da taset/164/bag+of+words.
- [2] "Basic linear algebra on nvidia gpus." NVIDIA Corporation, 2014, https://developer.nvidia.com/cublas.
- [3] "Storage Performance Development Kit (SPDK)." Intel, 2014, https://spdk.io/.
- [4] "Faiss: A library for efficient similarity search." Facebook, 2017, https://engineering.fb.com/2017/03/29/datainfrastructure/faiss-a-library-for-efficient-similaritysearch/.
- [5] "AVX used in SPTAG," 2018, https://github.com/micro soft/SPTAG/blob/main/AnnService/inc/Core/Common/ DistanceUtils.h.
- [6] "Distances in faiss wiki." Facebook Research, 2018, https://github.com/facebookresearch/faiss/wiki/MetricT ype-and-distances.
- [7] "NVIDIA Tesla V100-32GB GPU." NVIDIA Corporation, 2018, https://www.nvidia.com/en- gb/datacenter/tesla-v100.
- [8] "SPTAG: A library for fast approximate nearest neighbor search." Microsoft, 2018, https://github.com/microsoft /SPTAG.
- [9] "Faiss: Refine." Facebook, 2020, https://github.com/fac ebookresearch/faiss/wiki/Pre--and-post-processing.
- [10] "NVIDIA RTX A2000-6GB GPU." NVIDIA Corporation, 2020, https://www.nvidia.com/en-us/products/work stations/rtx-a2000.
- [11] "NVIDIA RTX A6000-48GB GPU." NVIDIA Corporation, 2020, https://www.nvidia.com/en-us/products/wo rkstations/rtx-a6000.
- [12] "SPACEV1B." Microsoft, 2020, https://github.com/mic rosoft/SPTAG.
- [13] "AMD EPYC 7453 CPU." AMD, 2021, https://ww w.amd.com/en/products/processors/server/epyc/7003 series/amd-epyc-7453.html.
- [14] "BIGANN Benchmarks," 2021, https://big-ann-benchm arks.com/neurips21.html.
- [15] "Elasticsearch: Open source, distributed, restful search engine." Elastic N.V., 2021, https://github.com/elastic /elasticsearch.
- [16] "Research talk: Approximate nearest neighbor search systems at scale." Microsoft Research, 2021, https: //www.youtube.com/watch?v=BnYNdSIKibQ&list=PLD 7HFcN7LXReJTWFKYqwMcCc1nZKIXBo9&index=9.
- [17] "NVIDIA Magnum IO." NVIDIA Corporation, 2022, https://www.nvidia.com/en-us/data-center/magnum-io/.
- [18] "Redis as a vector database quick start guide." Redis Ltd., 2022, https://redis.io/docs/latest/develop/get-starte d/vector-database.
- [19] "SAMSUNG 980 Pro PCIe 4.0 NVMe SSD 1TB." SAMSUNG Corporation, 2022, https://www.samsung. com/us/memory-storage/nvme-ssd/980-pro-pcie-4-0 nvme-ssd-1tb-sku-mz-v8p1t0b-am/.
- [20] "Hardware requirements of BaM." NVIDIA, 2023, ht

- tps://github.com/ZaidQureshi/bam#hardwaresystemrequirements.
- [21] "He, Who Can Pay Top Dollar For HBM Memory Controls AI Training." The Next Platform, 2024, https://www.nextplatform.com/2024/02/27/he-who-canpay-top-dollar-for-hbm-memory-controls-ai-training/.
- [22] "NVIDIA HGX H20 GPU." NVIDIA Corporation, 2024, https://viperatech.com/product/nvidia-hgx-h20.
- [23] "Scaling Semantic Search with FAISS: Challenges and Solutions for Billion-Scale Datasets." Medium, 2024, https://medium.com/@deveshbajaj59/scaling-semanticsearch-with-faiss-challenges-and-solutions-for-billionscale-datasets-1cacb6f87f95.
- [24] "AMD EPYC Server Processor." Advanced Micro Devices, Inc., 2025, https://www.amd.com/en/product s/specifications/server-processor.html.
- [25] "INTEL Xeon Server Processor." Intel Corporation, 2025, https://www.intel.com/content/www/us/en/prod ucts/details/processors/xeon.html.
- [26] "NVIDIA RTX PRO 5000 Blackwell GPU." NVIDIA Corporation, 2025, https://www.nvidia.com/en-us/pro ducts/workstations/professional-desktop-gpus/rtx-pro-5000/.
- [27] "NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU." NVIDIA Corporation, 2025, https://www. nvidia.com/rtx-pro-6000/.
- [28] L. Amsaleg, O. Chelly, T. Furon, S. Girard, M. E. Houle, K.-i. Kawarabayashi, and M. Nett, "Estimating local intrinsic dimensionality," in *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, ser. KDD, 2015.
- [29] I. Azizi, K. Echihabi, and T. Palpanas, "ELPIS: Graph-Based Similarity Search for Scalable Data Science," in *Proceedings of the VLDB Endowment, Volume 16, Issue 6*, ser. VLDB, 2023.
- [30] I. Azizi, K. Echihabi, and T. Palpanas, "Graph-Based Vector Search: An Experimental Evaluation of the Stateof-the-Art," in *Proceedings of the ACM on Management of Data, Volume 3, Issue 1*, ser. SIGMOD, 2025.
- [31] K. Chen, R. Nadig, M. Frouzakis, N. M. Ghiasi, Y. Liang, H. Mao, J. Park, M. Sadrosadati, and O. Mutlu, "REIS: A High-Performance and Energy-Efficient Retrieval System with In-Storage Processing," in *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, ser. ISCA, 2025.
- [32] Q. Chen, B. Zhao, H. Wang, M. Li, C. Liu, Z. Li, M. Yang, and J. Wang, "SPANN: highly-efficient billionscale approximate nearest neighbor search," in *Proceedings of the 35th International Conference on Neural Information Processing Systems*, ser. NeurIPS, 2021.
- [33] R. Chen, B. Liu, H. Zhu, Y. Wang, Q. Li, B. Ma, Q. Hua, J. Jiang, Y. Xu, H. Deng, and B. Zheng, "Approximate nearest neighbor search under neural similarity metric for large-scale recommendation," in *Proceedings of the 31st ACM International Conference on Information & Knowledge Management*, ser. CIKM, 2022.

- [34] Y. Chen, W. Ruys, and G. Biros, "KNN-DBSCAN: a DBSCAN in high dimensions," in *ACM Trans. Parallel Comput.*, 2025.
- [35] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, "A densitybased algorithm for discovering clusters in large spatial databases with noise," in *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining*, ser. KDD.
- [36] C. Fu, C. Xiang, C. Wang, and D. Cai, "Fast approximate nearest neighbor search with the navigating spreading-out graph," in *Proceedings of the VLDB Endowment, Volume 12, Issue 5*, ser. VLDB, 2019.
- [37] J. Gan and Y. Tao, "DBSCAN Revisited: Mis-Claim, Un-Fixability, and Approximation," in *Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data*, ser. SIGMOD, 2015.
- [38] J. Gao and C. Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search," in *Proceedings of the ACM on Management of Data, Volume 2, Issue 3*, ser. SIGMOD, 2024.
- [39] F. Groh, L. Ruppert, P. Wieschollek, and H. P. A. Lensch, "GGNN: Graph-Based GPU Nearest Neighbor Search," in *IEEE Transactions on Big Data*, 2023.
- [40] H. Guo and Y. Lu, "Achieving Low-Latency Graph-Based Vector Search via Aligning Best-First Search Algorithm with SSD," in *19th USENIX Symposium on Operating Systems Design and Implementation*, ser. OSDI, 2025.
- [41] M. Ha, E. Kim, and H. Kim, "H3: Hybrid Architecture Using High Bandwidth Memory and High Bandwidth Flash for Cost-Efficient LLM Inference," in *IEEE Computer Architecture Letters*, 2026.
- [42] G. Haas and V. Leis, "What Modern NVMe Storage Can Do, and How to Exploit it: High-Performance I/O for High-Performance Storage Engines," in *Proceedings of the VLDB Endowment, Volume 16, Issue 9*, ser. VLDB, 2023.
- [43] M. E. Houle, "Local intrinsic dimensionality i: An extreme-value-theoretic foundation for similarity applications," in *Similarity Search and Applications*, 2017.
- [44] Y. Huang, X. Fan, S. Yan, and C. Weng, "Neos: A NVMe-GPUs Direct Vector Service Buffer in User Space," in *2024 IEEE 40th International Conference on Data Engineering*, ser. ICDE, 2024.
- [45] J. Jang, H. Choi, H. Bae, S. Lee, M. Kwon, and M. Jung, "CXL-ANNS: Software-Hardware collaborative memory disaggregation and computation for Billion-Scale approximate nearest neighbor search," in *2023 USENIX Annual Technical Conference*, ser. ATC, 2023.
- [46] Y. Jun, S. Park, J.-U. Kang, S.-H. Kim, and E. Seo, "We ain't afraid of no file fragmentation: Causes and prevention of its performance impact on modern flash SSDs," in *22nd USENIX Conference on File and Storage Technologies*, ser. FAST, 2024.
- [47] H. Jegou, M. Douze, and C. Schmid, "Product quantiza- ´

- tion for nearest neighbor search," in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2011.
- [48] M. Kuschewski, J. Giceva, T. Neumann, and V. Leis, "High-Performance Query Processing with NVMe Arrays: Spilling without Killing Performance," in *Proceedings of the ACM on Management of Data, Volume 2, Issue 6*, ser. SIGMOD, 2024.
- [49] S. Li, J. Lin, F. Tu, Z. Wang, L. Liu, Y. Kang, Y. Ding, and Y. Xie, "ECSSD: Hardware/Data Layout Co-Designed In-Storage-Computing Architecture for Extreme Classification," in *Proceedings of the 50th Annual International Symposium on Computer Architecture*, ser. ISCA, 2023, pp. 814–827.
- [50] Y. Li, Y. Jin, B. Tian, H. Zhang, and M. Gao, "ANS-MET: Approximate Nearest Neighbor Search with Near-Memory Processing and Hybrid Early Termination," in *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, ser. ISCA, 2025.
- [51] D. G. Lowe, "Distinctive image features from scaleinvariant keypoints," in *International Journal of Computer Vision*, 2004.
- [52] Y. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs," in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2020.
- [53] M. D. Manohar, Z. Shen, G. Blelloch, L. Dhulipala, Y. Gu, H. V. Simhadri, and Y. Sun, "ParlayANN: Scalable and Deterministic Parallel Graph-Based Approximate Nearest Neighbor Search Algorithms," in *Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming*, ser. PPoPP, 2024.
- [54] J. Mohoney, D. Sarda, M. Tang, S. R. Chowdhury, A. Pacaci, I. F. Ilyas, T. Rekatsinas, and S. Venkataraman, "Quake: Adaptive Indexing for Vector Search," in *19th USENIX Symposium on Operating Systems Design and Implementation*, ser. OSDI, 2025.
- [55] H. Ootomo, A. Naruse, C. Nolet, R. Wang, T. Feher, and Y. Wang, "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs," in *2024 IEEE 40th International Conference on Data Engineering*, ser. ICDE, 2024.
- [56] J. J. Pan, J. Wang, and G. Li, "Survey of vector database management systems," in *The VLDB booktitle*, 2024.
- [57] J. Pennington, R. Socher, and C. Manning, "GloVe: Global vectors for word representation," in *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, ser. EMNLP, 2014.
- [58] A. Prokopenko, D. Lebrun-Grandie, and D. Arndt, "Fast tree-based algorithms for DBSCAN for low-dimensional data on GPUs," in *Proceedings of the 52nd International Conference on Parallel Processing*, ser. ICPP, 2023.
- [59] S. Qiu, W. Liu, Y. Hu, J. Yan, Z. Shen, X. Yao, R. Chen, G. Zhang, and Y. Zhang, "GeminiFS: A companion file system for GPUs," in *23rd USENIX Conference on File and Storage Technologies*, ser. FAST, 2025.

- [60] D. Quinn, M. Nouri, N. Patel, J. Salihu, A. Salemi, S. Lee, H. Zamani, and M. Alian, "Accelerating retrievalaugmented generation," in *Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1*, ser. ASPLOS, 2025.
- [61] Z. Qureshi, V. S. Mailthody, I. Gelado, S. Min, A. Masood, J. Park, J. Xiong, C. J. Newburn, D. Vainbrand, I.- H. Chung, M. Garland, W. Dally, and W.-m. Hwu, "Gpuinitiated on-demand high-throughput storage access in the bam system architecture," in *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2*, ser. ASPLOS, 2023.
- [62] J. Ren, M. Zhang, and D. Li, "HM-ANN: efficient billion-point nearest neighbor search on heterogeneous memory," in *Proceedings of the 34th International Conference on Neural Information Processing Systems*, ser. NeurIPS, 2020.
- [63] E. Schubert, J. Sander, M. Ester, H. P. Kriegel, and X. Xu, "DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN," in *ACM Trans. Database Syst.*, 2017.
- [64] B. Sim, Y. Kim, M. Kim, Y. Park, and J. W. Lee, "Instanns: Scalable approximate nearest neighbor search via cost-efficient in-storage processing," in *Proceedings of the 34th ACM International Conference on Information and Knowledge Management*, ser. CIKM, 2025.
- [65] Z. Song, J. Zhang, J. Sun, M. Sun, Z. Yang, Z. Zhang, X. Chen, F. Wu, H. Tang, and Z. Wang, "CAM: Asynchronous GPU-Initiated, CPU-Managed SSD Management for Batching Storage Access," in *IEEE 41st International Conference on Data Engineering*, ser. ICDE, 2025.
- [66] S. J. Subramanya, Devvrit, R. Kadekodi, R. Krishaswamy, and H. V. Simhadri, "DiskANN: fast accurate billion-point nearest neighbor search on a single node," in *Proceedings of the 33rd International Conference on Neural Information Processing Systems*, ser. NeurIPS, 2019.
- [67] J. Sun, G. Li, J. Pan, J. Wang, Y. Xie, R. Liu, and W. Nie, "GaussDB-Vector: A Large-Scale Persistent Real-Time Vector Database for LLM Applications," in *Proceedings of the VLDB Endowment, Volume 18, Issue 12*, ser. VLDB, 2025.
- [68] B. Tian, H. Liu, Z. Duan, X. Liao, H. Jin, and Y. Zhang, "Scalable billion-point approximate nearest neighbor search using SmartSSDs," in *2024 USENIX Annual Technical Conference*, ser. ATC, 2024.
- [69] B. Tian, H. Liu, Y. Tang, S. Xiao, Z. Duan, X. Liao, H. Jin, X. Zhang, J. Zhu, and Y. Zhang, "Towards Highthroughput and Low-latency Billion-scale Vector Search via CPU/GPU Collaborative Filtering and Re-ranking," in *23rd USENIX Conference on File and Storage Technologies*, ser. FAST, 2025.
- [70] G. T. Toussaint, "The relative neighbourhood graph of a

- finite planar set," in *Pattern Recognition*, 1980.
- [71] M. Wang, W. Xu, X. Yi, S. Wu, Z. Peng, X. Ke, Y. Gao, X. Xu, R. Guo, and C. Xie, "Starling: An i/oefficient disk-resident graph index framework for highdimensional vector similarity search on data segment," in *Proc. ACM Manag. Data*, ser. SIGMOD, 2024.
- [72] M. Wang, X. Xu, Q. Yue, and Y. Wang, "A comprehensive survey and experimental comparison of graph-based approximate nearest neighbor search," in *Proceedings of the VLDB Endowment, Volume 14, Issue 11*, ser. VLDB, 2021.
- [73] Y. Wang, S. Li, Q. Zheng, L. Song, Z. Li, A. Chang, H. H. Li, and Y. Chen, "NDSearch: Accelerating Graph-Traversal-Based Approximate Nearest Neighbor Search through Near Data Processing," in *Proceedings of the 51st Annual International Symposium on Computer Architecture*, ser. ISCA, 2024.
- [74] F. F. Xu, U. Alon, and G. Neubig, "Why do nearest neighbor language models work?" in *Proceedings of the 40th International Conference on Machine Learning*, ser. ICML, 2023.
- [75] Y. Xu, H. Liang, J. Li, S. Xu, Q. Chen, Q. Zhang, C. Li, Z. Yang, F. Yang, Y. Yang, P. Cheng, and M. Yang, "SPFresh: Incremental In-Place Update for Billion-Scale Vector Search," in *Proceedings of the 29th Symposium on Operating Systems Principles*, ser. SOSP, 2023.
- [76] J. Zhang, A. Naruse, X. Li, and Y. Wang, "Parallel topk algorithms on gpu: A comprehensive study and new methods," in *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, ser. SC, 2023.
- [77] Z. Zhang, F. Liu, G. Huang, X. Liu, and X. Jin, "Fast vector query processing for large datasets beyond GPU memory with reordered pipelining," in *21st USENIX Symposium on Networked Systems Design and Implementation*, ser. NSDI, 2024.
- [78] W. Zhao, S. Tan, and P. Li, "SONG: Approximate Nearest Neighbor Search on GPU," in *2020 IEEE 36th International Conference on Data Engineering*, ser. ICDE, 2020.
- [79] X. Zhong, H. Li, J. Jin, M. Yang, D. Chu, X. Wang, Z. Shen, W. Jia, G. Gu, Y. Xie, X. Lin, H. T. Shen, J. Song, and P. Cheng, "VSAG: An Optimized Search Framework for Graph-based Approximate Nearest Neighbor Search," in *Proceedings of the VLDB Endowment, Volume 18, Issue 12*, ser. VLDB, 2025.
- [80] C. Zou and A. A. Chien, "ASSASIN: Architecture Support for Stream Computing to Accelerate Computational Storage," in *Proceedings of the 55th Annual IEEE/ACM International Symposium on Microarchitecture*, ser. MI-CRO, 2022.