# *C. Description*

*1) How to access:* The artifact is archived in Zenodo<sup>1</sup> . It can also be accessed from GitHub using the command shown below:

- \$ git clone https://github.com/redbird-arch/ isca2026-busybarn-artifact.git
- *2) Hardware dependencies:* A multi-core CPU with at least 32 GB of RAM is required. The ablation study (Fig. 14) simulates a 16×16 core mesh and is memory-intensive (∼32 GB per task). All other experiments require ≤4 GB per task. A SLURM-managed cluster is supported but not required. For reference, we list our system configurations here:

For simulation experiments:

- OS: Ubuntu 22.04.5 LTS
- CPU: Intel 13th Generation Intel Core i9 Processors @ 3.00GHz (24 cores); Intel(R) Xeon(R) Gold 6348H CPU 24C @ 2.60GHz.
- DRAM: 64 GB; 1.5 TB

<sup>1</sup>https://doi.org/10.5281/zenodo.19686855

*3) Software dependencies:* Conda (Miniconda or Anaconda) with Python 3.9. Key Python packages: numpy, networkx, simanneal, matplotlib, tqdm, and PyYAML. All dependencies are listed in requirements.txt.

