# A Artifact Appendix

## A.1 Abstract

The artifact contains the source code of our prototype system and the scripts for conducting the experiments presented in the paper.

## A.2 Description & Requirements

The artifact is available at [https://doi.org/10.5281/zenodo.](https://doi.org/10.5281/zenodo.17213486) [17213486](https://doi.org/10.5281/zenodo.17213486).

A.2.1 Hardware Dependencies. The artifact requires an Orange Pi 5 Plus board [\[17\]](#page-14-10) (RK3588 CPU/NPU [\[21\]](#page-14-11)). A standalone machine is required to build the artifact and communicate with the board using a USB-to-USB cable. The machine should have at least 8 GB of memory and 100 GB of free disk space.

A.2.2 Software Dependencies. The standalone machine should operate on a Linux OS (Ubuntu 22.04.3 tested) and have the following dependencies installed: OpenHarmony Device Connector for board communication, Python3 and its matplotlib library for plotting and analysis, and Docker for containerized builds.

## A.3 Setup

The user needs to download the source code. Please refer to the README.md for details.

## A.4 Evaluation Workflow

## A.4.1 Major Claims.

- Claim C1: TZ-LLM can reduce TTFT by 76.1%∼90.9% compared to the Strawman baseline and incurs 5.2%∼28.3% overhead compared to the REE-LLM-Memory baseline. This is demonstrated by experiment E1, whose results are reported in Figure [10.](#page-11-1)
- Claim C2: TZ-LLM can increase decoding speed by 0.9%∼23.2% compared to the Strawman baseline and incurs 1.3%∼4.9% overhead compared to the REE-LLM baseline. This is demonstrated by experiment E2, whose results are reported in Figure [11.](#page-11-2)
- Claim C3: As more parameters are cached, partial parameter caching can reduce TTFT approximately linearly up to a threshold. This is demonstrated by experiment E3, whose results are reported in Figure [14.](#page-12-5)

## A.4.2 Experiments.

- Experiment E1 (approximately 60 compute-minutes): Run script scripts/1-end-to-end-prefill.sh, which evaluates the TTFT of TZ-LLM and other baselines across different benchmarks. The results are displayed in plots/figure10.pdf.
- Experiment E2 (approximately 20 compute-minutes): Run script scripts/2-end-to-end-decoding.sh, which evaluates the decoding speed of TZ-LLM and other baselines across different models. The results are displayed in plots/figure11.pdf.
- Experiment E3 (approximately 60 compute-minutes): Run script scripts/3-caching.sh, which evaluates the effect of partial parameter caching on the TTFT of TZ-LLM across different cache proportions and prompt lengths. The results are displayed in plots/figure14.pdf.