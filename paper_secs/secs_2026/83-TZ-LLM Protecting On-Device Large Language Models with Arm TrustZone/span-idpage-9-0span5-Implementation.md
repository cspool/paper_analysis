# <span id="page-9-0"></span>5 Implementation

We prototype TZ-LLM on OpenHarmony OS [\[16\]](#page-14-5) and its TEE system, which is an open-sourced version of Huawei's commercial HarmonyOS [\[8\]](#page-13-6). The LLM TA is built based on llama.cpp [\[13\]](#page-14-6), a popular on-device inference framework. The REE OS is OpenHarmony v4.1 with Linux v5.10. The NPU driver is Rockchip NPU driver v0.9.8 [\[22\]](#page-14-4).

The original TEE OS contains 17K LoC for basic functionalities, including thread management, IPC, interrupt dispatching, and memory management. We only extend it with 62 LoC to manage CMA page memory mapping and 50 LoC to support dynamic configuration of TZASC and TZPC. The llama.cpp inference framework is extended with 1.2K LoC for pipelined restoration, 1K LoC for integrating the data plane of the NPU drive, and the OpenSSL library [\[24\]](#page-14-21) for parameter decryption. Note that the computation graph is directly extracted via internal interfaces of llama.cpp. In the REE OS, we add only 364 LoC to the Linux kernel, which consists of 167 LoC in the NPU driver for shadow job scheduling and 197 LoC in the TZ driver for CMA allocation and deallocation.

The current implementation works on a Rockchip platform, while TZ-LLM design is applicable to other Arm platforms, such as Qualcomm. We investigate the open-source Linux driver for Qualcomm NPUs [\[18\]](#page-14-22) and confirm that we can also extract a small data plane driver from it.

