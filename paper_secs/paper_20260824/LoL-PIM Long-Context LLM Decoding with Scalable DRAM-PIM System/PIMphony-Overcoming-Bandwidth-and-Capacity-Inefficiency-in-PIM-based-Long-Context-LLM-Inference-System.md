# PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

Hyucksung Kwon\*1, Kyungmo Koo\*1, Janghyeon Kim<sup>1</sup> , Woongkyu Lee<sup>1</sup> , Minjae Lee<sup>1</sup> , Gyeonggeun Jung<sup>2</sup> , Hyungdeok Lee<sup>3</sup> , Yousub Jung<sup>3</sup> , Jaehan Park<sup>3</sup> , Yosub Song<sup>3</sup> , Byeongsu Yang<sup>3</sup> , Haerang Choi<sup>3</sup> , Guhyun Kim<sup>3</sup> , Jongsoon Won<sup>3</sup> , Woojae Shin<sup>3</sup> , Changhyun Kim<sup>3</sup> , Gyeongcheol Shin<sup>3</sup> , Yongkee Kwon<sup>3</sup> ,

Ilkon Kim<sup>3</sup> , Euicheol Lim<sup>3</sup> , John Kim<sup>2</sup> , Jungwook Choi‡<sup>1</sup>

<sup>1</sup>Hanyang University, Seoul, Republic of Korea, <sup>2</sup>KAIST, Daejeon, Republic of Korea,

<sup>3</sup>Solution Advanced Technology, SK hynix, Republic of Korea

{momarom, kookyungmo, kkt20, lwghanyang, Imj4666, choij}@hanyang.ac.kr gyeonggeun@kaist.ac.kr, jjk12@kaist.edu

{hyungdeok.lee, ryan.song, jaehan3.park, yosub.song, byeongsu.yang, haerang.choi, guhyun.kim, jongsoon.won, woojae.shin, changhyun4.kim, gyeongcheol.shin, yongkee.kwon, ilkon.kim, euicheol.lim}@sk.com

*Abstract*—The expansion of long-context Large Language Models (LLMs) creates significant memory system challenges. While Processing-in-Memory (PIM) is a promising accelerator, we identify that it suffers from critical inefficiencies when scaled to long contexts: severe channel underutilization, performancelimiting I/O bottlenecks, and massive memory waste from static KV cache management. In this work, we propose PIMphony, a PIM *orchestrator* that systematically resolves these issues with three co-designed techniques. First, *Token-Centric PIM Partitioning (TCP)* ensures high channel utilization regardless of batch size. Second, *Dynamic PIM Command Scheduling (DCS)* mitigates the I/O bottleneck by overlapping data movement and computation. Finally, a *Dynamic PIM Access (DPA)* controller enables dynamic memory management to eliminate static memory waste. Implemented via an MLIR-based compiler and evaluated on a cycle-accurate simulator, PIMphony significantly improves throughput for long-context LLM inference (up to 72B parameters and 1M context length). Our evaluations show performance boosts of up to 11.3× on PIM-only systems and 8.4× on xPU+PIM systems, enabling more efficient deployment of LLMs in real-world long-context applications.

