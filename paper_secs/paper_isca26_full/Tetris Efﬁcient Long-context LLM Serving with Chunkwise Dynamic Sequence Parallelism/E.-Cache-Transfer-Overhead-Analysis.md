# E. Cache Transfer Overhead Analysis

**CDSP Cache Balancing:** To evaluate the overhead under different length ratios, we set current chunk's token number to 128k/64k for LLaMA3-8B/70B, and vary the historical token number from 25% to  $2\times$  of it. For each setting, we test both intra-node and inter-node overheads. As shown in Fig. 17-(a)~(d), CDSP balancing only incurs up to 1.8% extra overhead, proving the efficiency of the overlap strategy.

**CDSP Handshake:** To assess the multi-instance cache transfer overhead, we first test under the largest SP sizes with max backend allocation. Since the capacity is sufficient under our settings, each prefill instance can be assigned a dedicated backend. As shown in Fig. 17-(e)~(f), cache transfer incurs 0.6%-11.8% (average 2.1%) overhead. We then halve the backend number to conduct stress tests under limited capacity, which results in only 1.5%-5.4% (average 3.8%) additional RPC overhead. The handshake-based management mechanism can efficiently utilize buffer-backed transfer backends.

