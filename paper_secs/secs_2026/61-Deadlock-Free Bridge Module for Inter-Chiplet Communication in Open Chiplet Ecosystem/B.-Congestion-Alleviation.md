# *B. Congestion Alleviation*

Beyond reducing the hardware overhead of the DFBM, the CVN-DB further alleviates network congestion through its shared buffer design. Consistent with the observation that positive feedback exists between congestion and deadlock [40]. The CVN-DB suppresses this through its prioritization mechanism, which enforces sequential draining of VNs in dependency order while simultaneously regulating packet injection to prevent resource oversubscription.

### *C. Compared with RC*

While DFBM adheres to RC's packet injection control, beyond the chiplet standardization advantages in Table I, it offers additional performance benefits.

- Implementation Complexity and Robustness: RC mandates chiplet-specific permission networks within the NoC, which elevates failure risks and verification overhead. By contrast, DFBM eliminates intra-chiplet modifications by externalizing control logic.
- Latency Efficiency: RC mandates persistent injection throttling even at near-zero load, incurring fixed latency

- overhead. In comparison, DFBM employs injection control that adjusts throttling according to congestion intensity, preserving near-native latency under light load.
- Resource Optimization: RC mandates dedicated per-VN buffers (rc buffers), while DFBM employs a shared cross-VN deadlock buffer (CVN-DB). The overhead associated with these dedicated buffers significantly affects router area and power dissipation in chiplets.

