# *E. Stage 4: Output Selection*

Stage 4 receives the lane-wise accumulated results and selects the output based on the input data type. The P lane results are concatenated into the final packed output word (e.g., four FP8 lanes or two BF16 lanes forming 32 bits). No additional normalization or conversion is required, and the stage emits one multi-lane MAC result per cycle.

## *F. Pipeline Behavior*

XtraMAC adopts a fixed four-stage logical pipeline in which each stage contains a bounded combinational block followed by a register boundary, cleanly separating logic evaluation from temporal sequencing. The DSP slice is configured with its internal pipeline registers disabled so that the multiplication behaves as a purely combinational block between Stage 1 and Stage 2 registers. All interface signals are time-aligned through matched delay slices to sustain continuous throughput. For example, the input datatype signal, consumed in both Stage 1 and Stage 4, is delayed by the appropriate number of register slices before reaching Stage 4, ensuring that control and data correspond to the same dynamic operation. Likewise, the input operand C, required only in Stage 3, is delayed to align with the products emerging from Stage 2.

XtraMAC fixes the pipeline at four logical stages but leaves the cycle count of each stage configurable at synthesis time. By default, every stage completes in a single clock cycle, yielding an end-to-end latency of four cycles. When a stage with complex logic (e.g., Stage 3) becomes the critical path, the designer can insert additional pipeline registers within it to trade latency for a higher clock frequency. This requires only extending the matched delay slices on parallel paths; the initiation interval remains one. By combining per-stageconfigurable latency with pipeline-aligned propagation, Xtra-MAC provides a deterministic latency and an initiation interval of one across all supported datatypes.

## *G. System Integration*

Although XtraMAC adds an input data type port for runtime datatype selection, this signal functions solely as pipelinealigned control metadata and does not modify the operand interface or timing behavior of a standard MAC unit. All datatype-dependent formatting and reconstruction logic resides inside the XtraMAC pipeline, so external modules continue to provide and consume the same lane-packed operands used in existing accelerators. With a fixed latency and an initiation interval of one across all datatypes, XtraMAC can replace a conventional MAC unit without requiring any schedule or interface changes, enabling easy system integration.

## V. EVALUATION

In this section, we evaluate XtraMAC in terms of its mixed-precision coverage and runtime datatype adaptability. We further compare its resource efficiency and performance against state-of-the-art FPGA baselines, including the AMD Xilinx Floating-Point Operator [\[1\]](#page-12-3) and TATAA [\[38\]](#page-13-8).

