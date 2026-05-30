# *B. Dynamic PIM Access (DPA) Instructions*

Based on the insights of Attention's computational structure, we introduce Dynamic PIM Access (DPA) instructions, a flexible instruction set designed to avoid the static execution model of conventional PIMs. As outlined in Fig. 10(b), DPA instructions empower the PIM to handle dynamic workloads by encapsulating repetitive token operations into a compact, runtime-executable format. This is achieved through two key instructions:

- Dyn-Loop instruction encodes a loop structure where the number of repetitions (Loop-Bound) is determined dynamically based on the request's actual token length, not a pre-defined maximum token length.
- Dyn-Modi instruction operates within this loop to adjust target operand fields (e.g., the row/col address for a MAC instruction) using a specified stride. This mechanism effectively generates a virtual address that is translated to a physical address on-the-fly, enabling the PIM to access dynamically allocated and non-contiguous KV cache memory—a capability impossible for conventional PIM controllers.

This dynamic and compact representation provides an advantage over prior PIM systems ( [16], [21], [54], [62]), which must generate instruction sequences whose size grows linearly with the token length. As shown in Fig. 10(c), this linear growth creates severe instruction buffer pressure and a scalability bottleneck for long-context inference. In comparison, DPA's encoding ensures the instruction size remains small and nearly constant regardless of context length, thereby avoiding command buffer bloat and enabling scalability to long context.

