# Learning 6L Reference

L1–L6 are coordinates for modifiable performance objects, not a required
pipeline and not six boxes that every Direction must fill.

| Layer | Modifiable objects |
|---|---|
| L1 Algorithm/Pipeline | compute graph, workload decomposition, dynamic parameters, approximation, parallelism |
| L2 Serving/Runtime | requests, batches, stages, queues, placement, cache, resource and execution scheduling |
| L3 Compiler | IR, dependency representation, passes, fusion, multiversioning, code generation |
| L4 Kernel | tile/warp/instruction pipelines, synchronization, memory movement, launch and kernel composition |
| L5 Architecture | compute/control units, scheduler, memory hierarchy, NoC and hardware primitives |
| L6 Chip/System | chiplet, PIM, wafer scale, packaging, interconnect and chip-level capacity boundaries |

Rules:

- An Anchor is a region center defined by acceleration scenario plus baseline
  and performance tension.
- The active Anchor set is the current Topic 6L space; adding an Anchor expands
  that space without silently changing the user Topic.
- One Anchor may touch any non-empty subset of layers.
- A Direction is one falsifiable modification path inside one Anchor. It may
  span one or several layers only when the causal chain requires them.
- Explain exact data, control, resource, or synchronization interfaces for
  cross-layer claims.
- Never assemble unrelated layer methods merely to claim full-stack coverage.
