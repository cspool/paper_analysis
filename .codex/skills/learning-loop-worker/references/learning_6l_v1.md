# Learning 6L Reference

L1–L6 are coordinates for concrete, modifiable performance objects. They are
not a required pipeline and not six boxes that every Anchor or Direction must
fill.

| Layer | Concrete object examples |
|---|---|
| L1 Algorithm/Pipeline | computation flow, operator sequence, input/output tensors, loop or operator dimensions, data dependencies, parallel sub-computations, approximation or pruning sites |
| L2 Serving/Runtime | request queues, batch formation, execution timeline, stage-to-resource mapping, placement, cache, prefill/decode orchestration, offline versus online decisions |
| L3 Compiler | IR and dependency representation, pass pipeline, fusion boundary, code generation, generated variants and runtime variant selection |
| L4 Kernel | grid/block decomposition, tiles, shared memory, registers, warp roles, instruction pipeline, synchronization, occupancy, bank conflicts, kernel composition and memory movement |
| L5 Architecture | data paths among HBM/cache/SMEM/register files/compute units, scheduling and control units, DMA/TMA, isolation, NoC, bandwidth and latency limits |
| L6 Chip/System | multi-GPU or multi-node topology, chiplets, die-to-die links, PIM, package interconnect, NUMA, capacity, power, area, thermal and scale-out limits |

## Sufficient concreteness

A non-null layer entry should let an independent Reviewer identify:

1. the observed or modified object;
2. the layer containing it;
3. the data, control, resource, or synchronization interface through which it
   affects another object when the claim crosses layers;
4. why it can affect a Goal metric or guardrail;
5. how that effect could be observed or falsified.

Layer names and broad technique labels alone do not establish coverage. For
example, `L4: kernel optimization` is only a keyword. A statement such as
`fuse token dispatch, grouped GEMM, and weighted sum to remove intermediate
HBM transfers and launches` identifies a reviewable L4 object and mechanism.

## Rules

- An Anchor is a region center defined by a concrete scenario, performance or
  execution baseline, remaining headroom, and observable performance tension.
- The active Anchor set is the current Topic 6L space; adding an Anchor expands
  that space without silently changing the user Topic.
- One Anchor may touch any non-empty subset of layers. Leave unrelated layers
  `null`; do not fill them with keywords.
- A Direction is one falsifiable modification path inside one Anchor. It may
  span several layers only when its causal chain requires them.
- Preserve the minimum object and interface detail needed to understand the
  performance tension and falsify the claim. Do not demand exhaustive
  implementation descriptions.
- Never assemble unrelated layer methods merely to claim full-stack coverage.
