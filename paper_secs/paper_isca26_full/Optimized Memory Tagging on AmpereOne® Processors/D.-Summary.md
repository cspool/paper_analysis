# D. Summary

An implementation where MTE allocation tags are colocated with data meets the requirements for MTE usage for production datacenter deployment: no impact on usable

<span id="page-5-2"></span>![](_page_5_Figure_0.jpeg)

Fig. 5: Ampere MTE System Architectural View

memory capacity, lower hardware complexity, and lower performance overhead. MTE tag storage co-located with data can best be accomplished using ECC bits, but doing so requires enhancements to the memory reliability feature support built on ECC bits to continue providing the high level of memory reliability required for datacenter usages. See Section [VI-B](#page-5-1) for more details.

#### VI. AMPERE'S MTE IMPLEMENTATION

<span id="page-5-0"></span>The AmpereOne® SoC, a high-performance datacenter server-class processor, offers configurations ranging from 96 to 192 custom Arm v8.6+ ISA-compliant Ampere cores. The AmpereOne® SoC's MTE implementation incorporates enhancements across the Core Processing Element, including the L1 and L2 caches, the coherency mesh/interconnect, and the memory hierarchy, including the system-level cache and the MCU. This section details the MTE support integrated into each of these components, focusing on the design and micro-architectural choices that enable Ampere's efficient and performant implementation of synchronous tag checking.

#### *A. Overview*

In Ampere's implementation, tags and data flow together throughout the SoC, the Core Processing Element, and the memory subsystem. This is achieved by ensuring that every subsystem handles the tags and data as a co-located *bundle*. The bundle is first created at the memory subsystem during a memory read when the memory controller reads the tags that are stored in ECC bits inline with the data. The tags are carried over the memory fabric using additional wires and held in caches in dedicated bits which flow through the Core Processing Element pipeline. [Figure 5](#page-5-2) provides an overview of the SoC components and their roles in generating, retaining, and transporting tags.

#### <span id="page-5-1"></span>*B. MTE in the MCU and DRAM*

Within the Memory Controller Unit (MCU), ECC bits are utilized to store allocation tags in DRAM. This necessitates the allocation of four ECC bits for every 16-byte DRAM granule. No tag checks are performed within either the MCU or the DRAM itself.

There is no latency impact within the MCU, as the tag bits are transferred along with the data bits, similar to Error-Correcting Code (ECC) bits. However, a notable impact arises concerning the utilization of ECC for error detection and correction. In certain DRAM and ECC configurations, insufficient metadata bits may be available to store MTE tags. To address this, some ECC bits must be re-purposed for tag storage. Ampere accommodates this by offering multiple ECC schemes tailored to various DRAM configurations. Specifically, a novel ECC scheme is employed that provides the requisite reliability protection for datacenter class SoCs while simultaneously making bits available for MTE allocation tags [\[45\]](#page-12-15). The selection of the appropriate ECC scheme is determined by both the DIMM configuration and the desired MTE support. AmpereOne® SoC systems support four schemes for ECC:

- A. SECDED baseline without MTE support (SECDED-64+8): The code word is 72 bits (64 bits data + 8 bits ECC) with 8 codewords per 64 byte cacheline.
- B. SECDED with MTE support (SECDEC-128+4+9): The code word is 128+4+9 bits (128 bits data + 4 bits tag + 9 bits ECC) with 4 codewords per 64B cacheline. This scheme doubles the data granule from (A), allows storage of tag in spare unused ECC bits, retaining full memory reliability feature support without compromise.
- C. SymbolECC baseline without MTE support (SymbolECC-64+16): 64 bytes data divided into 8 codewords of 10 nibbles. Reed-Solomon allows one symbol correction per codeword. All the available metadata bits are used for ECC.
- D. SymbolECC with MTE support (SymbolECC-64-14+2): Similar to (C) but borrows 2 bits from ECC Parity, resulting in some Correctable errors becoming only Detectable. Compared to (C), this provides 100% of the fault detection, 100% correction of bounded faults, and 99.98% correction of unbounded faults.

For deploying MTE, options B and D are available to the cloud service provider, and both showcase reliability metrics within the bounds required for datacenter operation.

#### *C. MTE in the Coherent Mesh*

Within the mesh, tags are transported using reserved bits that are supported for implementation specific uses. This usage mirrors the use of metadata and ECC bits in the MCU and DRAM with tags being transported along with the data. On the AmpereOne® SoC, this required customization of the mesh to allow for metadata transport along with data. All cache line storage in the mesh is expanded to include tag bits, and this storage is also covered by ECC. No tag checks are performed in the Mesh.

#### *D. MTE in the Core Processing Element (PE)*

Within the Core PE, data tags are transported alongside their corresponding data throughout the pipeline via widened datapaths and are stored co-resident with data in the widened caches. Tag validation is performed at the cache-lookup point within the PE pipeline. These checks proceed in parallel with address translation and access-permission checks (e.g., page-fault detection) and therefore do not introduce additional pipeline stages or stalls.

For loads, this design imposes negligible overhead relative to untagged execution because tag validation is integrated into the existing cache-lookup path. For stores, the primary effect is that the memory tag of the target cache line must be retrieved and validated before the store can commit, which constitutes the dominant MTE-related cost in the core. The implementation incorporates mitigations for this extra cost, including early line fetch for tagged stores.

For tagged stores, the cache-line fetch (including its tag) is initiated during address translation, just like for any normal load, which enables out-of-order overlap of the data/tag fetch with other work. As a result, tag check validation for stores can complete far earlier than the commit pipeline stage.

Store-to-load forwarding presents a specific challenge: a younger load may alias an older store whose target line's memory tag has not yet been fetched and validated. To preserve correctness, the allocation tag (address tag) is recorded in the store buffer, and a load is permitted to forward only if its allocation tag matches that of the older store. At the time of forwarding, the outcome of the store's tag check may still be unknown; however, if the allocation tags match, the subsequent tag validation will either succeed for both operations or fault the store (and thus the speculative load), maintaining correctness. Forwarding across a tag-store instruction (which writes the memory tag) is disallowed, as it could violate this invariant property. Thus, many stores experience no additional latency, while others incur a modest delay.

