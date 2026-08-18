# *D. Stream Flipping*

The 3D-DRAM stack connects logic to DRAM through a dense µbump array. These short vertical links reduce per-bit I/O energy to 0.45 pJ/bit, ∼6× lower than HBM3, but the much higher aggregate bandwidth amplifies total switching power. At 100 TB/s a Raptor card would dissipate ∼360 W in 3D-DRAM I/O alone. Reducing bit transitions on the µbump array is therefore essential.

![](_page_4_Figure_8.jpeg)

Fig. 6: Naive column-staggering versus stream-blocking. (a) A naive scheme staggers column indices across three banks so each pair of column accesses can pack one 128B flit without overfetch, but this requires a 192B data-shifting network and complex per-address column patterns. (b) Stream blocking instead uses a fixed two-access pattern with a small 96B partial buffer and 96B output buffer, fully utilizing every column access and eliminating overfetch.

Commodity DRAM interfaces address this with data bus inversion (DBI): for each beat in a multi-cycle burst, the PHY decides whether to invert the word to minimize transitions and signals the choice on a dedicated DBI pin. Our 3D-DRAM instead transfers a single-cycle 256-bit word per bank with no burst structure and no DBI pin. Measurements show that DBI-equivalent encoding would reduce I/O energy by 18% (to ∼0.37 pJ/bit), so we implement it at the architectural level.

*Stream flipping* exploits two properties of Raptor: weights and KV cache are laid out as stream-blocked tiles producing long, near-unit-stride 128 B flit streams per channel, and the software stack controls tile placement. On a write, the memory controller compares each flit to the previously written flit on the same channel and selectively inverts it to minimize transitions, recording a single metadata bit per flit in a compact side region. On a read (Figure 7), the controller fetches the metadata bit alongside the flit and inverts it if set. The metadata follows the same stream-blocked mapping, and this architectural DBI scheme reduces 3D-DRAM I/O energy by 18% without changes to the DRAM PHY.

