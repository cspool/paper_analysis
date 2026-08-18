# *C. Stream Blocking*

Each 3D-DRAM bank is a 1364×124 array; the row buffer spans all 124 columns, and each column access returns 256 bits (32 B). A key design goal is to expose many independent channels, each feeding the weight buffer (WB) of a single tensor engine (TE). All banks in a channel are activated in a stagged manner. Channels must operate independently so that refresh, scrub, or other maintenance events on one bank do not stall unrelated channels.

*1) Channel Layout and Bank Budget:* Each slice has 16 weight buffers (WB) and 16 DRAM channels. Each channel delivers a 128 B flit to its WB. A single-column access from one bank supplies 32 B, so matching a 128 B flit requires four co-accessed banks. However, 256 channels per chiplet, across four banks each, require 1024 banks, yet the die implements only 840. After reserving 72 as spares (§IV-E), 768 remain,

![](_page_4_Figure_0.jpeg)

Fig. 5: A practical implementation with three banks per channel. Reserving 72 of the 840 banks as spares leaves 768 usable banks, so that each of the 256 channels per chiplet can be formed from exactly three banks.

resulting in three per channel (Figure 5). Three co-accessed banks return 96 B per access, so a 128 B flit requires two accesses and causes a systematic overfetch.

- *2) Naive Column-Staggering:* Staggering column indices across banks so each pair of accesses yields exactly 128 B (Figure 6(a)) eliminates overfetch but requires the controller to buffer 192 B, shift non-aligned fragments into 128 B flits, and track per-address column patterns, which complicates the datapath and hinders timing closure.
- *3) Stream Blocking:* We instead adopt *stream blocking* (Fig. 6(b)). The software stack tiles weights and KV cache into per-channel blocks with near-unit-stride 128 B streaming access to maximize row-buffer locality. Each flit is split into a 96 B aligned portion stored across the three banks at the same column index (32 B per bank) and a 32 B partial; partials from neighboring flits are packed into a shared partial region.

On a read, the controller first accesses the partial region, caching 96 B of 32 B fragments for consecutive flits. A second access fetches the 96 B aligned region and merges it with the buffered partial. Both accesses follow a fixed pattern with small buffers (96 B each), reducing the datapath to fixed-index concatenation and fully utilizing every column access.

