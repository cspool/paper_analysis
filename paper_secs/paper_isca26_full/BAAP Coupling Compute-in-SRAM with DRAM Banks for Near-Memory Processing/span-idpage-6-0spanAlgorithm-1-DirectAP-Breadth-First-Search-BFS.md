# <span id="page-6-0"></span>Algorithm 1 DirectAP Breadth-First Search (BFS)

```
Input: Graph edges loaded into AP column-wise with fields:
1: FROM, TO, PROC(ESSED), CURR(ENT)
2: procedure AP-BFS(ap, start_node)
3: traversal_order ← [ ]
4: traversal_order.push(start_node)
5: ap.search(TO, start_node)
6: ap.searchacc(FROM, start_node)
7: ap.update(CURR, true)
8: repeat
9: tmp ← [ ]
10: ap.search({CURR,PROC}, {true,false})
11: if ap.tag popcount() = 0 then
12: break ▷ Current frontier empty. BFS done!
13: end if
14: ap.update(PROC, true)
15: tagged ← ap.read tags()
16: ap.set tags(0)
17: for all edge e in tagged do
18: if e.TO ∈/ traversal_order then
19: tmp.push(e.TO)
20: ap.searchacc({FROM,PROC},{e.TO,false})
21: end if
22: end for
23: ap.write tags to(CURR, true)
24: for all node in tmp do
25: ap.search(TO, node)
26: ap.update(PROC, true)
27: traversal_order.push(node)
28: end for
29: until BFS done!
30: Output traversal_order
31: end procedure
```

PROCESSED to prevent future redundant visits.

## *C. Amenability to the DRAM process*

Modern DRAM processes impose stringent constraints on bank-adjacent logic: fewer metal layers than logic CMOS, slower devices, and tight routing pitch. These limits have historically forced bank-level engines to be extremely simple, as exemplified by UPMEM's 14-stage in-order core (topping out around 500 MHz in the "2x nm" DRAM process) [\[11\]](#page-13-4). At the same time, commodity SRAM macros are routinely routed and closed in peripheries offering as few as three metal layers, so a regular, grid-structured scratchpad macro sits comfortably within the routing resources that DRAM vendors already allocate around each bank. BAAP is deliberately scoped to live within those same constraints by repurposing the existing per-bank SRAM scratchpad, rather than modifying the DRAM subarrays, so it inherits the feasibility envelope of SRAM macros that have already been demonstrated in commercial DRAM silicon.

Physically, BAAP only alters the scratchpad region that UPMEM-like designs already integrate beside each bank. We replace the conventional 6T WRAM with 6T push-rule subarrays that add a single extra wordline, introduce modest changes in the SRAM sense amplifiers to enable switching from differential to single-ended mode, one small AND gate per BL/BLB pair, and tag latches to capture search results. All new structures are short-range, pitch-matched wires and cells that remain within the WRAM macro's routing resources; no wide datapaths or long global wires are introduced, and the array behaves identically to a conventional SRAM in scratchpad mode. Switching modes is a control-only change in the WRAM periphery that has been demonstrated in a 28 nm test chip [\[30\]](#page-13-23), which supports on-the-fly SRAM/BCAM/TCAM reconfiguration with negligible performance impact. Area overheads are explicitly quantified and capacity-adjusted in our evaluation (Section [IV\)](#page-7-0), and BAAP leaves DRAM mats, their sense amplifiers, and the external DRAM timing interface completely unchanged.

Timing and power are kept conservative by design. AP operations toggle only the bitlines in the SRAM slice, confining dynamic activity to the WRAM region and avoiding the long global buses and drive strengths required by subarray ALUs or bitline-computing proposals [\[4\]](#page-12-6), [\[20\]](#page-13-27), [\[37\]](#page-13-22). We conservatively underclock the bank-local AP to 350 MHz—below UPMEM's ≈ 500 MHz core—, to respect DRAM-process limits on peripheral logic. (We perform a sensitivity study (Section [IV-D\)](#page-10-0) in which BAAP is clocked down to 225 MHz; it still delivers competitive performance.) Lastly, functionality is realized via simple sequences of CAM searches and updates over a regular, grid-structured SRAM macro, without adding side arithmetic units in an already tight design that uses a DRAM process. While adopting alternative compute-in-SRAM designs might be possible, our focus here is to present a first-in-kind compute-in-SRAM-in-DRAM solution that is realizable and that yields significant speedups for a broad set of applications. A design space exploration of compute-in-SRAM designs in this context is left for future work. As in prior PIM systems, this modest increase in local switching is more than offset by reduced data movement on off-chip channels. Our evaluation (Section [IV\)](#page-7-0) shows that, for applications with high arithmetic intensity, shorter execution times combined with localized WRAM activity translate into lower end-to-end energy despite operating in a DRAM process.

In summary, BAAP adds associative capability exactly where DRAM vendors already place SRAM, with changes commensurate with prior reconfigurable-SRAM silicon. As we demonstrate in the following section, this can be achieved while staying within proven bank-adjacent power and thermal envelopes.

# <span id="page-6-0"></span>Algorithm 1 DirectAP Breadth-First Search (BFS)

```
Input: Graph edges loaded into AP column-wise with fields:
1: FROM, TO, PROC(ESSED), CURR(ENT)
2: procedure AP-BFS(ap, start_node)
3: traversal_order ← [ ]
4: traversal_order.push(start_node)
5: ap.search(TO, start_node)
6: ap.searchacc(FROM, start_node)
7: ap.update(CURR, true)
8: repeat
9: tmp ← [ ]
10: ap.search({CURR,PROC}, {true,false})
11: if ap.tag popcount() = 0 then
12: break ▷ Current frontier empty. BFS done!
13: end if
14: ap.update(PROC, true)
15: tagged ← ap.read tags()
16: ap.set tags(0)
17: for all edge e in tagged do
18: if e.TO ∈/ traversal_order then
19: tmp.push(e.TO)
20: ap.searchacc({FROM,PROC},{e.TO,false})
21: end if
22: end for
23: ap.write tags to(CURR, true)
24: for all node in tmp do
25: ap.search(TO, node)
26: ap.update(PROC, true)
27: traversal_order.push(node)
28: end for
29: until BFS done!
30: Output traversal_order
31: end procedure
```

PROCESSED to prevent future redundant visits.

## *C. Amenability to the DRAM process*

Modern DRAM processes impose stringent constraints on bank-adjacent logic: fewer metal layers than logic CMOS, slower devices, and tight routing pitch. These limits have historically forced bank-level engines to be extremely simple, as exemplified by UPMEM's 14-stage in-order core (topping out around 500 MHz in the "2x nm" DRAM process) [\[11\]](#page-13-4). At the same time, commodity SRAM macros are routinely routed and closed in peripheries offering as few as three metal layers, so a regular, grid-structured scratchpad macro sits comfortably within the routing resources that DRAM vendors already allocate around each bank. BAAP is deliberately scoped to live within those same constraints by repurposing the existing per-bank SRAM scratchpad, rather than modifying the DRAM subarrays, so it inherits the feasibility envelope of SRAM macros that have already been demonstrated in commercial DRAM silicon.

Physically, BAAP only alters the scratchpad region that UPMEM-like designs already integrate beside each bank. We replace the conventional 6T WRAM with 6T push-rule subarrays that add a single extra wordline, introduce modest changes in the SRAM sense amplifiers to enable switching from differential to single-ended mode, one small AND gate per BL/BLB pair, and tag latches to capture search results. All new structures are short-range, pitch-matched wires and cells that remain within the WRAM macro's routing resources; no wide datapaths or long global wires are introduced, and the array behaves identically to a conventional SRAM in scratchpad mode. Switching modes is a control-only change in the WRAM periphery that has been demonstrated in a 28 nm test chip [\[30\]](#page-13-23), which supports on-the-fly SRAM/BCAM/TCAM reconfiguration with negligible performance impact. Area overheads are explicitly quantified and capacity-adjusted in our evaluation (Section [IV\)](#page-7-0), and BAAP leaves DRAM mats, their sense amplifiers, and the external DRAM timing interface completely unchanged.

Timing and power are kept conservative by design. AP operations toggle only the bitlines in the SRAM slice, confining dynamic activity to the WRAM region and avoiding the long global buses and drive strengths required by subarray ALUs or bitline-computing proposals [\[4\]](#page-12-6), [\[20\]](#page-13-27), [\[37\]](#page-13-22). We conservatively underclock the bank-local AP to 350 MHz—below UPMEM's ≈ 500 MHz core—, to respect DRAM-process limits on peripheral logic. (We perform a sensitivity study (Section [IV-D\)](#page-10-0) in which BAAP is clocked down to 225 MHz; it still delivers competitive performance.) Lastly, functionality is realized via simple sequences of CAM searches and updates over a regular, grid-structured SRAM macro, without adding side arithmetic units in an already tight design that uses a DRAM process. While adopting alternative compute-in-SRAM designs might be possible, our focus here is to present a first-in-kind compute-in-SRAM-in-DRAM solution that is realizable and that yields significant speedups for a broad set of applications. A design space exploration of compute-in-SRAM designs in this context is left for future work. As in prior PIM systems, this modest increase in local switching is more than offset by reduced data movement on off-chip channels. Our evaluation (Section [IV\)](#page-7-0) shows that, for applications with high arithmetic intensity, shorter execution times combined with localized WRAM activity translate into lower end-to-end energy despite operating in a DRAM process.

In summary, BAAP adds associative capability exactly where DRAM vendors already place SRAM, with changes commensurate with prior reconfigurable-SRAM silicon. As we demonstrate in the following section, this can be achieved while staying within proven bank-adjacent power and thermal envelopes.

