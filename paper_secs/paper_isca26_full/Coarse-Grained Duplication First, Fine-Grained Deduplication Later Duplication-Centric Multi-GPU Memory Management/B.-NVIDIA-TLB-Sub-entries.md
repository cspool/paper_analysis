# *B. NVIDIA TLB Sub-entries*

Traditionally, a TLB entry maps one virtual page to one physical page—a simple one-to-one correspondence, as in the L1 TLBs of recent NVIDIA GPUs (e.g., Ampere). To extend reach, NVIDIA's L2/L3 TLBs partition each entry into subentries [46]. Each entry comprises 16 sub-entries; each subentry maps one page (4KB, 64 KB or 2 MB) within an aligned 32KB, 1 MB or 32 MB region, respectively.

Figure 1 (b) illustrates the process of Lookup with sub-entry TLBs. A virtual address is split into a virtual page number (VPN) and a page offset. The lower bits of the VPN further divide into a TLB index (set selection) and a sub-entry index, while the remaining higher bits form the virtual page base (VPB). On access, the TLB index selects a set 1 , and the VPB from the virtual address is compared against VPB tags stored in that set 2 . If there is a VPB match (an "entry hit"), the sub-entry index selects the corresponding slot 3 ; if that slot is non-zero (valid), the translation hits. If no VPB matches, the access misses and triggers a page-table walk.

On a walk completion, if the page lies within the region already covered by an existing TLB entry, the translation is installed into the corresponding sub-entry slot. If no covering entry exists, the LRU entry is evicted (clearing all 16 subentries), a new entry is allocated for the region.

# *B. NVIDIA TLB Sub-entries*

Traditionally, a TLB entry maps one virtual page to one physical page—a simple one-to-one correspondence, as in the L1 TLBs of recent NVIDIA GPUs (e.g., Ampere). To extend reach, NVIDIA's L2/L3 TLBs partition each entry into subentries [46]. Each entry comprises 16 sub-entries; each subentry maps one page (4KB, 64 KB or 2 MB) within an aligned 32KB, 1 MB or 32 MB region, respectively.

Figure 1 (b) illustrates the process of Lookup with sub-entry TLBs. A virtual address is split into a virtual page number (VPN) and a page offset. The lower bits of the VPN further divide into a TLB index (set selection) and a sub-entry index, while the remaining higher bits form the virtual page base (VPB). On access, the TLB index selects a set 1 , and the VPB from the virtual address is compared against VPB tags stored in that set 2 . If there is a VPB match (an "entry hit"), the sub-entry index selects the corresponding slot 3 ; if that slot is non-zero (valid), the translation hits. If no VPB matches, the access misses and triggers a page-table walk.

On a walk completion, if the page lies within the region already covered by an existing TLB entry, the translation is installed into the corresponding sub-entry slot. If no covering entry exists, the LRU entry is evicted (clearing all 16 subentries), a new entry is allocated for the region.

