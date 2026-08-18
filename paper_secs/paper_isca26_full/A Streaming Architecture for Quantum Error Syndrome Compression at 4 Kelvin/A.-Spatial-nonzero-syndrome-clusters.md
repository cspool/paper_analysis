# A. Spatial nonzero syndrome clusters

Nonzero syndromes arise from single data errors, measurement errors, or error chains (Section II-A). The method introduced here targets the first category—spatial clusters

TABLE I PRIORITY AMONG THE FOUR SPATIAL PATTERNS BASED ON THEIR INDEX-REDUCTION POTENTIAL. HIGHER SCORES OVERRIDE LOWER ONES.

| Pattern    | Error | Opcode/Priority | Index reduction |  |
|------------|-------|-----------------|-----------------|--|
| Cross      | Y     | 3               | 75%             |  |
| Vertical   | X/Z   | 2               | 50%             |  |
| Horizontal | X/Z   | 1               | 50%             |  |
| Isolated   | M     | 0               | 0%              |  |

of nonzero syndromes caused by single bit-flip (X), phaseflip (Z), or bit- and phase-flip (Y ) errors. The goal is to represent the two or four nonzero syndromes associated with these errors using one index.

We identify three patterns and assign a distinct opcode to each: a horizontal pair (X or Z errors), a vertical pair (X or Z errors), and a cross (Y error). Figure 4 provides an illustration. Detecting these patterns within a surface code requires only a local view of neighboring ancilla qubits. This resembles how local decoders operate in hierarchical solutions; however, a crucial distinction is that local decoders rely on this limited view to make decisions, which results in accuracy losses [2], [61]. In contrast, our method makes no decoding decisions at this stage. Instead, it applies compression that is losslessly reversed before performing full-accuracy decoding.

Now, assume that ancilla qubits within a logical qubit are indexed in ascending order from left to right and top to bottom. Upon encountering the first nonzero syndrome, its index gets recorded to indicate its position within the lattice. Its neighboring syndromes are then examined for nonzero values matching the horizontal, vertical, or cross patterns described above. If a match is found, the corresponding opcode from Figure 4 is appended to the stored index, while the indices of other syndromes forming the pattern are omitted. This process continues across the entire lattice.

The cross pattern provides the highest compression savings, representing four syndrome indices with just one. Thus, it is assigned the highest priority, overriding any horizontal or vertical matches. If both horizontal and vertical pair patterns are detected but not a cross, priority is given to the vertical pair. While the method would still function with horizontal pairs taking precedence, vertical pairs are chosen as they result in simpler circuitry. If no pattern matches—indicating the syndrome is part of a measurement error or an error chain of arbitrary length—the opcode is set to 0, marking it as an isolated error, which we handle in Section IV-B. Table I summarizes these priorities.

Figure 5 quantifies the compression performance of this method across error rates ranging from 0.01% to 1% and code distances d from 11 to 31. The results show that, in the presence of only data errors, it consistently eliminates 57–61% of indices. When measurement errors are introduced, the reduction drops to 32–35%, as these errors typically do not form multi-syndrome clusters within a single round, causing them to be represented similarly in the sparse representation and our scheme. This observation motivates the temporal

![](_page_4_Figure_7.jpeg)

Fig. 5. Nonzero syndrome index reduction achieved by compressing spatially local clusters in surface codes across various code distances and physical error rates. With only data errors, the number of such indices decreases by 57–61%. When measurement errors are included, the reduction drops to 32–35% due to measurement errors not forming multi-syndrome clusters in a single measurement round.

pattern-based compression approach that follows.

Note that existing syndrome compression schemes, such as AFS, do not handle nonzero syndromes and therefore achieve no index reduction (rate = 0.0).

# A. Spatial nonzero syndrome clusters

Nonzero syndromes arise from single data errors, measurement errors, or error chains (Section II-A). The method introduced here targets the first category—spatial clusters

TABLE I PRIORITY AMONG THE FOUR SPATIAL PATTERNS BASED ON THEIR INDEX-REDUCTION POTENTIAL. HIGHER SCORES OVERRIDE LOWER ONES.

| Pattern    | Error | Opcode/Priority | Index reduction |  |
|------------|-------|-----------------|-----------------|--|
| Cross      | Y     | 3               | 75%             |  |
| Vertical   | X/Z   | 2               | 50%             |  |
| Horizontal | X/Z   | 1               | 50%             |  |
| Isolated   | M     | 0               | 0%              |  |

of nonzero syndromes caused by single bit-flip (X), phaseflip (Z), or bit- and phase-flip (Y ) errors. The goal is to represent the two or four nonzero syndromes associated with these errors using one index.

We identify three patterns and assign a distinct opcode to each: a horizontal pair (X or Z errors), a vertical pair (X or Z errors), and a cross (Y error). Figure 4 provides an illustration. Detecting these patterns within a surface code requires only a local view of neighboring ancilla qubits. This resembles how local decoders operate in hierarchical solutions; however, a crucial distinction is that local decoders rely on this limited view to make decisions, which results in accuracy losses [2], [61]. In contrast, our method makes no decoding decisions at this stage. Instead, it applies compression that is losslessly reversed before performing full-accuracy decoding.

Now, assume that ancilla qubits within a logical qubit are indexed in ascending order from left to right and top to bottom. Upon encountering the first nonzero syndrome, its index gets recorded to indicate its position within the lattice. Its neighboring syndromes are then examined for nonzero values matching the horizontal, vertical, or cross patterns described above. If a match is found, the corresponding opcode from Figure 4 is appended to the stored index, while the indices of other syndromes forming the pattern are omitted. This process continues across the entire lattice.

The cross pattern provides the highest compression savings, representing four syndrome indices with just one. Thus, it is assigned the highest priority, overriding any horizontal or vertical matches. If both horizontal and vertical pair patterns are detected but not a cross, priority is given to the vertical pair. While the method would still function with horizontal pairs taking precedence, vertical pairs are chosen as they result in simpler circuitry. If no pattern matches—indicating the syndrome is part of a measurement error or an error chain of arbitrary length—the opcode is set to 0, marking it as an isolated error, which we handle in Section IV-B. Table I summarizes these priorities.

Figure 5 quantifies the compression performance of this method across error rates ranging from 0.01% to 1% and code distances d from 11 to 31. The results show that, in the presence of only data errors, it consistently eliminates 57–61% of indices. When measurement errors are introduced, the reduction drops to 32–35%, as these errors typically do not form multi-syndrome clusters within a single round, causing them to be represented similarly in the sparse representation and our scheme. This observation motivates the temporal

![](_page_4_Figure_7.jpeg)

Fig. 5. Nonzero syndrome index reduction achieved by compressing spatially local clusters in surface codes across various code distances and physical error rates. With only data errors, the number of such indices decreases by 57–61%. When measurement errors are included, the reduction drops to 32–35% due to measurement errors not forming multi-syndrome clusters in a single measurement round.

pattern-based compression approach that follows.

Note that existing syndrome compression schemes, such as AFS, do not handle nonzero syndromes and therefore achieve no index reduction (rate = 0.0).

