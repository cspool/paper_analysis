# *B. Error Correcting Codes*

Error Correcting Codes (ECC) can detect and correct errors by adding redundancy to data, forming a codeword [6]–[10]. A common baseline is Single-Error Correction, Double-Error Detection (SEC–DED), which corrects any single-bit error and detects (but cannot correct) any double-bit error in a codeword.

DRAM errors often appear as *bursts* aligned with the device's internal organization (e.g., I/O pins, sub-wordline). In such settings, non-binary, *symbol*-based ECC can be more efficient than bit-oriented codes: by grouping a spatially clustered error into one symbol, the decoder repairs the entire cluster as a single unit, reducing the redundancy needed for a given level of protection.

Reed–Solomon (RS) [11] codes are a canonical family of symbol-based codes. An RS(n, k) code protects k data symbols with (n−k) parity symbols to form an n-symbol codeword. The decoder can correct up to ⌊(n − k)/2⌋ arbitrary symbol errors, provided the code length satisfies n ≤ 2 <sup>m</sup> − 1, where m is the symbol size. In practice, memory word sizes rarely match the full code length, so systems use *shortened* RS codes: some leading symbol positions are conceptually fixed (not stored/transmitted), and encoding/decoding proceed as usual over the remaining positions. Shortening preserves the same error-correction capability for the active symbols; corrections in the omitted positions are reported as error detections instead of silent miscorrections.

RangeGuard uses two RS configurations: (1) a 4-bit RS(12, 8), providing *Double-Symbol Correction (DSC)* for 4-bit RID symbols, and (2) an 8-bit RS(10, 8), providing *Single-Symbol Correction (SSC)* for 8-bit RID symbols. Both consume exactly 16 parity bits per 256-bit block, matching the HBM budget.

