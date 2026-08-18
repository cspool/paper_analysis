# *E. Metadata Overhead of DynFP*

DynFP uses a 4-bit and 1-bit per-group format index for weights and K/V cache, respectively, together with an 8-bit per-group scaling factor. For typical 4-bit weight quantization with group size 32, this leads to an effective bit-width of 4.375 bits, only a 2.9% increase (0.125 bits) compared to MXFP4 [\[36\]](#page-14-19). Metadata is stored in main memory and loaded with weight tensors. At runtime, the Unified Format Converter decodes the format index within 1 cycle to select E/M settings and special-case mappings (Figure. [13\)](#page-7-2), while scaling factors are held in dedicated 8-bit registers in the GEMM array's Rescale unit, contributing only 0.73% of GEMM area.

