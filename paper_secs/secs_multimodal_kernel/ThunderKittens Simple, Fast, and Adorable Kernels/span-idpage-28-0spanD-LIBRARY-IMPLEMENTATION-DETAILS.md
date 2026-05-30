# <span id="page-28-0"></span>D LIBRARY IMPLEMENTATION DETAILS

This section provides additional implementation details for THUNDERKITTENS.

### D.1 TILE DATA STRUCTURES

The core primitive in THUNDERKITTENS is the tile data structure as introduced in section [3.1.](#page-4-1) Tiles exist at the shared memory and register memory levels of the GPU hierarchy, and are created with multiples of 16 × 16 in dimension.

Precision THUNDERKITTENS is extensible across data types: FP32, FP16, BF16, and FP8. Designing a unified tile data structure that seamlessly supports different data types is challenging for two reasons:

- 1. Each data type requires using different memory layouts both at the shared and register memory levels, in order to use specialized hardware instructions like tensor cores. [3](#page-28-1)
- 2. Each data type uses a different amount of space, meaning that the 16 × 16 tile that the user sees could contain a fixed number of bits, fixed number of elements, or some other option. Ideally, we can store elements in fully packed formats (e.g., bf16 2 for bf16, e4m3 8x4 for e4m3 FP8).

We let the users think in terms of the number of elements per tile. When the user defines a 16 × 32 tile for instance, we store this as 16 × 8 packed elements of e4m3 8x4 or 16 × 16 packed elements of bf16 2 in registers. In the library, we define and operate an *underlying tile width* for the tiles to hide this complexity from the user. Taking care of differences across data types at tile data structure level, we can then use the exact same library functions (e.g., mma, exp, cumsum) across tiles of different data types, preserving the simplicity of the library.

Padding Some AI workloads require shapes that are *not* multiples of 16. THUNDERKITTENS provides mechanisms to support these workloads, too, without compromising performance on hardware-friendly workloads.

- 1. Loads & Stores. THUNDERKITTENS loads and stores take an optional template arguments for whether to assume tensors are multiples of 16. If not, each load or store is preceded by a check to ensure that it is in-bounds. Out-of-bound loads are filled with zeros; out-of-bound stores are not performed. For safety, these checks are enabled by default; however, they are not free, and they can also be disabled by setting the appropriate template flag. This abides by TK's philosophy of "extensible, but with good defaults." For TMA loads and stores, we use the built-in hardware padding features (that is, out-of-bounds accesses are automatically filled in with zeros).
- 2. Fills & Masks. In addition to preventing illegal memory accesses, one often needs to alter data within a tile to prevent accidental computations. THUNDERKITTENS provides functionality for this, too, in the form of six functions: top fill, bottom fill, left fill, right fill, triu, and tril, which respectively fill the top, bottom, left, right, upper triangle, or lower triangle of tiles. In our experience, these functions have proven sufficient for all kernels considered.

