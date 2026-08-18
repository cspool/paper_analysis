# Cerberus: Cross-Layer ECC Co-Design for Robust and Efficient Memory Protection

Junhwan Kim\* ¶, Seunghyun Kim† ¶, Yesin Ryu‡ , Saeid Gorgin§ , and Jungrae Kim\* \*Dept. of Electrical and Computer Engineering, †Dept. of Artificial Intelligence, ‡Dept. of Semiconductor and Display Engineering, Sungkyunkwan University, Suwon, Republic of Korea §School of Physics, Engineering and Computer Science, University of Hertfordshire, Hatfield, United Kingdom {june9918, kshyun0815, seleneyou}@skku.edu, s.gorgin@herts.ac.uk, dale40@skku.edu

*Abstract*—As DRAM scales to higher density and I/O speeds, ensuring data correctness becomes increasingly difficult. Industry has responded with a three-layer stack—on-die ECC (O-ECC), link ECC (L-ECC), and system ECC (S-ECC)—but these layers have evolved independently, often duplicating redundancy, leaving coverage gaps, and occasionally interfering.

We propose *Cerberus*, a cross-layer ECC co-design that unifies protection across device, link, and system while preserving each layer's native role. At its core is an *Encode-Once, Decode-Many (EODM)* architecture: the controller performs a single encoding whose redundancy is reused by L-ECC for immediate write-path detection/retry, by O-ECC for in-device repair on reads, and by S-ECC for strong end-to-end recovery. Cerberus jointly designs complementary parity/syndrome structures, orders decoders and allocates the correction budget to prevent miscorrection amplification, and enables selective correction under tight redundancy constraints. Our evaluations show improved resilience to clustered and peripheral faults while reducing redundancy overhead, underscoring the importance of coordinated crosslayer protection for next-generation memory systems, such as custom HBMs.

*Index Terms*—DRAM, Reliability, ECC, Multi-layer ECC

# Cerberus: Cross-Layer ECC Co-Design for Robust and Efficient Memory Protection

Junhwan Kim\* ¶, Seunghyun Kim† ¶, Yesin Ryu‡ , Saeid Gorgin§ , and Jungrae Kim\* \*Dept. of Electrical and Computer Engineering, †Dept. of Artificial Intelligence, ‡Dept. of Semiconductor and Display Engineering, Sungkyunkwan University, Suwon, Republic of Korea §School of Physics, Engineering and Computer Science, University of Hertfordshire, Hatfield, United Kingdom {june9918, kshyun0815, seleneyou}@skku.edu, s.gorgin@herts.ac.uk, dale40@skku.edu

*Abstract*—As DRAM scales to higher density and I/O speeds, ensuring data correctness becomes increasingly difficult. Industry has responded with a three-layer stack—on-die ECC (O-ECC), link ECC (L-ECC), and system ECC (S-ECC)—but these layers have evolved independently, often duplicating redundancy, leaving coverage gaps, and occasionally interfering.

We propose *Cerberus*, a cross-layer ECC co-design that unifies protection across device, link, and system while preserving each layer's native role. At its core is an *Encode-Once, Decode-Many (EODM)* architecture: the controller performs a single encoding whose redundancy is reused by L-ECC for immediate write-path detection/retry, by O-ECC for in-device repair on reads, and by S-ECC for strong end-to-end recovery. Cerberus jointly designs complementary parity/syndrome structures, orders decoders and allocates the correction budget to prevent miscorrection amplification, and enables selective correction under tight redundancy constraints. Our evaluations show improved resilience to clustered and peripheral faults while reducing redundancy overhead, underscoring the importance of coordinated crosslayer protection for next-generation memory systems, such as custom HBMs.

*Index Terms*—DRAM, Reliability, ECC, Multi-layer ECC

