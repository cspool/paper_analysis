# 7 Conclusion

In this paper, we identify the growing All-to-All overhead in largescale distributed DL computing and explore algorithm and scheduling optimizations for fault-free and fault-tolerant communication. For fault-free torus networks, we propose HalfRing to improve single-dimension transmission efficiency, and DimRotation to enhance overall bandwidth utilization. For a torus with link failures, we introduce FoldedRing for basic fault-tolerance and further propose MATE, which leverages multi-dimensional links to accelerate

communication on faulty rings. Evaluation shows that HalfRing and DimRotation achieve average speedups of 1.56× and 1.45×, respectively, and up to 2.28× when combined. Under a single link failure, MATE yields a 1.37× speedup over the fault-free baseline. Our approaches can also achieve respective speedups of 1.57× and 1.61× compared to Google's routing methods.

