# 6 Conclusion

We introduce AdaRETAKE, a training-free method for adaptive redundancy reduction in MLLMs. By dynamically allocating compression ratios across frames and model layers, AdaRETAKE achieves more efficient video token compression. This allows us to scale up to more frames and extract valuable information within the same computational budget. Integrated into state-ofthe-art MLLMs, AdaRETAKE enables processing of up to 2,048 frames and outperforms existing methods on benchmarks such as VideoMME,

MLVU, LongVideoBench, and LVBench by a large margin.

## 7 Limitations

While AdaRETAKE can be integrated into most MLLMs, it may also inherit their inherent limitations, such as factual inaccuracies, biases, and hallucinations.

## 8 Acknowledgment

This work was supported in part by the National Natural Science Foundation of China(Grant Nos. 62376069 and 62236003), in part by the Young Elite Scientists Sponsorship Program by CAST (Grant No. 2023QNRC001), in part by Guangdong Basic and Applied Basic Research Foundation (Grant No. 2024A1515012027), in part by Jiangsu Science and Technology Major Program (Grant No. BG2024041), and in part by the Shenzhen Science and Technology Program (Grant Nos. KQTD20240729102207002 and ZDSYS20230626091203008).

