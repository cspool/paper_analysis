# **F List of Experimental Models**

The detailed configurations for all experiments conducted in this study are presented in Tables [7](#page-32-0) (activation ratio), Tables [8](#page-33-0) (expert granularity), Tables [9](#page-34-0) (shared experts), Tables [10](#page-35-0) (layer arrangement), and Tables [11](#page-36-0) (compute allocation between attention and FFNs).

<span id="page-32-0"></span>Table 7 Experimental configurations for the expert activation ratio analysis. Within each group, the number of activated experts (*E<sup>a</sup>* = 2) is fixed, while the total number of experts (*E*) is varied to study the effect of the activation ratio.

| nlayers | dmodel | dexpert | nheads | nkv_head | E                        | Ea | Es | η       | B   | Max training FLOPs |
|---------|--------|---------|--------|----------|--------------------------|----|----|---------|-----|--------------------|
| 8       | 384    | 320     | 8      | 2        | [2,4,8,16,32,64,128,256] | 2  | 1  | 1.52e-3 | 98  | 2e18               |
| 8       | 512    | 512     | 8      | 2        | [2,4,8,16,32,64,128,256] | 2  | 1  | 1.31e-3 | 147 | 6e18               |
| 10      | 640    | 640     | 10     | 2        | [2,4,8,16,32,64,128,256] | 2  | 1  | 1.11e-3 | 228 | 2e19               |
| 14      | 768    | 768     | 12     | 4        | [2,4,8,16,32,64,128,256] | 2  | 1  | 9.5e-4  | 342 | 6e19               |
| 16      | 1024   | 1024    | 16     | 4        | [2,4,8,16,32,64,128,256] | 2  | 1  | 8.1e-4  | 531 | 2e20               |
| 22      | 1280   | 1280    | 20     | 4        | [2,4,8,16,32,64,128,256] | 2  | 1  | 7.0e-4  | 795 | 6e20               |

<span id="page-33-0"></span>Table 8 Experimental configurations for the expert granularity analysis. Within each group, the base model architecture is fixed while the MoE configuration (total experts *E*, activated experts *Ea*, shared experts *Es*, and expert dimension *d*expert) is varied to study the effect of granularity.

| nlayers | dmodel | nheads | E                                     | Ea                           | Es                         | dexpert                                 | B   | η       | Max training FLOPs |
|---------|--------|--------|---------------------------------------|------------------------------|----------------------------|-----------------------------------------|-----|---------|--------------------|
| 8       | 384    | 8      | 64<br>128<br>192<br>256<br>384<br>512 | 2<br>4<br>6<br>8<br>12<br>16 | 1<br>2<br>3<br>4<br>6<br>8 | 384<br>192<br>128<br>96<br>64<br>48     | 98  | 1.52e-3 | 2e18               |
| 8       | 512    | 8      | 64<br>128<br>192<br>256<br>384<br>512 | 2<br>4<br>6<br>8<br>12<br>16 | 1<br>2<br>3<br>4<br>6<br>8 | 512<br>256<br>170<br>128<br>85<br>64    | 147 | 1.31e-3 | 6e18               |
| 10      | 640    | 10     | 64<br>128<br>192<br>256<br>384<br>512 | 2<br>4<br>6<br>8<br>12<br>16 | 1<br>2<br>3<br>4<br>6<br>8 | 640<br>320<br>213<br>160<br>106<br>80   | 228 | 1.11e-3 | 2e19               |
| 14      | 768    | 12     | 64<br>128<br>192<br>256<br>384<br>512 | 2<br>4<br>6<br>8<br>12<br>16 | 1<br>2<br>3<br>4<br>6<br>8 | 768<br>384<br>256<br>192<br>128<br>96   | 342 | 9.5e-4  | 6e19               |
| 16      | 1024   | 16     | 64<br>128<br>192<br>256<br>384<br>512 | 2<br>4<br>6<br>8<br>12<br>16 | 1<br>2<br>3<br>4<br>6<br>8 | 1024<br>512<br>341<br>256<br>170<br>128 | 531 | 8.1e-4  | 2e20               |
| 22      | 1280   | 20     | 64<br>128<br>192<br>256<br>384<br>512 | 2<br>4<br>6<br>8<br>12<br>16 | 1<br>2<br>3<br>4<br>6<br>8 | 1280<br>640<br>426<br>320<br>213<br>160 | 795 | 7.0e-4  | 6e20               |

<span id="page-34-0"></span>Table 9 Experimental configurations for the shared expert ratio analysis. Within each group, we fix the total number of experts (*E* = 256) and the total number of activated pathways (*E<sup>a</sup>* + *E<sup>s</sup>* = 12), while varying the ratio between specialized experts (*Ea*) and shared experts (*Es*) to study its impact on performance.

| nlayers | dmodel | nheads | E                                      | Ea                           | Es                          | dexpert                                | B   | η       | Max training FLOPs |
|---------|--------|--------|----------------------------------------|------------------------------|-----------------------------|----------------------------------------|-----|---------|--------------------|
| 8       | 384    | 8      | 256<br>256<br>256<br>256<br>256<br>256 | 2<br>4<br>6<br>8<br>11<br>12 | 10<br>8<br>6<br>4<br>1<br>0 | 96<br>96<br>96<br>96<br>96<br>96       | 98  | 1.52e-3 | 2e18               |
| 8       | 512    | 8      | 256<br>256<br>256<br>256<br>256<br>256 | 2<br>4<br>6<br>8<br>11<br>12 | 10<br>8<br>6<br>4<br>1<br>0 | 128<br>128<br>128<br>128<br>128<br>128 | 147 | 1.31e-3 | 6e18               |
| 10      | 640    | 10     | 256<br>256<br>256<br>256<br>256<br>256 | 2<br>4<br>6<br>8<br>11<br>12 | 10<br>8<br>6<br>4<br>1<br>0 | 160<br>160<br>160<br>160<br>160<br>160 | 228 | 1.11e-3 | 2e19               |
| 14      | 768    | 12     | 256<br>256<br>256<br>256<br>256<br>256 | 2<br>4<br>6<br>8<br>11<br>12 | 10<br>8<br>6<br>4<br>2<br>0 | 192<br>192<br>192<br>192<br>192<br>192 | 342 | 9.5e-4  | 6e19               |
| 16      | 1024   | 16     | 256<br>256<br>256<br>256<br>256<br>256 | 2<br>4<br>6<br>8<br>11<br>12 | 10<br>8<br>6<br>4<br>1<br>0 | 256<br>256<br>256<br>256<br>256<br>256 | 531 | 8.1e-4  | 2e20               |
| 22      | 1280   | 20     | 256<br>256<br>256<br>256<br>256<br>256 | 2<br>4<br>6<br>8<br>11<br>12 | 10<br>8<br>6<br>4<br>1<br>0 | 320<br>320<br>320<br>320<br>320<br>320 | 795 | 7.0e-4  | 6e20               |

<span id="page-35-0"></span>Table 10 Experimental configurations for the arrangement of MoE and dense layers analysis. Within each group, the total number of layers is fixed at 60, while the mix of dense layers (*n*dense\_layers) and MoE layers (*n*moe\_layers) is varied to study the impact of their ratio and placement on performance.

| nlayers | ndense_layers    | nmoe_layers          | dmodel | df f n | nheads | E  | Ea | Es | dexpert | B   | η       | Max training FLOPs |
|---------|------------------|----------------------|--------|--------|--------|----|----|----|---------|-----|---------|--------------------|
| 60      | 0<br>1<br>2      | 60<br>59<br>58       | 384    | 1280   | 8      | 64 | 2  | 1  | 384     | 98  | 1.52e-3 | 2e18               |
|         | 3                | 57                   |        |        |        |    |    |    |         |     |         |                    |
|         | 0                | 60                   |        |        |        |    |    |    |         |     |         |                    |
| 60      | 1                | 59                   | 512    | 2048   | 8      | 64 | 2  | 1  | 512     | 147 | 1.31e-3 | 6e18               |
| 2<br>3  |                  | 58<br>57             |        |        |        |    |    |    |         |     |         |                    |
| 60      | 0<br>1<br>2<br>3 | 60<br>59<br>58<br>57 | 640    | 2560   | 10     | 64 | 2  | 1  | 640     | 228 | 1.11e-3 | 2e19               |
| 60      | 0<br>1<br>2<br>3 | 60<br>59<br>58<br>57 | 768    | 3072   | 12     | 64 | 2  | 1  | 768     | 342 | 9.5e-4  | 6e19               |
| 60      | 0<br>1<br>2<br>3 | 60<br>59<br>58<br>57 | 1024   | 4096   | 16     | 64 | 2  | 1  | 1024    | 531 | 8.1e-4  | 2e20               |
| 60      | 0<br>1<br>2<br>3 | 60<br>59<br>58<br>57 | 1280   | 5120   | 20     | 64 | 2  | 1  | 1280    | 795 | 7.0e-4  | 6e20               |

<span id="page-36-0"></span>Table 11 Experimental configurations for analyzing the compute allocation between attention and FFNs. Within each group, the core MoE structure is held constant, while we systematically vary the model's hidden dimension (*d*model) and the expert dimension (*d*expert) to explore the optimal trade-off in compute allocation between the attention mechanism and the FFN experts.

| layers | dmodel | dexpert | nheads | nkv_head | E  | Es | Ea | η       | B   | Max training FLOPs |
|--------|--------|---------|--------|----------|----|----|----|---------|-----|--------------------|
| 8      | 352    | 450     | 8      | 2        | 64 | 1  | 2  | 1.52e-3 | 96  | 2e18               |
| 8      | 368    | 380     | 8      | 2        | 64 | 1  | 2  | 1.52e-3 | 96  | 2e18               |
| 8      | 384    | 320     | 8      | 2        | 64 | 1  | 2  | 1.52e-3 | 96  | 2e18               |
| 8      | 400    | 260     | 8      | 2        | 64 | 1  | 2  | 1.52e-3 | 96  | 2e18               |
| 8      | 416    | 208     | 8      | 2        | 64 | 1  | 2  | 1.52e-3 | 96  | 2e18               |
| 8      | 480    | 626     | 8      | 2        | 64 | 1  | 2  | 1.31e-3 | 160 | 6e18               |
| 8      | 512    | 512     | 8      | 2        | 64 | 1  | 2  | 1.31e-3 | 160 | 6e18               |
| 8      | 544    | 410     | 8      | 2        | 64 | 1  | 2  | 1.31e-3 | 160 | 6e18               |
| 8      | 560    | 364     | 8      | 2        | 64 | 1  | 2  | 1.31e-3 | 160 | 6e18               |
| 8      | 576    | 320     | 8      | 2        | 64 | 1  | 2  | 1.31e-3 | 160 | 6e18               |
| 10     | 600    | 766     | 10     | 2        | 64 | 1  | 2  | 1.11e-3 | 224 | 2e19               |
| 10     | 640    | 640     | 10     | 2        | 64 | 1  | 2  | 1.11e-3 | 224 | 2e19               |
| 10     | 680    | 528     | 10     | 2        | 64 | 1  | 2  | 1.11e-3 | 224 | 2e19               |
| 10     | 700    | 476     | 10     | 2        | 64 | 1  | 2  | 1.11e-3 | 224 | 2e19               |
| 10     | 740    | 380     | 10     | 2        | 64 | 1  | 2  | 1.11e-3 | 224 | 2e19               |
| 14     | 696    | 988     | 12     | 4        | 64 | 1  | 2  | 9.5e-3  | 320 | 6e19               |
| 14     | 768    | 768     | 12     | 4        | 64 | 1  | 2  | 9.5e-3  | 320 | 6e19               |
| 14     | 816    | 642     | 12     | 4        | 64 | 1  | 2  | 9.5e-3  | 320 | 6e19               |
| 14     | 840    | 584     | 12     | 4        | 64 | 1  | 2  | 9.5e-3  | 320 | 6e19               |
| 14     | 888    | 474     | 12     | 4        | 64 | 1  | 2  | 9.5e-3  | 320 | 6e19               |
| 16     | 896    | 1378    | 16     | 4        | 64 | 1  | 2  | 8.1e-3  | 512 | 2e20               |
| 16     | 1024   | 1024    | 16     | 4        | 64 | 1  | 2  | 8.1e-3  | 512 | 2e20               |
| 16     | 1088   | 876     | 16     | 4        | 64 | 1  | 2  | 8.1e-3  | 512 | 2e20               |
| 16     | 1152   | 742     | 16     | 4        | 64 | 1  | 2  | 8.1e-3  | 512 | 2e20               |
| 16     | 1184   | 680     | 16     | 4        | 64 | 1  | 2  | 8.1e-3  | 512 | 2e20               |
| 22     | 1120   | 1686    | 20     | 4        | 64 | 1  | 2  | 7.0e-3  | 768 | 6e20               |
| 22     | 1280   | 1280    | 20     | 4        | 64 | 1  | 2  | 7.0e-3  | 768 | 6e20               |
| 22     | 1360   | 1110    | 20     | 4        | 64 | 1  | 2  | 7.0e-3  | 768 | 6e20               |
| 22     | 1440   | 956     | 20     | 4        | 64 | 1  | 2  | 7.0e-3  | 768 | 6e20               |
| 22     | 1520   | 816     | 20     | 4        | 64 | 1  | 2  | 7.0e-3  | 768 | 6e20               |