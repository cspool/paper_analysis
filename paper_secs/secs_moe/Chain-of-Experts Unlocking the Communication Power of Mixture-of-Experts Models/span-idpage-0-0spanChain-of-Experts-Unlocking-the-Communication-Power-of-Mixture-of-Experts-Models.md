# <span id="page-0-0"></span>Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models

Zihan Wang1\*, Rui Pan2\*, Jiarui Yao<sup>2</sup> , Robert Csord ´ as´ 3 , Linjie Li<sup>4</sup> , Lu Yin<sup>5</sup> Jiajun Wu<sup>3</sup> , Tong Zhang<sup>2</sup> , Manling Li1† , Shiwei Liu6†

<sup>1</sup>Northwestern University <sup>2</sup>University of Illinois Urbana-Champaign <sup>3</sup>Stanford University <sup>4</sup>University of Washington <sup>5</sup>University of Surrey <sup>6</sup>University of Oxford

## Abstract

We propose Chain-of-Experts (CoE), a new Mixture-of-Experts (MoE) architecture that introduces sequential expert communication within each layer. Unlike traditional MoE models, where experts operate independently in parallel, CoE processes tokens iterative across a chain of experts inside a layer. To support dynamic expert selection across iterations, CoE employs a dedicated router at each iteration step within a layer. This design allows tokens to re-evaluate and select different experts during each iteration, rather than being statically assigned. As a result, CoE introduces a flexible routing mechanism that increases the diversity of expert combinations and enriches the model's representational capacity. CoE demonstrates improved performance under fixed compute: on Math reasoning tasks, it reduces validation loss from 1.20 to 1.12 compared to a standard MoE. Beyond performance, CoE offers a new *scaling axis*—depth through expert iteration—which complements conventional width/depth scaling. For example, using 2× iterations matches the performance of 3× expert selections (in width), while reducing memory usage by 17.6–42% relative to other scaling strategies. Our analysis reveals that CoE's benefits stem from its iterative residual structure and enhanced expert specialization empowered by iterative routing, which together unlock more expressive representations. Code is available at <https://github.com/ZihanWang314/coe>.

