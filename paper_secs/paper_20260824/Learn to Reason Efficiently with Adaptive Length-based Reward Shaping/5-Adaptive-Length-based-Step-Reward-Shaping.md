# 5 Adaptive Length-based Step Reward Shaping

### 5.1 Design Principles

We highlight two key limitations not addressed in the design of LASER: (1) LASER requires specifying a fixed target length prior to training; however, as the model evolves during training, the optimal response length may also change and should ideally adapt dynamically. (2) Additionally, different questions demand reasoning traces of varying lengths—simple questions may be effectively addressed with shorter reasoning, while more complex questions benefit from longer, more detailed deliberation.

Therefore, we extend LASER to be Dynamic and Difficulty-aware, which we term as LASER-D. Rather than using a single fixed target length, our approach dynamically adjusts the target length throughout training and tailors it to questions of varying difficulty. Concretely, LASER-D *decouples the target length hyperparameter across different queries, allowing distinct target lengths to be assigned to various queries. Moreover, these target length hyperparameters are dynamically adjusted throughout training.*

We separate queries into three buckets of easy, medium, and hard difficulty levels, based on the correctness rates within the rollout batch – for each question, we have k rollouts and use thresholds k/3 and 2k/3 to separate them. As such, we have three distinct target length hyperparameters for these three query groups. Notably, we perform difficulty assessment for the queries during realtime RL training and use the training rollout batch, thus it only incurs negligible overhead on the computation. Being dynamic and difficulty-aware, one challenge raised is how to set the dynamic processes of the decoupled target length hyperparameters. Next, we introduce an *automatic* adapting mechanism, to adapt them without any manual intervention.

#### <span id="page-5-1"></span>5.2 Automatic Adapting Mechanism

LASER-D is driven by an automatic adapting mechanism that periodically evaluates and adjusts the target length parameters (L<sup>A</sup> in Table [2\)](#page-4-1) for each difficulty level. Specifically, we first extract a small monitoring dataset D<sup>M</sup> (e.g., 500 samples) from training data that mirrors the distribution of the training data. Every N training steps (e.g., 20), our approach searches and sets the target length hyperparameters based on this monitoring dataset.

Denote the three-class difficulty level of a query as d, to determine the target length hyperparameters, we propose a metric called Expected Correct Responses (ECR), which estimates how many complete, correct responses we can expect for each difficulty level given response length limits. Formally, we sample K responses for each query in the monitoring set,[2](#page-5-0) and ECR is computed as

$$ECR_d = P_{l,d} \cdot |C_d| \tag{3}$$

<span id="page-5-0"></span>Practically, K is set to be the same as the rollout size used during training, in order to maintain consistency with the training scenario.

where Pl,d is the coverage ratio (proportion of responses that fit within a given token length l). The value |Cd| is fixed for each difficulty group. Since we use the ratio of correct responses within each rollout group to determine the difficulty level, there is a minimum number of correct responses for each group (e.g., 6, 3, and 1 correct responses for easy, medium, and hard levels, respectively, when K = 8). We set |Cd| as these minimum values for each group.

The monitoring module enumerates potential target lengths from the lower bound target length L<sup>T</sup> tokens up to the maximum context window (16,384 tokens) with an interval of I, computing coverage ratios Pl,d for each length. We select the smallest target length as the adaptive target length L<sup>A</sup> satisfying ECR<sup>d</sup> ≥ 1 for each difficulty level d, ensuring at least one complete and correct response.

Intuitively, this mechanism sets the target length as the minimal generation length such that at least one rollout response is expected to be correct. This approach is reasonable because generating sequences shorter than this length would likely be detrimental, as correct responses are less probable. Conversely, generating longer sequences may be redundant, since correct responses can already be obtained with a shorter generation length.

Dynamic and Difficulty-Aware Reward During training, we apply these monitoring-derived parameters to implement dynamic and difficulty-aware rewards. Each training question's difficulty level is determined using the same classification method described earlier. Easier questions receive smaller target lengths (i.e. smaller scaling factor β), while harder questions receive larger ones (i.e. larger scaling factor β). Since monitoring runs every N steps, the difficulty-dependent target lengths are automatically adapted to the evolving policy model.

Computational Efficiency This automatic adapting mechanism adds minimal computational overhead. By using a small monitoring dataset and evaluating only periodically, our method increases computation by just 3.5% in our experiments.

