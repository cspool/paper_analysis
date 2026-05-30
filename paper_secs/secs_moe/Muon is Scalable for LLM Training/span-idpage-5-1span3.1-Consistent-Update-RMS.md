# <span id="page-5-1"></span>3.1 Consistent Update RMS

As discussed in Sec [2.2,](#page-2-5) we aim to match the update RMS across all matrix parameters and also match it with that of AdamW. We experimented with two methods to control the Muon update RMS among parameters and compared them to a baseline that only maintains a consistent RMS with AdamW:

1. Baseline. We multiplied the update matrix by 0.2 · √ H (H is the model hidden size) to maintain a consistent update RMS with AdamW. Note that max(A, B) equals to H for most matrices.

$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t (0.2 \cdot \mathbf{O}_t \cdot \sqrt{H} + \lambda \mathbf{W}_{t-1})$$
(5)

2. Update Norm. We can directly normalize the updates calculated via Newton-Schulz iterations so its RMS strictly becomes 0.2;

$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t (0.2 \cdot \mathbf{O}_t / \text{RMS}(\mathbf{O}_t) + \lambda \mathbf{W}_{t-1})$$
(6)

3. Adjusted LR. For each update matrix, we can scale its learning rate by a factor of 0.2 · p max(A, B) based on its shape.

$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t (0.2 \cdot \mathbf{O}_t \cdot \sqrt{\max(A, B)} + \lambda \mathbf{W}_{t-1})$$
(7)

Analysis We designed experiments to illustrate the impact of Muon update RMS at an early training stage, because we observed that unexpected behaviors happened very quickly when training models at larger scale. We experimented with small scale 800M models as described in [3.2.](#page-5-0) The problem of inconsistent update RMS is more pronounced when the disparity between matrix dimensions increases. To highlight the problem for further study, we slightly modify the model architecture by replacing the Swiglu MLP with a standard 2-layer MLP, changing the shape of its matrix parameters from [H, 2.6H] to [H, 4H]. We evaluated the model's loss and monitored a few of its parameters' RMS, specifically, attention query (shape [H, H]) and MLP (shape [H, 4H]). We evaluated the model after training for 4B tokens out of a 20B-token schedule. From Table [1,](#page-5-2) we observed several interesting findings:

- 1. Both Update Norm and Adjusted LR achieved better performances than Baseline;
- 2. For the MLP weight matrix of shape [H, 4H], both Update Norm and Adjusted LR obtain a weight RMS that is roughly doubled comparing to Baseline. This is reasonable as p max(H, 4H)/ √ H = 2, so the update RMS of Update Norm and Adjusted LR is roughly two times of Baseline;
- 3. For the attention query weight matrix of shape [H, H], Update Norm still norms the update, while Adjusted LR does not because p max(H, H)/ √ H = 1. As a result, Adjusted LR results in a similar weight RMS as Baseline, but Update Norm has a larger weight rms similar to its MLP.

Based on these findings, we choose the Adjusted LR method for future experiments because it has lower cost.

