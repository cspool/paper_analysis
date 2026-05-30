# <span id="page-4-2"></span>(b) Lightweight expert placement.

Fig. 6. The comparison of a traditional and lightweight expert placement. The load is imbalanced in traditional expert placement. In a lightweight one, each expert is placed into necessary devices to balance the load. The Trans and Agg primitives are involved to communicate their parameters and gradients respectively.

Trans is first launched to transfer the parameters. After that, each device contains the parameters of some expert, thus its local inputs routed to these experts could be computed locally. After the backward computation, the gradients of an expert could be generated in several devices. As each device only maintains the optimizer states of one expert, a Agg primitive is launched to aggregate gradients of each expert to its original device. This design has two advantages: 1) Only part of the model states are communicated. 2) The model states are only communicated among a subset of devices.

Fig 6 illustrates a comparison of a traditional and lightweight expert placement. As shown in Fig. 6a, 5, 2, and 2 inputs are routed to  $E_0$ ,  $E_1$ , and  $E_2$  respectively. After the A2A communication, three devices are responsible for the computation of 5, 2, and 2 inputs as each of the devices only contains parameters of a distinct expert (e.g., Dev. 0 contains  $E_0$ 's parameters), resulting in an imbalanced load among devices. Fig. 6b shows a balanced load achieved by the lightweight expert placement. Experts are mapped to devices according to the routing results produced by the gate network. Parameters of  $E_0$  are sent from Dev. 0 only to Dev. 1 as inputs in Dev. 2 are not routed to  $E_0$ . Similarly, parameters of  $E_1$  are transferred to Dev. 0 and Dev. 1 for their expert computation. It maps experts to necessary devices and only communicates their parameters and gradients, effectively avoiding heavy model states transferring.

## B. Performance model

It's necessary to evaluate lightweight expert placements under various device loads. Therefore, the planner abstracts a performance model to estimate the execution time of a MoE layer employing a lightweight expert placement. Table II presents notations and descriptions used in the performance model.

After employing a lightweight expert placement, a MoE layer performs four A2A communication operations, one forward expert computation operation EFC, one backward computation operation EBC, one Trans operation, and one Agg operation. To accurately evaluate the execution time of the MoE layer, we establish our performance model according to the implementation of operations and hardware characteristics.

**A2A communication.** Tutel [5] presents an efficient A2A implementation used in the training of a MoE model. In this

TABLE II NOTATIONS

<span id="page-4-3"></span>

| Notation       | Description                                               |  |  |  |
|----------------|-----------------------------------------------------------|--|--|--|
| T              | Execution time of an operation                            |  |  |  |
| R              | Inputs received by a device from other devices            |  |  |  |
| $\overline{B}$ | Average communication bandwidth                           |  |  |  |
| H              | Inputs computed in a device                               |  |  |  |
| t              | Computation throughput                                    |  |  |  |
| s              | Number of selected experts should be transferred          |  |  |  |
| n              | Number of devices a selected expert not be transferred to |  |  |  |
| E              | Number of experts in a MoE layer                          |  |  |  |
| D              | Number of devices                                         |  |  |  |

implementation, devices use point-to-point(P2P) communication primitives to achieve the A2A communication operation. Based on this, we define the execution time of an A2A operation as below.

$$T_{A2A}(R) = \max_{i} \frac{R_i \cdot size(input)}{\overline{R}},$$
 (1)

where  $R_i$  is the total number of inputs received by device-i from other devices and size(input) is the size of a input.

**Expert computation.** Next, we formulate the duration of the forward and backward expert computation. In the expert computation procedure, the computations of devices are performed simultaneously. However, computations of different experts are launched sequentially in a device. To depict this characteristic, we define the execution time of FEC as

$$T_{FEC}(H) = \max_{i} \frac{H_i}{t},\tag{2}$$

where  $H_i$  is the number of inputs computed in device-i.

It is widely recognized that the time required for backward computation in DNN training is roughly double that of forward computation, which is the same for MoE model training. Therefore, we define the execution time of BEC as

$$T_{BEC}(H) = 2\max_{i} \frac{H_i}{t},\tag{3}$$

Trans and Agg primitives. Finally, we formulate the overhead of Trans and Agg primitives. The duration time of Trans and Agg primitives depends on two elements. The first element is the number of transferred experts, which determines communication rounds. The second element is the number of devices communicated in a primitive, which influences

the communication scales. Therefore, the TT rans(s, n) and TAgg(s, n) are defined as below.

$$T_{Trans}(s,n) = \frac{s * (D-n) * size(e_j.params)}{D * \overline{B}}, \quad (4)$$

$$T_{Agg}(s,n) = \frac{s * (D-n) * size(e_j.grads)}{D * \overline{B}},$$
 (5)

where the size(e<sup>j</sup> .params) and size(e<sup>j</sup> .grad) are the size of parameters and gradients for the j-th expert.

In summary, the overall execution time of the MoE layer with lightweight expert placement can be represented as

$$T'(R, H, s, n) = 4T_{A2A}(R) + 3T_{FEC}(H) + T_{Trans}(s, n) + T_{Agg}(s, n)$$
(6)

