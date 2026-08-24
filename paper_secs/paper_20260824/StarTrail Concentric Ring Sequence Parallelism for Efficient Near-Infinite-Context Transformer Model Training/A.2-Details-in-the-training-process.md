# A.2 Details in the training process

Apart from the attention process described in Algorithm [1,](#page-4-0) we would like do provide a few other details to be more comprehensive. First, before the concentric ring attention, we need to initialize the the Keys and Values and determine each GPU's position within the rings.

Initially, for the setup stage, it is essential to establish the sub-rings by rearranging the activation positions. Specifically, during forward propagation, the queries do not require rearrangement; however, the keys and values must be transmitted to their corresponding positions in the ring prior to commencing the loop. As illustrated in Figure [6](#page-5-0) and algorithn, this initialization ensures that each team member holds a different shard of keys and values. Moreover, it guarantees that no two teams within the same ring possess identical keys and values. Initially, for the setup stage, it is essential to establish the sub-rings by rearranging the activation positions. Specifically, during forward propagation, the queries do not require rearrangement; however, the keys and values must be transmitted to their corresponding positions in the ring prior to commencing the loop. As illustrated in Figure [6](#page-5-0) and algorithm, this initialization ensures that each team member holds a different shard of keys and values. Moreover, it guarantees that no two teams within the same ring possess identical keys and values.

### Algorithm 2 get\_init\_send()

Require: inter-team rank rt, intra-team rank ra, inter-team dimension dt, intra-team dimension d<sup>a</sup>

- 1: team group size = d<sup>t</sup> / d<sup>a</sup>
- 2: target team group rank = r<sup>a</sup>
- 3: target team = target team group rank \* team group size + r<sup>t</sup> // d<sup>a</sup>
- 4: target device intra-team rank = r<sup>t</sup> % d<sup>a</sup>
- 5: target global rank = target team \* d<sup>a</sup> + target device intra-team rank
- 6: return target global rank

After the initialization of activations, we can set up the rings by providing the GPUs their last and next GPU within their rings, as is described in Algorithm [3.](#page-21-0)

