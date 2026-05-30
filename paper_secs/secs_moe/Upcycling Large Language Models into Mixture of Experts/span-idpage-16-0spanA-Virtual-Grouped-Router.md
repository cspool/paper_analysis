# <span id="page-16-0"></span>A Virtual Grouped Router

The following pseudo code illustrates our virtual grouped router approach for upcycling into E2G2T2:

```
# sharding FFN into expert shards G =2
FFN = [ FFN_0 , FFN_1 ]
# copying to form multiple experts ; E =2
experts = [ FFN_0 , FFN_1 ,
            FFN_0 , FFN_1 ]
# random initialized router
router_prob = tensor ([0.4 , 0.2 ,
                         0.3 , 0.1])
router_top2 = tensor ([0.4 , 0.0 ,
                         0.3 , 0.0])
# For dense model
FFN ( x ) = FFN_0 ( x ) + FFN_1 ( x )
# In case of MoE
FFN_moe ( x ) = router_top2@experts = 0.4 FFN_0 ( x ) + 0.3 FFN_0 ( x ) ̸=FFN ( x )
```

```
# virtual group - initialize every group with same weights
router_prob = tensor ([0.3 , 0.3 ,
                        0.2 , 0.2])
router_top2 = tensor ([0.3 , 0.3 ,
                        0.0 , 0.0])
# one of each FFN shard is guaranteed to be selected
FFN_moe ( x ) = 0.3 FFN_0 + 0.3 FFN_1 ≈ FFN ( x ) / 4
```