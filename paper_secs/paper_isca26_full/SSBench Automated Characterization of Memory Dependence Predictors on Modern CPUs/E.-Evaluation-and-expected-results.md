# *E. Evaluation and expected results*

SSBench can be started by simply running:

```
$ python main.py -c <core-id>
```

Where <core-id> is the core ID that SSBench is bound to. SSBench automatically detects the underlying architecture. Currently supported platforms include Intel, AMD, Arm (Cortex / Neoverse) and Apple Silicon. The architecture can also be specified manually:

```
$ python main.py -c <core-id> -a <arch>
```

Where <arch> is one of the following strings: "intel", "amd", "arm", "apple" or "neoverse".

If root privileges are unavailable, use -u to avoid sudo requirement:

```
$ python main.py -c <core-id> -u
```

It is worth noting that without root access, physical address mapping may be inaccurate, which can affect hash function inference in some MDP designs.

The characterization results will be generated in the file data/characterization.json. Each MDP characterization is represented as a structured dictionary. The format of the dictionary is detailed in the README.md file of the artifact. The artifact also includes some example outputs generated on different CPUs. For example, the output of the Apple M4 performance core is as follows:

```
{
"type_time_dict": {"S": [[189, 189]], "B":
   [[274, 275]], "R": [[299, 299]], "b1":
   231, "b2": 287, "b3": 448},
"exist": true,
"state machine": {
    "store_exist": true,
    "store_sm": [0, -1, 7, 1, 7, 1, 0, 0],
    "load_exist": false,
    "load_sm": [],
    "store_seq": "4a"},
"hash": {
    "hash_func": [[2], [3], [4], [5], [6],
        [13], [7, 14], [8, 15], [9, 16], [10,
        17], [11, 18], [12, 19]],
    "hash_va": true,
    "hash_seq": "4a",
    "expected_sm_val": 7},
"org": {
    "eviction_set_size": 69,
    "confidence_eviction_set_size": 1.0,
    "replacement_policy": "lru",
    "confidence_replacement_policy": 1.0,
    "size": 69,
    "set": 1,
    "set_index": 0,
    "confidence_parameters": 1.0},
"time": {"exist": 20.817784070968628, "sm":
   19.586023807525635, "hash":
   340.8100039958954, "org":
   1086.165694952011}
}
```