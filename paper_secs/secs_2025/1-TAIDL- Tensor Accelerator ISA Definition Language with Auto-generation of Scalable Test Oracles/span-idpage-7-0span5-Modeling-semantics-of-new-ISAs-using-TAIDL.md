# <span id="page-7-0"></span>5 Modeling semantics of new ISAs using TAIDL

We discuss key language properties that enable and assist architects to define the semantics of new tensor accelerator ISAs in TAIDL.

#### <span id="page-7-4"></span>5.1 Theoretically complete

Theoretically, the XLA-HLO operator set is Turing-complete. XLA-HLO supports an unbounded number of dimensions and an unbounded dimension sizes. It also includes select<sup>4</sup> and while<sup>5</sup> operators, allowing for conditional branching and unbounded loops. This makes XLA-HLO a superset of FLooP [55], a theoretical programming language that is proven to be a Turing-complete language. Therefore, TAIDL can express all computable functions.

