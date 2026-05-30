# <span id="page-19-0"></span>E Task Implementation

```
#def i ne GEMMs 2
   st r uct __al i gn__( 16) Task {
        const byt e* aDat a;
        ar r ay<const byt e* , GEMMs> bDat a;
        ar r ay<byt e* , GEMMs> cDat a;
        ar r ay<const byt e* , GEMMs> dDat a;
        byt e* r cDat a;
        ui nt 64_t * f l ags;
        ui nt M;
        ui nt syncI dx;
        ui nt t i l eI dx;
        ui nt bat chI dx;
        ui nt peer I dx;
        ui nt exper t I dx;
        ui nt i sPeer Remot e;
        TaskType t askType;
        ui nt 16_t t i l eSi ze;
        / / Pad t i l l 128- byt e cache l i ne
        ui nt paddi ng[ 6] = { } ;
   }
 1
 2
 3
 4
 5
 6
 7
 8
 9
10
11
12
13
14
15
16
17
18
19
20
```

Figure 15: *Task Struct*. TaskType ∈ {GEMM0, GEMM1, Combine}

