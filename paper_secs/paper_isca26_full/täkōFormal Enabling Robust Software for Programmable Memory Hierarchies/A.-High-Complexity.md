# *A. High Complexity*

In tak¨ o, accesses performed during callbacks can interleave ¯ with accesses on core threads, increasing system complexity and counterintuitive outcomes. The confusion is exacerbated by the fact that callbacks can also access regular addresses. Programmers thus have to reckon not only with phantom address semantics, but also with how cache events for phantom addresses can trigger changes in the values of regular addresses. Next, we provide an example of this complexity.

