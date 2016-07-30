# nd4clj
## Rationale
If you have the use case that numpy fulfills and want to use java, nd4j is your
best bet.  If your language is clojure consider this library (nd4clj)
for your needs.  This libary is an implementation of core.matrix
using the nd4j.  It has similar performance caracteristics for
functions that map directly from nd4j to core.matrix.
Becuse the common matrix datastructure from nd4j and deeplearning4j
the NDArray is used in this implemetation, the NDArray manipulated
by this librarie's functions can be expected to work as deeplearning4j
training data.  

