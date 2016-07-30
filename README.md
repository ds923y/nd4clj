# nd4clj
### Rationale
If you want to use clojure with the deeplearning4j library a clojure
wrapper for nd4j functions is needed.  This wrapper implements the
idiomatic and familiar clojure matrix interface `core.matrix`.
Existing clojure code using the `core.matrix` can change its
implementation to nd4clj and take advatage of the performance
of nd4j and interoperability with deeplearning4j libraries.
Becuse the common matrix datastructure between nd4j, nd4clj and deeplearning4j
is the NDArray, the NDArray manipulated
by this librarie's functions can be expected to work as deeplearning4j
training data.
### Tutorial
Add `[org.clojars.ds923y/nd4clj "0.1.0-SNAPSHOT"]` to your project.clj
file.  Here is an example of usage.
```
(ns useit.core
  (:require [nd4clj.matrix :as imp]
            [clojure.core.matrix :as indarray])
  (:gen-class))

(def A (indarray/array [[0 1] [2 3]]))
(def I (indarray/array [[1 0] [0 1]]))

(defn -main
  [& args]
  (println (indarray/mmul A I)))
```

### Limitations
This is a largely compliant implementation.  The bare minimum of
protocols are supported though.  Most people should not have problems.

