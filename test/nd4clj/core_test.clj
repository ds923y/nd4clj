(ns nd4clj.core-test
  (:require [clojure.pprint :refer [pprint]]
            [clojure.core.matrix :refer :all :as m]
            [clojure.test :refer :all]
            [clojure.core.matrix.compliance-tester :as compliance]
            [nd4clj.matrix])
  (:import [org.nd4j.linalg.factory Nd4j]))


;(def canonical-object (Nd4j/create 4 2))
                                        ;(imp/register-implementation canonical-object)

(clojure.core.matrix/set-current-implementation :nd4j)

#_(deftest slice-tests
  (is (equals [3 4] (slice (matrix :nd4j [[1 2] [3 4]]) 1))) ;; second slice
  (is (equals [1 3] (slice (matrix :nd4j [[1 2] [3 4]]) 1 0))) ;; first slice of second dimension
  ) 

#_(deftest shape-tests
  (is (= [2] (shape (first (slices (matrix :nd4j [[2 0] [0 2]])))))) ;; first of slice sequence
  (is (nil? (shape (first (eseq (matrix :nd4j [[2 0] [0 2]])))))) ;; first element
  ) 

(deftest compliance-test
  (clojure.core.matrix.compliance-tester/instance-test (matrix :nd4j [[2 0] [0 2]]))
  (clojure.core.matrix.compliance-tester/compliance-test (matrix :nd4j [[2 0] [0 2]]))) 
