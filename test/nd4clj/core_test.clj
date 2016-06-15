(ns nd4clj.core-test
  (:require [clojure.pprint :refer [pprint]]
            [clojure.core.matrix :refer [matrix mmul] :as mat]
            [clojure.test :refer :all]
            [clojure.core.matrix.compliance-tester :as compliance]
            [nd4clj.kiw :refer :all])
  (:import [org.nd4j.linalg.factory Nd4j]))


;(def canonical-object (Nd4j/create 4 2))
                                        ;(imp/register-implementation canonical-object)

(clojure.core.matrix/set-current-implementation :nd4j)

(deftest compliance-test
  (clojure.core.matrix.compliance-tester/instance-test (mat/matrix :nd4j [[2 0] [0 2]]))) 
