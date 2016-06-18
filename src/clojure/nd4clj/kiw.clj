(ns nd4clj.kiw
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m]
            [clojure.reflect :as r]
            [clojure.core.matrix.impl.wrappers :as wp]
            [clojure.core.matrix.impl.default :as dflt]
                                        ; [clojure.core.typed :as t]
            [clojure.core.matrix.utils :as util]
            [clojure.core.matrix.compliance-tester :as ct]
            [clojure.core.matrix.implementations :as imp])
  (:use [clojure.pprint :only [print-table]])
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.api.ndarray INDArray]
           [clojure.lang Numbers]
           [org.nd4j.linalg.indexing IntervalIndex INDArrayIndex]))

;; TODO: eliminate reflection warnings
(set! *warn-on-reflection* true)
;; (set! *unchecked-math* true)

(defn convert-to-nested-vectors [m]
  (let [sp (reverse (vec (.shape m)))
        flattened (vec (.asDouble (.data m)))]
    (first (reduce #(partition %2 %1) flattened sp))))

(defn convert-mn [m data] (let [data-p (cond (instance? org.nd4j.linalg.api.ndarray.INDArray data)
                                             (convert-to-nested-vectors data)
                                             (instance? clojure.lang.PersistentVector data)
                                             (if (instance? java.lang.Number (first data)) [data] data)
                                             (instance? java.lang.Number data)
                                             [[data]])
                                crr (Nd4j/create
                                     (double-array (vec (flatten data-p)))
                                     (int-array
                                      (loop [cur data-p lst []]
                                        (if (not (sequential? cur))
                                          lst
                                          (recur (first cur) (conj lst (count cur)))))))] crr))

(extend-type org.nd4j.linalg.api.ndarray.INDArray
  mp/PImplementation
  (mp/implementation-key [m] :nd4j)
  (mp/meta-info [m] {:doc "nd4j implementation of the core.matrix
  protocols. Allows different backends, e.g. jblas or jcublas for
  graphic cards."})
  (mp/construct-matrix [m data]
    (convert-mn m data))
  (mp/new-vector [m length]
    (Nd4j/create (int length)))
  (mp/new-matrix [m rows columns]
    (Nd4j/create (int rows) (int columns)))
  (mp/new-matrix-nd [m shape]
    (Nd4j/create (int-array shape)))
  (mp/supports-dimensionality? [m dimensions]
    (>= dimensions 2))
  mp/PDimensionInfo
  (mp/dimensionality [m]
    (alength (.shape m)))
  (mp/get-shape [m]
    (vec (int-array (.shape m))))
  (mp/is-scalar? [m]
    false)
  (mp/is-vector? [m]
    (.isVector m))
  (mp/dimension-count [m dimension-number]
    (aget (.shape m) dimension-number))
  mp/PIndexedAccess
  (mp/get-1d [m row]
    (let [ixs (int-array [row])]
      (.getDouble m row)))
  (mp/get-2d [m row column]
    (let [ixs (int-array [row column])]
      (.getDouble m ixs)))
  (mp/get-nd [m indexes]
    (let [ixs (int-array indexes)]
      (.getDouble m ixs)))
  mp/PIndexedSetting
  (mp/set-1d [m row v]
    (let [d (.dup m)
          ixs (int-array row)]
      (.putScalar d ixs (double v))))
  (mp/set-2d [m row column v]
    (let [d (.dup m)
          ixs (int-array [row column])]
      (.putScalar d ixs (double v))))
  (mp/set-nd [m indexes v]
    (let [d (.dup m)
          indexes (int-array indexes)]
      (.putScalar d indexes (double v))))
  (mp/is-mutable? [m] false)
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  mp/PBroadcast
  (mp/broadcast [m target-shape]
    (.broadcast m (int-array target-shape)))
  mp/PBroadcastLike
  (mp/broadcast-like [m a]
    (mp/broadcast (mp/construct-matrix m a) (mp/get-shape m)))
  mp/PBroadcastCoerce ;mp/convert-to-nested-vectors
  (mp/broadcast-coerce [m a]
    (.broadcast (mp/construct-matrix m a) (.shape m)))
  mp/PImmutableAssignment
  (mp/assign [m source]
    (let [r (mp/broadcast-coerce m source)]
      (if (identical? r source) (mp/clone r) r)))
  mp/PFunctionalOperations
  (mp/element-seq [m]
    (loop [pos 0 cmp []]
      (if (= pos (dec (.slices m)))
        cmp
        (recur (inc pos) (conj cmp (.slice m pos))))))
  (mp/element-map
    ([m f] (map f (mp/element-seq m)))
    ([m f a] (map f (mp/element-seq m) (mp/element-seq a)))
    ([m f a more] (apply (partial map f) m a (map mp/element-seq more))))
  (mp/element-map!
    ([m f] (mp/element-map m f))
    ([m f a] (mp/element-map m f a))
    ([m f a more] (mp/element-map m f a more)))
  (mp/element-reduce
    ([m f] (reduce f (mp/element-seq m)))
    ([m f init] (reduce f init (mp/element-seq m))))

  mp/PDoubleArrayOutput
  (mp/to-double-array [m] (.asDouble (.data m)))
  (mp/as-double-array [m] nil)
  mp/PSquare
  (mp/square [m] (.muli m m))
  mp/PMatrixPredicates
  (mp/identity-matrix? [m]
    (.equals (Nd4j/eye (aget (.shape m) 0)) m))
  (mp/zero-matrix? [m] (and (zero? (.minNumber m)) (zero? (.maxNumber m))))
  (mp/symmetric? [m] false)
  mp/PSliceJoinAlong
  (join-along [m a dim]  (Nd4j/concat (int dim) (into-array org.nd4j.linalg.api.ndarray.INDArray [m a])))
  mp/PVectorOps
  (mp/vector-dot [a b] (.mmul a b))
  (mp/length [a] (.distance2 (Nd4j/zeros (.shape a)) a))
  (mp/length-squared [a] (.squaredDistance (Nd4j/zeros (.shape a)) a))
  (mp/normalise [a]  (Nd4j/norm2 a))
  ;mp/PMatrixAdd
  ;(mp/matrix-add [m a])
  ;(mp/matrix-sub [m a])

  ;mp/PTypeInfo
  ;(mp/element-type [m] Double/TYPE)  
  ;mp/PMatrixMultiply
  ;(mp/matrix-multiply [m ^org.nd4j.linalg.api.ndarray.INDArray a]
  ;  (let [a (mp/construct-matrix m a)]
  ;    (cond (and (.isRowVector m) (.isRowVector ^org.nd4j.linalg.api.ndarray.INDArray a))
  ;          (.mmul m (.transpose ^org.nd4j.linalg.api.ndarray.INDArray a))
  ;          :else
  ;          (.mmul m a)
  ;          #_(and (.isRowVector m) (.isColumnVector a))
  ;          )))
  mp/PMatrixCloning
  (mp/clone [m]
    (.dup m))

  ;mp/PZeroDimensionConstruction
  ;(mp/new-scalar-array
  ;  ([m] (Nd4j/scalar 0))
  ;  ([m value] (Nd4j/scalar value)))
  ;mp/PZeroDimensionAccess
  ;(mp/get-0d [m] m #_(.getDouble m 0))
  ;(mp/set-0d! [m value] (.putScalar m value))
  ;mp/PZeroDimensionSet
  ;(mp/set-0d [m ^double value] (.putScalar (.dup m) value))
  ;mp/PSpecialisedConstructors
  ;(mp/identity-matrix [m dims] (println m) (println dims) (Nd4j/eye dims))
  ;(mp/diagonal-matrix [m diagonal-values] "Create a diagonal matrix with the specified leading diagonal values" (println "diagonal-values " (type diagonal-values) " " diagonal-values)
  ;                    (println (Nd4j/create (double-array diagonal-values)))
  ;                    (Nd4j/diag (Nd4j/create (double-array diagonal-values))))

  mp/PConversion
  (mp/convert-to-nested-vectors [m]
    (let [sp (reverse (vec (.shape m)))
          flattened (vec (.asDouble (.data m)))]
      (first (reduce #(partition %2 %1) flattened sp))))
  ;mp/PMatrixSlices
  ;(mp/get-row [m i] (println "get-row") (.getRow m i))
  ;(mp/get-column [m i] (println "get-column") (.getColumn m i))
  ;(mp/get-major-slice [m i] (println "get-major-slice ") (println i) (println (vec (.shape m))) (.slice m i))
  ;(mp/get-slice [m dimension i] (println "get-slice") (.slice m i dimension))

  ;mp/PSliceView
  ;(mp/get-major-slice-view [m i] (.slice m i (.majorStride m)))
  mp/PSliceSeq
  (mp/get-major-slice-seq [m] (map #(.slice m %) (range (mp/dimension-count m 0))))
  ;(reduce #(conj %1 (.slice m %2)) [] (range (aget (.shape m) 0)))
  ;mp/PSubVector
  ;(mp/subvector [m start length]
  ;  (if (mp/is-vector? m)
  ;    (.get m (into-array INDArrayIndex [(doto (IntervalIndex. false 1) (.init start (+ start length)))]))
  ;    (throw (ex-info "Not a vector" {:shape (.shape m)}))))
  ;mp/PMatrixSubComponents
                                        ;(mp/main-diagonal [m] (Nd4j/diag m))
 ;mp/PCoercion
 ;(mp/coerce-param [m param] (mp/construct-matrix m param))
  mp/PTranspose
  (mp/transpose [m] (.transpose m))
  mp/PComputeMatrix
  (mp/compute-matrix [m shape f]
    (let [m (mp/new-matrix-nd m shape)
          data (reduce (fn [m ix] (mp/set-nd m ix (apply f ix))) m (util/base-index-seq-for-shape shape))
          result (clojure.walk/prewalk #(if (instance? org.nd4j.linalg.api.ndarray.INDArray %) (convert-to-nested-vectors %) %) data)]
      result))

  mp/PMatrixEquality
  (mp/matrix-equals [a b] (if (= (type a) (type b)) (.equals a b) (.equals a (mp/construct-matrix a b))))
  mp/PMatrixEqualityEpsilon
  (mp/matrix-equals-epsilon [a b eps]
    (let [b-new (if (instance? org.nd4j.linalg.api.ndarray.INDArray b) b (convert-mn a (m/to-nested-vectors b)))
          a-add (.add a eps)
          a-sub (.sub a eps)
          prt2  (println (type a-add))
          gt    (.gt a-add b-new)
          lt    (.lt a-sub b-new)
          gt-min (.minNumber gt)
          gt-max (.maxNumber gt)
          lt-min (.minNumber lt)
          lt-max (.maxNumber lt)]
      (= gt-min gt-max lt-min lt-max)))
  mp/PDoubleArrayOutput
  (mp/to-double-array [m] (.asDouble (.data m)))
  (mp/as-double-array [m] nil)
  mp/PValidateShape
  (mp/validate-shape
    [m]
    (vec (.shape m)))
  ;mp/PMatrixTypes
  ;(mp/diagonal? [m] )
  ;(mp/upper-triangular? [m] "Returns true if the matrix m is upper triangualar")
  ;(mp/lower-triangular? [m] "Returns true if the matrix m is lower triangualar")
  ;(mp/positive-definite? [m] "Returns true if the matrix is positive definite")
  ;(mp/positive-semidefinite? [m] "Returns true if the matrix is positive semidefinite")
  ;(mp/orthogonal? [m eps] "Returns true if the matrix is orthogonal")
)

(extend-type clojure.lang.PersistentVector
  mp/PMatrixEquality
  (mp/matrix-equals
    [a b] (if (= (type a) (type b)) (= a b) (.equals (mp/construct-matrix b a) b))))

(extend-type java.lang.Number
  mp/PMatrixEquality
  (mp/matrix-equals
    [a b]
    (if (= (type a) (type b)) (= a b) (let [w (-> b flatten)] (and (= 1 (count w)) (= (first w) a))))))

#_(defprotocol PTranspose
    "Protocol for array transpose operation"
    (transpose [m]
      "Returns the transpose of a matrix. Equivalent to reversing the \"shape\".
     Note that:
     - The transpose of a scalar is the same scalar
     - The transpose of a 1D vector is the same 1D vector
     - The transpose of a 2D matrix swaps rows and columns"))
;(clojure.core.matrix.utils/extends-deep? mp/PImplementation org.nd4j.linalg.cpu.NDArray)


(def canonical-object (Nd4j/create 2 2))
(def N (mp/construct-matrix canonical-object [[0 0] [0 0]]))
(imp/register-implementation :nd4j N)
(clojure.core.matrix/set-current-implementation :nd4j)

(def N (mp/construct-matrix canonical-object [[0 1 2] [3 4 5] [6 7 8] [9 10 11]]))

(defn -main []
  (binding [imp/*debug-options* {:print-registrations true}] (ct/compliance-test :nd4j))
  (println "Everything looks good!"))

