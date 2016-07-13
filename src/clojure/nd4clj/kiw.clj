(ns nd4clj.kiw
  (:require [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix :as m]
            [clojure.reflect :as r]
            [clojure.core.matrix.utils :as util]
            [clojure.core.matrix.compliance-tester :as ct]
            [clojure.core.matrix.implementations :as imp])
  (:use [clojure.pprint :only [print-table]])
  (:import [org.nd4j.linalg.factory Nd4j]
           [java.util Arrays]
           [org.nd4j.linalg.api.ndarray INDArray]
           [clojure.lang Numbers]
           [org.nd4j.linalg.api.ops.impl.transforms Pow]
           [org.nd4j.linalg.indexing IntervalIndex INDArrayIndex]))

;; TODO: eliminate reflection warnings
(set! *warn-on-reflection* true)
;; (set! *unchecked-math* true)

(defn sub-matrix [matrix a b c d]
(let [idx1 (IntervalIndex. true 1)
            idx2 (IntervalIndex. true 1)
            idxs (into-array INDArrayIndex [idx1 idx2])]
        (.init idx1 (int a) (int b))
        (.init idx2 (int c) (int d))
        (.get matrix idxs))
  )

(defn triangleUpper [matrix n offset1 offset2]
  ;(println n offset1 offset2)
  (let [nh (long (/ n 2))
        nr (long (mod n 2))
        a0 (+ nh offset1)
        b0 (+ (- n 1) offset1)
        c0 (+ 0 offset2)
        d0 (+ (- nh 1) offset2)
        sub (sub-matrix matrix a0 b0 c0 d0)]
    
    (and (if (> nh 1) 
        (and (triangleUpper matrix nh offset1 offset2)
             (triangleUpper matrix (+ nh nr) (+ nh offset1) (+ nh offset2)))
        true)
        (= 0.0 (.minNumber sub) (.maxNumber sub)))
    ))

(defn triangleLower [matrix n offset1 offset2]
  ;(println n offset1 offset2)
  (let [nh (long (/ n 2))
        nr (long (mod n 2))
        a0 (+ 0 offset1) ;(+ nh offset1)
        b0 (+ (- nh 1) offset1) ;(+ (- n 1) offset1)
        c0 (+ nh offset2) ;(+ 0 offset2)
        d0 (+ (- n 1) offset2) ;(+ (- nh 1) offset2)
        sub (sub-matrix matrix a0 b0 c0 d0)]
    (and (if (> nh 1) 
        (and (triangleLower matrix nh offset1 offset2)
             (triangleLower matrix (+ nh nr) (+ nh offset1) (+ nh offset2)))
        true)
        (= 0.0 (.minNumber sub) (.maxNumber sub)))
    ))

(defn square? [matrix] (apply = (vec (.shape matrix))))

(defn convert-to-nested-vectors [m]
  (let [sp (reverse (vec (.shape m)))
        flattened (vec (.asDouble (.data m)))]
    (first (reduce #(partition %2 %1) flattened sp))))

;;TODO: fix conversion to persistent array
(defn convert-mn [m data]
  (let [data-p (cond (instance? org.nd4j.linalg.api.ndarray.INDArray data)
                     (convert-to-nested-vectors data)
                     (instance? clojure.lang.PersistentVector data)
                     (if (instance? java.lang.Number (first data)) [data] (clojure.walk/prewalk #(if (instance? org.nd4j.linalg.api.ndarray.INDArray %) (first (convert-to-nested-vectors %)) %) data))
                     (instance? java.lang.Number data)
                     [[data]]
                     ;(or (instance? clojure.core.matrix.impl.wrappers.NDWrapper data) (instance? clojure.core.matrix.impl.ndarray_object.NDArray data))
                     
                     (or (instance? (Class/forName "[D") data) (instance? (Class/forName "[[D") data))
                     (let [pvec (m/to-nested-vectors data)] (if (instance? java.lang.Number (first pvec)) [pvec] pvec))
                     :else
                     (m/to-nested-vectors data))
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
    (let [dim (alength (.shape m))] (if (and (= dim 2) (or (.isRowVector m) (.isColumnVector m))) (if (> (.length m) 1) 1 0) dim)))
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
  mp/PBroadcastCoerce 
  (mp/broadcast-coerce [m a]
    (.broadcast (mp/construct-matrix m a) (.shape m)))
  mp/PImmutableAssignment
  (mp/assign [m source]
    (let [r (mp/broadcast-coerce m source)]
      (if (identical? r source) (mp/clone r) r)))
  mp/PFunctionalOperations
  (mp/element-seq [m]
    (vec (.asDouble (.data (.ravel m)))))
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
  mp/PSquare
  (mp/square [m] (.muli m m))
  mp/PMatrixPredicates
  (mp/identity-matrix? [m]
    (and (square? m) (mp/diagonal? m) (let [diag (mp/main-diagonal m)] (= 1.0 (.minNumber diag) (.maxNumber diag)))))
  (mp/zero-matrix? [m] (and (zero? (.minNumber m)) (zero? (.maxNumber m))))
  (mp/symmetric? [m] false)
  mp/PSliceJoinAlong
  (mp/join-along [m a dim]  (Nd4j/concat (int dim) (into-array org.nd4j.linalg.api.ndarray.INDArray [m a])))
  mp/PVectorOps
  (mp/vector-dot [a b] (.mmul a b))
  (mp/length [a] (.distance2 (Nd4j/zeros (.shape a)) a))
  (mp/length-squared [a] (.squaredDistance (Nd4j/zeros (.shape a)) a))
  (mp/normalise [a]  (Nd4j/norm2 a))

  mp/PMatrixCloning
  (mp/clone [m]
    (.dup m))

  mp/PConversion
  (mp/convert-to-nested-vectors [m]
    (let [sp (reverse (vec (.shape m)))
          flattened (vec (.asDouble (.data m)))]
      (first (reduce #(mapv vec (partition %2 %1)) flattened sp))))
  mp/PMatrixSlices
  (mp/get-row [m i]  (.getRow m i))
  (mp/get-column [m i]  (.getColumn m i))
  (mp/get-major-slice [m i] (.slice m i))
  (mp/get-slice [m dimension i] (.slice m i dimension))
  mp/PSliceSeq
  (mp/get-major-slice-seq [m] (mapv #(.slice m %) (range (mp/dimension-count m 0))))
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
          gt    (.gt a-add b-new)
          lt    (.lt a-sub b-new)
          gt-min (.minNumber gt)
          gt-max (.maxNumber gt)
          lt-min (.minNumber lt)
          lt-max (.maxNumber lt)]
      (= gt-min gt-max lt-min lt-max)))
  mp/PDoubleArrayOutput
  (mp/to-double-array [m] (.asDouble (.data (.dup m))))
  (mp/as-double-array [m] nil)
  mp/PValidateShape
  (mp/validate-shape
    [m]
    (vec (.shape m)))
  mp/PReshaping
  (mp/reshape [m shape] (.reshape m (int-array shape)))
  mp/PMatrixAdd
  (mp/matrix-add [m a] (.add m a))
  (mp/matrix-sub [m a] (.sub m a))
  mp/PZeroDimensionConstruction
  (mp/new-scalar-array
    ([m] (Nd4j/scalar 0))
    ([m value] (Nd4j/scalar value)))
  mp/PMatrixScaling
  (mp/scale [m a] (.mul m a))
  (mp/pre-scale [m a] (.mul m a))
  mp/PMatrixMutableScaling
  (mp/scale [m a] (.muli m a))
  (mp/pre-scale [m a] (.muli m a))
  mp/PMatrixMultiply
  (mp/matrix-multiply [m a] (.mmul m a))
  (mp/element-multiply [m a] (.mul m a))
    mp/PMatrixDivide
  (mp/element-divide
    ([m] (.rdiv m 1))
    ([m a] (.div m a)))
  mp/PMatrixDivideMutable
  (mp/element-divide!
    ([m] (.rdivi 1 m))
    ([m a] (.divi m a)))
  mp/PExponent
  (mp/element-pow [m exponent] (let [result (Nd4j/create (.shape m))] (.exec (Nd4j/getExecutioner) (Pow. (.dup m) result exponent)) result))
    mp/PMatrixTypes
  (mp/diagonal? [m] (and (triangleUpper m (aget (.shape m) 0) 0 0) (triangleLower m (aget (.shape m) 0) 0 0)))
  (mp/upper-triangular? [m] (triangleUpper m (aget (.shape m) 0) 0 0))
  (mp/lower-triangular? [m] (triangleLower m (aget (.shape m) 0) 0 0))
  (mp/positive-definite? [m] (throw (UnsupportedOperationException. "my exception message"))) ;TODO:
  (mp/positive-semidefinite? [m] (throw (UnsupportedOperationException. "my exception message")))
  (mp/orthogonal? [m eps] (mp/matrix-equals-epsilon (.mmul (.transpose m) m) (Nd4j/eye (aget (.shape m) 0)) eps))
  mp/PMatrixSubComponents
  (mp/main-diagonal [m] (Nd4j/diag m))
    (mp/main-diagonal [m] (Nd4j/diag m))
  mp/PValueEquality
  (mp/value-equals [m a] (mp/matrix-equals m a))
  )

(extend-type clojure.lang.PersistentVector
  mp/PMatrixEquality
  (mp/matrix-equals
    [a b]  (if (= (type a) (type b)) (= a b) (.equals (mp/construct-matrix b a) b))))

(def canonical-object (Nd4j/create 2 2))

(extend-type java.lang.Number
    mp/PBroadcast
  (mp/broadcast [m target-shape]
    (.broadcast (mp/construct-matrix canonical-object m) (int-array target-shape)))
  mp/PMatrixEquality
  (mp/matrix-equals
    [a b]
    (if (= (type a) (type b)) (= a b) (let [w (-> b flatten)] (and (= 1 (count w)) (= (first w) a))))))


(def N (mp/construct-matrix canonical-object [[0 0] [0 0]]))
(imp/register-implementation :nd4j N)
(clojure.core.matrix/set-current-implementation :nd4j)

(def N (mp/construct-matrix canonical-object [[0 1 2] [3 4 5] [6 7 8] [9 10 11]]))

(defn -main []
  ;(println (mapv #(-> % mp/to-double-array Arrays/toString) (mp/get-major-slice-seq N)))
  (binding [imp/*debug-options* {:print-registrations true}] (ct/compliance-test :nd4j))
  (println "Everything looks good!"))

