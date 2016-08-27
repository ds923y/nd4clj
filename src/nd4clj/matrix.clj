(ns nd4clj.matrix
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
           [org.nd4j.linalg.api.shape Shape]
           [org.nd4j.linalg.cpu.nativecpu NDArray]
           [org.nd4j.linalg.api.ops.impl.transforms Pow]
           [org.nd4j.linalg.indexing IntervalIndex INDArrayIndex]))



;; TODO: eliminate reflection warnings
(set! *warn-on-reflection* true)
;; (set! *unchecked-math* true) 

(defn- sub-matrix [^INDArray matrix a b c d]
  (let [idx1 (IntervalIndex. true 1)
        idx2 (IntervalIndex. true 1)
        idxs (into-array INDArrayIndex [idx1 idx2])]
    (.init idx1 (int a) (int b))
    (.init idx2 (int c) (int d))
    (.get matrix idxs)))

(defn- triangleUpper [^INDArray matrix n offset1 offset2]
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
         (= 0.0 (.minNumber ^INDArray sub) (.maxNumber ^INDArray sub)))))

(defn- triangleLower [^INDArray matrix n offset1 offset2]
  (let [nh (long (/ n 2))
        nr (long (mod n 2))
        a0 (+ 0 offset1) 
        b0 (+ (- nh 1) offset1)
        c0 (+ nh offset2) 
        d0 (+ (- n 1) offset2)
        sub (sub-matrix matrix a0 b0 c0 d0)]
    (and (if (> nh 1)
           (and (triangleLower matrix nh offset1 offset2)
                (triangleLower matrix (+ nh nr) (+ nh offset1) (+ nh offset2)))
           true)
         (= 0.0 (.minNumber ^INDArray sub) (.maxNumber ^INDArray sub)))))

(defn- rotate2 [^INDArray matrix dim pos]
  (let [dim-sz (-> matrix (.size dim))
        components (reduce #(conj %1 (.slice matrix %2 (int dim))) [] (range dim-sz))
        n-pos (mod pos dim-sz)]
    (if (< (count components) 2)
      (first components)
      (let [to-ret (Nd4j/create (.shape matrix))
            pret (vec (concat (take-last (- dim-sz n-pos) components) (take n-pos components)))    
            ret (Nd4j/concat (int dim) (into-array INDArray pret))] (.reshape ret (.shape matrix))))))

(defn- amt ([a b c] (cons a (lazy-seq (amt (nth b c) b (inc c)))))
  ([b] (amt (nth b 0) b (inc 0))))

(defn- matrix-indexes [^INDArray matrix] (let [shp (vec (.shape matrix))
                                     total (reduce * shp)
                                     steps (vec (reductions * 1 shp))]
                                 (identity (map #(amt (cycle (flatten (map (fn [a] (repeat (steps %) a)) (range (shp %)))))) (range (count shp))))))

(defn- insert-helper [dim m-idxs inner] (concat (take dim m-idxs) [(repeat inner)] (drop dim m-idxs)))

(defn- rotate3 [^INDArray matrix dim pos]
  (let [dim-sz (-> matrix (.size dim))
        components (map #(.slice matrix % (int dim)) (range dim-sz))
        n-pos (mod pos dim-sz)]
    (if (< (count components) 2)
      (first components)
      (let [to-ret (Nd4j/create (.shape matrix))
            pret (vec (concat (identity (take-last (- dim-sz n-pos) components)) (take n-pos components)))
            m-idxs (matrix-indexes (first pret))
            to-concat (map #(apply map vector (insert-helper dim m-idxs %)) (range (count pret)))
            indexes-c (apply map vector m-idxs)] (doseq [i1 (range (count pret))
                                                         i2 (range (reduce * (vec (.shape (first pret)))))]
                                                   (.putScalar to-ret (int-array (nth (nth to-concat i1) i2)) (.getDouble (nth pret i1) (int-array (nth indexes-c i2))))) to-ret))))

(defn- square? [^INDArray matrix] (apply = (vec (.shape matrix))))

(declare convert-to-nested-vectors)
(declare convert-mn)
(declare wrap-matrix)

(defn- m-new-scalar-array
                       ([m] (mp/construct-matrix m (Nd4j/scalar 0)))
                       ([m value] (mp/construct-matrix m (Nd4j/scalar value))))


(deftype clj-INDArray [^INDArray a ^Boolean vector ^Boolean scalar]
 ; MyInterface ; implement the specified protocol (i.e. interface)
    
    ; each function's scope is defined by 
    ; the object provided as the first argument
    ; i.e. something that is of the `MyClass` type
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
  (mp/dimensionality [m] (alength (.shape a)))
  (mp/get-shape [m] #_(println "get-shape" (type a) a (type m) m) #_(flush) (let [sp (vec (.shape a))]
                      (cond vector [(apply max sp)]
                            scalar nil
                            :else sp)))
  (mp/is-scalar? [m]
    scalar)
  (mp/is-vector? [m]
    vector)
  (mp/dimension-count [m dimension-number]
    (aget (.shape a) dimension-number))
  mp/PIndexedAccess
  (mp/get-1d [m row]
    (let [ixs (int-array [row])]
      (.getDouble a row)))
  (mp/get-2d [m row column]
    (let [ixs (int-array [row column])]
      (.getDouble a ixs)))
  (mp/get-nd [m indexes]
    (let [ixs (int-array indexes)]
      (.getDouble a ixs)))
  mp/PIndexedSetting
  (mp/set-1d [m row v]
    (let [d (.dup a)
          ixs (int-array row)]
      (.putScalar d ixs (double v))))
  (mp/set-2d [m row column v]
    (let [d (.dup a)
          ixs (int-array [row column])]
      (.putScalar d ixs (double v))))
  (mp/set-nd [m indexes v]
    (let [d (.dup a)
          indexes (int-array indexes)]
      (.putScalar d indexes (double v))))
  (mp/is-mutable? [m] false)
    mp/PMatrixAdd
  (mp/matrix-add [m w] (wrap-matrix m (.add a (.a (mp/construct-matrix m w)))))
  (mp/matrix-sub [m w] (wrap-matrix m (.sub a (.a (mp/construct-matrix m w)))))
  mp/PZeroDimensionConstruction
  (mp/new-scalar-array [m] (mp/construct-matrix m 0))
  (mp/new-scalar-array [m value] (mp/construct-matrix m value))
  mp/PMatrixEquality
  (mp/matrix-equals [m m2] (if (= (type m) (type m2)) (.equals a (.a m2)) (.equals a (.a (mp/construct-matrix m m2)))))
  mp/PMatrixScaling
  (mp/scale [m b] (wrap-matrix m (.mul a (double b))))
  (mp/pre-scale [m b] (wrap-matrix m (.mul a (double b))))
    mp/PBroadcast
  (mp/broadcast [m target-shape]
    (wrap-matrix m (.broadcast (.a m) (int-array target-shape))))
  mp/PBroadcastLike
  (mp/broadcast-like [m b]
    (mp/broadcast (mp/construct-matrix m b) (mp/get-shape m)))
  mp/PBroadcastCoerce
  (mp/broadcast-coerce [m b]
    (wrap-matrix m (.broadcast (.a (mp/construct-matrix m b)) (.shape (.a m)))))
  )
;vector scalar
(defn- wrap-matrix [m mx] (->clj-INDArray mx (.vector m) (.scalar m)))

(defn- convert-to-nested-vectors [^INDArray m]
  (let [sp (reverse (vec (.shape m)))
        flattened (vec (.asDouble (.data m)))]
    (first (reduce #(partition %2 %1) flattened sp))))

;;TODO: fix conversion to persistent array
(defn- convert-mn [m data]
  (if  (instance? nd4clj.matrix.clj-INDArray data)
                     data
  (let [data-p (cond (instance? org.nd4j.linalg.api.ndarray.INDArray data)
                     (convert-to-nested-vectors data)
                     (instance? clojure.lang.PersistentVector data)
                     (if (instance? java.lang.Number (first data))  #_(throw (IllegalArgumentException. "1D vector fuctionality not implemented: supports >= 2D")) [data] (clojure.walk/prewalk #(if (instance? org.nd4j.linalg.api.ndarray.INDArray %) (first (convert-to-nested-vectors %)) %) data))
                     (instance? java.lang.Number data)
                     #_(throw (IllegalArgumentException. "scalar 0D fuctionality not implemented: supports >= 2D")) [[data]]
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
                  (recur (first cur) (conj lst (count cur)))))))]
    (cond (instance? java.lang.Number data) (->clj-INDArray crr false true)
          (and (instance? clojure.lang.PersistentVector data) (instance? java.lang.Number (first data))) (->clj-INDArray crr true false)
          :else (->clj-INDArray crr false false)))))

#_(extend-type org.nd4j.linalg.api.ndarray.INDArray
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
  (mp/dimensionality [m] (alength (.shape m)))
  (mp/get-shape [m] (vec (.shape m)))
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
  (mp/square [m] (mp/element-pow m 2))
  mp/PMatrixPredicates
  (mp/identity-matrix? [m]
    (and (square? m) (mp/diagonal? m) (let [diag (mp/main-diagonal m)] (= 1.0 (.minNumber ^INDArray diag) (.maxNumber ^INDArray diag)))))
  (mp/zero-matrix? [m] (and (zero? (.minNumber ^INDArray m)) (zero? (.maxNumber ^INDArray m))))
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
  (mp/get-major-slice [m i] (if (or (.isColumnVector m) (.isRowVector m)) (->clj-INDArray m) (.slice m i)))
  (mp/get-slice [m dimension i] (.slice m i dimension))
  mp/PSliceSeq
  (mp/get-major-slice-seq [m] (mapv #(mp/get-major-slice m %) (range (mp/dimension-count m 0))))
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
          gt-min (.minNumber ^INDArray gt)
          gt-max (.maxNumber ^INDArray gt)
          lt-min (.minNumber ^INDArray lt)
          lt-max (.maxNumber ^INDArray lt)]
      (= gt-min gt-max lt-min lt-max)))
  mp/PDoubleArrayOutput
  (mp/to-double-array [m] (println "will exit") (System/exit 0) (.asDouble (.data (.dup m))))
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
  (mp/positive-definite? [m] (throw (UnsupportedOperationException. "not implemented"))) 
  (mp/positive-semidefinite? [m] (throw (UnsupportedOperationException. "not implemented")))
  (mp/orthogonal? [m eps] (mp/matrix-equals-epsilon (.mmul (.transpose m) m) (Nd4j/eye (aget (.shape m) 0)) eps))
  mp/PMatrixSubComponents
  (mp/main-diagonal [m] (Nd4j/diag m))
  (mp/main-diagonal [m] (Nd4j/diag m))
  mp/PValueEquality
  (mp/value-equals [m a] (mp/matrix-equals m a))
  mp/PRotate
  (mp/rotate [m dim places] (if (= (alength (.shape m)) 2) (rotate2 m dim places) (rotate3 m dim places))))

(extend-type clojure.lang.PersistentVector
  mp/PMatrixEquality
  (mp/matrix-equals
    [a b]  (if (= (type a) (type b)) (= a b) (.equals (mp/construct-matrix b a) b))))

(def canonical-object (->clj-INDArray (Nd4j/create 2 2) false false))

(extend-type java.lang.Number
  mp/PBroadcast
  (mp/broadcast [m target-shape]
    (.broadcast (mp/construct-matrix canonical-object m) (int-array target-shape)))
  mp/PMatrixEquality
  (mp/matrix-equals
    [a b]
    (if (= (type a) (type b)) (= a b) (let [w (-> b flatten)] (and (= 1 (count w)) (= (first w) a))))))

(def N (mp/construct-matrix canonical-object [[0 0] [0 0]]))
(imp/register-implementation :nd4j canonical-object)
(clojure.core.matrix/set-current-implementation :nd4j)


