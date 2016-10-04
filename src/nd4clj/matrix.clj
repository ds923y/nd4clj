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
           [org.nd4j.linalg.api.ops.impl.broadcast BroadcastCopyOp]
           [org.nd4j.linalg.indexing IntervalIndex INDArrayIndex]))

;; TODO: eliminate reflection warnings
(set! *warn-on-reflection* true)
;; (set! *unchecked-math* true)

(declare convert-to-nested-vectors)
(declare convert-mn)
(declare wrap-matrix)
(declare empty-matrix)

(defn- broadcast [m target-shape]
  (let [dim-prj (->> target-shape count range (drop 1) int-array)
        crt (Nd4j/create (int-array target-shape))
        vbn (BroadcastCopyOp. crt (.a m) crt (int-array []))
        to-ret (.exec (Nd4j/getExecutioner) vbn (int-array dim-prj))] (wrap-matrix m to-ret)))

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

(defn- get-index-iterate-fn
  ([shape cur carry]
   (if-let [s (seq shape)]
     (if carry
       (let [rlt (mod (inc (first cur)) (first s))]
         (cons rlt (get-index-iterate-fn (rest s) (rest cur) (zero? rlt))))
       cur)
     []))
  ([shape]
   (fn [cur]
     (if-let [s (seq shape)]
       (let [rlt (mod (inc (first cur)) (first s))]
         (cons rlt (get-index-iterate-fn (rest s) (rest cur) (zero? rlt))))
       []))))

(defn- iterate-indexes [shape]
  (let [begin (repeat (count shape) 0)
        iterate-fn (get-index-iterate-fn shape)]
    (iterate iterate-fn begin)))

(defn- concat-along [is-vec old matrix dim]
  (if (= (alength (.shape old)) 2) ;nd4j concat does not work for indexes hi
    (Nd4j/concat #^int (int dim) #^"[Lorg.nd4j.linalg.api.ndarray.INDArray;" (into-array org.nd4j.linalg.api.ndarray.INDArray matrix))
    (let [iterater (iterate-indexes (.shape old))
          to-return (Nd4j/create (.shape old))]
      (doseq [idx (range (.length old))]
        (let [i1 (nth iterater idx)
              i2 (keep-indexed #(if (not= %1 dim) %2) i1)
              sliced-matrix (matrix (nth i1 dim))]
          (.putScalar ^INDArray to-return #^ints (int-array i1) (.getDouble ^INDArray sliced-matrix (int-array i2)))))
      to-return)))

(defn- rotate4 [old dim pos]
  (let [^INDArray matrix (.a old)
        dim-sz (-> matrix (.size dim))
        components (if (not (.vector old))
                     (map #(.slice ^INDArray matrix % (int dim)) (range dim-sz))
                     (map #(sub-matrix matrix % % 0 0) (range (.length matrix))))
        n-pos (mod pos (if (.vector old) (.length matrix) dim-sz))
        to-ret (Nd4j/create #^ints (.shape ^INDArray matrix))
        swapped (vec (concat (take-last (- (count components) n-pos) components) (take n-pos components)))]
    (concat-along (.vector old) matrix swapped dim)))

(defn- square? [^INDArray matrix] (apply = (vec (.shape matrix))))

(defn- m-new-scalar-array
  ([m] (mp/construct-matrix m (Nd4j/scalar #^double (double 0))))
  ([m value] (mp/construct-matrix m (Nd4j/scalar #^double (double value)))))

(deftype clj-INDArray [^INDArray a ^Boolean vector ^Boolean scalar ^Boolean empty]
  mp/PImplementation
  (mp/implementation-key [m] :nd4j)
  (mp/meta-info [m] {:doc "nd4j implementation of the core.matrix
  protocols. Allows different backends, e.g. jblas or jcublas for
  graphic cards."})
  (mp/construct-matrix [m data]
    (let [res (convert-mn m data)]
      (if (nil? data) (empty-matrix (.a ^clj-INDArray res)) res)))
  (mp/new-vector [m length]
    (let [res (Nd4j/create #^int (int length))
          e (zero? length)
          s (= length 1)
          v (> length 1)] (wrap-matrix m res v s e)))
  (mp/new-matrix [m rows columns]
    (let [res (Nd4j/create #^int (int rows) #^int (int columns))
          e (zero? (min rows columns))
          s (= rows columns 1)
          v (and (= (min rows columns) 1) (> (max rows columns) 1))] (wrap-matrix m res v s e)))
  (mp/new-matrix-nd [m shape]
    (let [res (Nd4j/create #^ints (int-array shape))
          e (zero? (apply min shape))
          s (apply = 1 shape)
          v (and (= (apply min shape) 1) (> (apply max shape) 1))] (wrap-matrix m res v s e)))
  (mp/supports-dimensionality? [m dimensions]
    (>= dimensions 0))
  mp/PDimensionInfo
  (mp/dimensionality [m] (let [w (alength (.shape ^INDArray a))]
                           (cond scalar 0 vector 1 :else w)))
  (mp/get-shape [m] (let [sp (vec (.shape ^INDArray a))]
                      (cond vector [(apply max sp)]
                            scalar 0
                            :else sp)))
  (mp/is-scalar? [m]
    scalar)
  (mp/is-vector? [m]
    vector)
  (mp/dimension-count [m dimension-number]
    (cond
      (and scalar (not= dimension-number 0)) (throw (IllegalArgumentException. "bad args"))
      (and vector (not= dimension-number 0) (throw (IllegalArgumentException. "bad args")))
      (or (neg? dimension-number) (>= dimension-number (mp/dimensionality m)) (throw (IllegalArgumentException. "bad args"))))
    (cond
      scalar 1
      vector (->> a (.shape) vec (apply max))
      :else (aget (.shape ^INDArray a) dimension-number)))
  mp/PIndexedAccess
  (mp/get-1d [m row]
    (let [ixs (int-array [row])]
      (.getDouble ^INDArray a #^int row)))
  (mp/get-2d [m row column]
    (let [ixs (int-array [row column])]
      (.getDouble ^INDArray a #^ints ixs)))
  (mp/get-nd [m indexes]
    (let [ixs (int-array indexes)]
      (.getDouble ^INDArray a #^ints ixs)))
  mp/PIndexedSetting
  (mp/set-1d [m row v]
    (let [d (.dup ^INDArray a)
          ixs (int-array row)]
      (.putScalar ^INDArray d #^ints ixs #^double (double v))))
  (mp/set-2d [m row column v]
    (let [d (.dup ^INDArray a)
          ixs (int-array [row column])]
      (.putScalar ^INDArray d #^ints ixs #^double (double v))))
  (mp/set-nd [m indexes v]
    (let [d (.dup ^INDArray a)
          indexes (int-array indexes)]
      (.putScalar ^INDArray d #^ints indexes #^double (double v))))
  (mp/is-mutable? [m] false)
  mp/PMatrixAdd
  (mp/matrix-add [m w] (wrap-matrix m (.add ^INDArray a ^INDArray (.a ^clj-INDArray (mp/construct-matrix m w)))))
  (mp/matrix-sub [m w] (wrap-matrix m (.sub ^INDArray a ^INDArray (.a ^clj-INDArray (mp/construct-matrix m w)))))
  mp/PZeroDimensionConstruction
  (mp/new-scalar-array [m] (mp/construct-matrix m 0))
  (mp/new-scalar-array [m value] (mp/construct-matrix m value))
  mp/PMatrixEquality
  (mp/matrix-equals [m m2] (if (= (type m) (type m2)) (.equals ^INDArray a ^INDArray (.a ^clj-INDArray m2)) (.equals ^INDArray a ^INDArray (.a ^clj-INDArray (mp/construct-matrix m m2)))))
  mp/PMatrixScaling
  (mp/scale [m b] (wrap-matrix m (.mul ^INDArray a ^Number (double b))))
  (mp/pre-scale [m b] (wrap-matrix m (.mul ^INDArray a ^Number (double b))))
  mp/PMatrixPredicates
  (mp/identity-matrix? [m]
    (let [h (.a m)] (and (square? a) (mp/diagonal? m) (let [diag (.a ^clj-INDArray (mp/main-diagonal m))] (= 1.0 (.minNumber ^INDArray diag) (.maxNumber ^INDArray diag))))))
  (mp/zero-matrix? [m] (let [h (.a m)] (and (zero? (.minNumber ^INDArray h)) (zero? (.maxNumber ^INDArray h)))))
  (mp/symmetric? [m] false)
  mp/PSpecialisedConstructors
  (mp/identity-matrix
    [m dims] (wrap-matrix m (Nd4j/eye dims) false false))
  (mp/diagonal-matrix
    [m diagonal-values]
    (let [to-ret (Nd4j/eye #^int (int (count diagonal-values)))]
      (doseq [i (range (count diagonal-values))]
        (.put ^INDArray to-ret #^int i #^int i ^Number (get diagonal-values i)))
      (wrap-matrix m to-ret false false)))
  mp/PMatrixTypes
  (mp/diagonal? [m] (and (triangleUpper a (aget (.shape ^INDArray a) 0) 0 0) (triangleLower ^INDArray a (aget (.shape ^INDArray a) 0) 0 0)))
  (mp/upper-triangular? [m] (triangleUpper ^INDArray a (aget (.shape ^INDArray a) 0) 0 0))
  (mp/lower-triangular? [m] (triangleLower ^INDArray a (aget (.shape ^INDArray a) 0) 0 0))
  (mp/positive-definite? [m] (throw (UnsupportedOperationException. "not implemented")))
  (mp/positive-semidefinite? [m] (throw (UnsupportedOperationException. "not implemented")))
  (mp/orthogonal? [m eps]
    (mp/matrix-equals-epsilon
     (wrap-matrix m (.mmul (.transpose a) a))
     (wrap-matrix m (Nd4j/eye (aget (.shape a) 0))) eps))
  mp/PMatrixSubComponents
  (mp/main-diagonal [m] (wrap-matrix m (Nd4j/diag ^INDArray a)))
  ;mp/PMatrixEqualityEpsilon
  ;(mp/matrix-equals-epsilon [s b eps]
  ;  (let [b-new (if (instance? org.nd4j.linalg.api.ndarray.INDArray (.a ^clj-INDArray b)) (.a ^clj-INDArray b) (convert-mn a (m/to-nested-vectors (.a ^clj-INDArray b))))
  ;        a-add (.add ^INDArray a ^Number eps)
  ;        a-sub (.sub ^INDArray a ^Number eps)
  ;        gt    (.gt ^INDArray a-add ^INDArray b-new)
  ;        lt    (.lt ^INDArray a-sub ^INDArray b-new)
  ;        gt-min (.minNumber ^INDArray gt)
  ;        gt-max (.maxNumber ^INDArray gt)
  ;        lt-min (.minNumber ^INDArray lt)
  ;        lt-max (.maxNumber ^INDArray lt)]
  ;    (= gt-min gt-max lt-min lt-max)))
  mp/PBroadcast
  (mp/broadcast [m target-shape] (broadcast m target-shape))
  mp/PBroadcastLike
  (mp/broadcast-like [m z]
    (let [to-broadcast (mp/construct-matrix m z)]
      (if (.scalar ^clj-INDArray to-broadcast)
        (wrap-matrix m (.assign ^INDArray (Nd4j/create (.shape a)) ^java.lang.Number z))
        (mp/broadcast to-broadcast (mp/get-shape m)))))
  mp/PBroadcastCoerce
  (mp/broadcast-coerce [m z]
    (mp/broadcast-like m z))
  mp/PValueEquality
  (mp/value-equals [m r] (mp/matrix-equals m r))
  mp/PRotate
  (mp/rotate [m dim places] (wrap-matrix m (rotate4 m dim places)))
  mp/PVectorView
  (mp/as-vector [m] (if (and (= (alength (.shape a)) 2) (or (.isColumnVector a) (.isRowVector a) vector)) (if (.isRowVector a) (wrap-matrix m a true false) (wrap-matrix m (.reshape a (int-array [1 (max (vec (.shape a)))])) true false)) (convert-mn a (vec (.asDouble (.data (.ravel a)))))))
  mp/PReshaping
  (mp/reshape [m shape] (let [v (if (= (count (vec shape)) 1) true false)
                              shape (if v (conj shape 1) shape)]
                          (wrap-matrix m (.reshape a (int-array shape)) v false)))
  mp/PElementCount
  (mp/element-count [m] (if empty 0 (.length a)))
  mp/PSameShape
  (mp/same-shape? [w r] (let [b (convert-mn w r)] (and (= (mp/get-shape w) (mp/get-shape b)) (= empty (.empty ^clj-INDArray b)) (= scalar (.scalar ^clj-INDArray b)) (= vector (.vector ^clj-INDArray b)))))
  mp/PImmutableAssignment
  (mp/assign [m source]
    (let [r (mp/broadcast-coerce m source)]
      (if (identical? r source) (mp/clone r) r)))
  mp/PMatrixCloning
  (mp/clone [m]
    (wrap-matrix m (.dup a)))
  mp/PExponent
  (mp/element-pow [m exponent] (let [result (Nd4j/create (.shape a))] (.exec (Nd4j/getExecutioner) (Pow. (.dup a) result exponent)) (wrap-matrix m result)))
  mp/PSquare
  (mp/square [m] (mp/element-pow m 2))
  mp/PMatrixDivide
  (mp/element-divide
    [m] (wrap-matrix m (.rdiv a 1)))
  (mp/element-divide
    [m w] (wrap-matrix m (.div a (.a (convert-mn m w)))))
  Object
  (toString [m] (str a))
;clojure.lang.Seqable
;(seq [m] )
;; The semantics should be to return a seq / iterator of row-major slices
;;
;; This enables matrices to be used with regular Clojure sequence operations e.g.
;;;;;;;;;;;;;;;;;;;;;;
;  PTypeInfo
)

(defn- wrap-matrix ([^clj-INDArray m ^INDArray mx] (->clj-INDArray mx (.vector m) (.scalar m) (.empty m)))
  ([^clj-INDArray _ ^INDArray mx ^Boolean vector ^Boolean scalar] (->clj-INDArray mx vector scalar false))
  ([^clj-INDArray _ ^INDArray mx ^Boolean vector ^Boolean scalar ^Boolean empty] (->clj-INDArray mx vector scalar empty)))

(defn- empty-matrix ([^INDArray mx] (->clj-INDArray mx false false true)))

(defn- convert-to-nested-vectors [^INDArray m]
  (let [sp (reverse (vec (.shape m)))
        flattened (vec (.asDouble (.data m)))]
    (first (reduce #(partition %2 %1) flattened sp))))

(defn- convert-mn [m data]
  (if  (instance? nd4clj.matrix.clj-INDArray data)
    data
    (let [data-p (cond ;(instance? org.nd4j.linalg.api.ndarray.INDArray data)
                     ;(convert-to-nested-vectors data)
                   (instance? clojure.lang.PersistentVector data)
                   (if (instance? java.lang.Number (first data)) [data] (clojure.walk/prewalk #(if (instance? org.nd4j.linalg.api.ndarray.INDArray %) (first (convert-to-nested-vectors %)) %) data))
                   (instance? java.lang.Number data)
                   [[data]]
                     ;(or (instance? (Class/forName "[D") data) (instance? (Class/forName "[[D") data))
                     ;(let [pvec (m/to-nested-vectors data)] (if (instance? java.lang.Number (first pvec)) [pvec] pvec))
                   :else
                   (let [pre (m/to-nested-vectors data)] (cond (mp/is-scalar? pre) [[pre]] (mp/is-vector? pre) [pre] :else pre)))
          crr (Nd4j/create
               (double-array (vec (flatten data-p)))
               (int-array
                (loop [cur data-p lst []]
                  (if (not (sequential? cur))
                    lst
                    (recur (first cur) (conj lst (count cur)))))))]
      (->clj-INDArray crr (mp/is-vector? data) (mp/is-scalar? data) false))))

(def canonical-object (->clj-INDArray (Nd4j/create 2 2) false false false))

(imp/register-implementation :nd4j canonical-object)
                                        ;(clojure.core.matrix/set-current-implementation :nd4j)

;(mp/get-slice-seq (m/matrix :ndarray [1 2]) 0)

                                        ;(mp/is-vector? (m/matrix :ndarray (m/matrix :nd4j [[1 2]])))


(vec (.shape (.a (m/matrix :nd4j [[1 2] [3 2]]))))
(let [array (.a (m/matrix :nd4j [[1 2] [3 2] [4 5]]))
      sp (vec (.shape array))
      repeat-num (reduce #(conj %1 (* (sp %2) (last %1))) [1] (range (dec (count sp))))]
  (map #(->> %1 range (map (fn [x] (repeat %2 x)))) sp repeat-num))
