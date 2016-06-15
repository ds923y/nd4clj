(defproject nd4clj "0.1.0-SNAPSHOT"
    :plugins [[lein-typed "0.3.5"]]
  :description "An implementation of core.matrix protocols with nd4j."
  :url "https://github.com/whilo/nd4clj"
  :source-paths      ["src/clojure"]
  :java-source-paths ["src/java"]
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [net.mikera/core.matrix "0.36.2-SNAPSHOT"]
                 [net.mikera/core.matrix.testing "0.0.5-SNAPSHOT"]
                 [org.nd4j/nd4j-x86 "0.4-rc3.8"]
                 ;[org.clojure/core.typed "0.3.23"]
                 #_[org.nd4j/jcublas "6.5"]
                 #_[org.nd4j/nd4j-jcublas-6.5 "0.0.3.5.5.6-SNAPSHOT"]]
;    :injections [(require 'clojure.core.typed)
;                 (clojure.core.typed/install)]
;  :core.typed {:check [nd4clj.core]}
  )
