(defproject org.clojars.ds923y/nd4clj "0.1.0-SNAPSHOT"
  :plugins [[lein-cljfmt "0.5.3"]]
  :description "An implementation of core.matrix protocols with nd4j."
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [com.google.guava/guava "19.0"]
                 [net.mikera/core.matrix "0.51.1-SNAPSHOT"]
                 [org.nd4j/nd4j-native "0.5.0"]]
  
 :profiles {:dev {:dependencies [[net.mikera/core.matrix "0.51.1-SNAPSHOT" :classifier "tests"]] 
                   }} 
  )
