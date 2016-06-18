(defproject nd4clj "0.1.0-SNAPSHOT"
:plugins [[lein-cljfmt "0.5.3"]]
  :description "An implementation of core.matrix protocols with nd4j."
  :source-paths      ["src/clojure"]
  :java-source-paths ["src/java"]
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [net.mikera/core.matrix "0.40.0"]
                 [net.mikera/core.matrix.testing "0.0.4"]
                 [org.nd4j/nd4j-native "0.4-rc3.10"]
                 ]
  )