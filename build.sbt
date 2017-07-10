name := "SpRL"

version := "1.0"

scalaVersion := "2.12.2"

resolvers += "CogcompSoftware" at "http://cogcomp.cs.illinois.edu/m2repo/"
libraryDependencies ++= Seq(
  "edu.illinois.cs.cogcomp" % "saul_2.11" % "0.5.7",
  "edu.illinois.cs.cogcomp" % "saul-examples_2.11" % "0.5.7",
  "org.tallison" % "jmatio" % "1.2",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.7.2",
  "org.deeplearning4j" % "deeplearning4j-scaleout-api" % "1.0",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "0.7.2",
  "org.nd4j" % "nd4j-native-platform" % "0.7.2"
)

        