package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.io.PrintStream

import edu.illinois.cs.cogcomp.lbjava.classify.{FeatureVector, ScoreSet}
import edu.illinois.cs.cogcomp.lbjava.learn.Learner
import edu.tulane.cs.hetml.nlp.BaseTypes.Relation
import MultiModalSpRLDataModel._

class ImageSupportsSpClassifier extends Learner("sprl.ImageSpClassifier") {

  override def allowableValues: Array[String] = {
    Array[String]("false", "true")
  }

  override def equals(o: Any): Boolean = {
    getClass == o.getClass
  }

  override def scores(example: AnyRef): ScoreSet = {
    val result: ScoreSet = new ScoreSet
    val r = example.asInstanceOf[Relation]
    val aligned = triplets(r) ~> tripletToVisualTriplet
    if (aligned.isEmpty) {
      result.put("none", 1.0)
      result.put("true", 0.0)
      result.put("false", 0.0)
    }
    else {
      val scores = getImageSpScores(aligned.head).take(5)
      val sp = r.getArgument(1).getText.replaceAll(" ", "_").toLowerCase()
      val found = scores.find(x=> x._1.trim.equalsIgnoreCase(sp))
      if (found.isEmpty) {
        result.put("none", 0.0)
        result.put("true", 0.0)
        result.put("false", 1.0)
      }
      else {
        result.put("none", 0.0)
        result.put("true", found.get._2 * 10)
        result.put("false", 0.0)
      }
    }
    result
  }

  override def write(printStream: PrintStream): Unit = ???

  override def scores(ints: Array[Int], doubles: Array[Double]): ScoreSet = ???

  override def classify(ints: Array[Int], doubles: Array[Double]): FeatureVector = ???

  override def learn(ints: Array[Int], doubles: Array[Double], ints1: Array[Int], doubles1: Array[Double]): Unit = ???
}