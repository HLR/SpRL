package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.io.PrintStream

import edu.illinois.cs.cogcomp.lbjava.classify.{FeatureVector, ScoreSet}
import edu.illinois.cs.cogcomp.lbjava.learn.Learner
import edu.tulane.cs.hetml.nlp.BaseTypes.Relation
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.vision.ImageTriplet

class ImageSupportsSpClassifier2 extends Learner("sprl.ImageSpClassifier") {

  override def allowableValues: Array[String] = {
    Array[String]("false", "true")
  }

  override def equals(o: Any): Boolean = {
    getClass == o.getClass
  }

  override def scores(example: AnyRef): ScoreSet = {
    val result: ScoreSet = new ScoreSet
    val r = example.asInstanceOf[ImageTriplet]
    val scores = getImageSpScores(r).take(20)
    val sp = r.getSp
    val found = scores.find(x => x._1.trim.equalsIgnoreCase(sp))
    if (found.isEmpty) {
      result.put("true", 0.0)
      result.put("false", 1.0)
    }
    else {
      result.put("true", found.get._2 * 10)
      result.put("false", 0.0)
    }
    result
  }

  override def write(printStream: PrintStream): Unit = ???

  override def scores(ints: Array[Int], doubles: Array[Double]): ScoreSet = ???

  override def classify(ints: Array[Int], doubles: Array[Double]): FeatureVector = ???

  override def learn(ints: Array[Int], doubles: Array[Double], ints1: Array[Int], doubles1: Array[Double]): Unit = ???
}