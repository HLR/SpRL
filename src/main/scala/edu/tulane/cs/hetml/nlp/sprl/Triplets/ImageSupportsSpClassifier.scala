package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.io.PrintStream

import edu.illinois.cs.cogcomp.lbjava.classify.{FeatureVector, ScoreSet}
import edu.illinois.cs.cogcomp.lbjava.learn.Learner
import edu.tulane.cs.hetml.nlp.BaseTypes.Relation
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._

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
    val aligned = getAlignedVisualTriplet(r)
    if(aligned != null) {
      result.put("true", 1.0)
      result.put("false", 0.0)
    }
    else if(r.getArgument(1).getText.equalsIgnoreCase("in between")){
      result.put("true", 0.0)
      result.put("false", 1.0)
    }
    else{
      val scores = getImageSpScores(aligned)
      result.put("true", 1.0)
      result.put("false", 0.0)
    }
    result
  }

  override def write(printStream: PrintStream): Unit = ???

  override def scores(ints: Array[Int], doubles: Array[Double]): ScoreSet = ???

  override def classify(ints: Array[Int], doubles: Array[Double]): FeatureVector = ???

  override def learn(ints: Array[Int], doubles: Array[Double], ints1: Array[Int], doubles1: Array[Double]): Unit = ???
}