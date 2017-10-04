package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.io.PrintStream

import edu.illinois.cs.cogcomp.lbjava.classify.{FeatureVector, ScoreSet}
import edu.illinois.cs.cogcomp.lbjava.learn.Learner
import edu.tulane.cs.hetml.nlp.BaseTypes.Relation
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._

class SegmentPhraseSimilarityClassifier extends Learner("sprl.SegmentPhraseSimilarity") {

  override def allowableValues: Array[String] = {
    Array[String]("false", "true")
  }

  override def equals(o: Any): Boolean = {
    getClass == o.getClass
  }

  override def scores(example: AnyRef): ScoreSet = {
    val pair = example.asInstanceOf[Relation]
    val phrase = segmentPhrasePairs(pair) ~> segmentPhrasePairToPhrase head
    val seg = segmentPhrasePairs(pair) ~> -segmentToSegmentPhrasePair head
    val phraseHead = headWordFrom(phrase)
    val concept = seg.getSegmentConcept
    val sim = getSimilarity(phraseHead, concept)
    val result: ScoreSet = new ScoreSet
    result.put("false", 0)
    result.put("true", sim)
    result
  }

  override def write(printStream: PrintStream): Unit = ???

  override def scores(ints: Array[Int], doubles: Array[Double]): ScoreSet = ???

  override def classify(ints: Array[Int], doubles: Array[Double]): FeatureVector = ???

  override def learn(ints: Array[Int], doubles: Array[Double], ints1: Array[Int], doubles1: Array[Double]): Unit = ???
}