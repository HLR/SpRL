package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import edu.illinois.cs.cogcomp.lbjava.learn.SparseNetworkLearner
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierDataModel._

object WordasClassifierClassifiers {

  class SingleWordasClassifer(word: String) extends Learnable(wordsegments) {
    def label = wordLabel is word

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = List(wordSegFeatures)
  }

  object ExpressionasClassifer extends Learnable(expressionSegmentPairs) {
    def label = expressionLabel

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = List(expressionScore, expressionSegFeatures)
  }
}
