package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import edu.illinois.cs.cogcomp.lbjava.infer.{FirstOrderConstant, FirstOrderConstraint}
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.constraint.ConstraintTypeConversion._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierDataModel._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers._

object WordExpressionSegmentConstraints {

  val uniqueExpressionSegmentPairs = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      val expSegs = (sentences(s) ~> sentenceToSegmentSentencePairs).toList

      // The segments assigned to a expression in a sentence should be at most 1
      expSegs.groupBy(_.getArgumentId(0).split("_")(0)).foreach(segExp => {
        a and segExp._2._atmost(1)(x => ExpressionasClassifer on x is "true")
      })

      expSegs.groupBy(_.getArgumentId(0)).foreach(expSeg => {
        a and expSeg._2._atmost(1)(x => ExpressionasClassifer on x is "true")
      })
      a
  }

//  val integrityExpression = ConstrainedClassifier.constraint[Sentence] {
//    var a: FirstOrderConstraint = null
//    s: Sentence =>
//      a = new FirstOrderConstant(true)
//      (sentences(s) ~> sentenceToSegmentSentencePairs).foreach {
//        x =>
//          a = a and (
//            ((ExpressionasClassifer on x) is "true") <==>
//              ((getWordasClassifierPredication(x)).equals("true"))
//            )
//      }
//      a
//  }

  val expressionConstraints = ConstrainedClassifier.constraint[Sentence]{

    x: Sentence => uniqueExpressionSegmentPairs(x)
  }

  def getWordasClassifierPredication(r: Relation): Boolean = {
    true
  }
}
