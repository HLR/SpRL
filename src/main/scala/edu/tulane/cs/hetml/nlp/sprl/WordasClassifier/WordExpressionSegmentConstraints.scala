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

      // The segment should be assigned to one expression
      expSegs.groupBy(_.getArgumentId(0).split("_")(0)).foreach(segExp => {
        a and segExp._2._atmost(1)(x => ExpressionasClassifer on x is "true")
      })

      // The expression should be assigned to one segment
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
//          val words = s.getText.split(" ").toList.filter(w => trainedWordClassifier.contains(w))
//
//          words.foreach{
//            w =>
//              val seg = expressionSegmentPairs(x) ~> AssmeSegment
//              var ws = new WordSegment(w, s)
//              val c = trainedWordClassifier(w)
//              a = a and (
//                ((ExpressionasClassifer on x) is "true") ==> (c on ws is "true")
//                )
//          }
//      }
//      a
//  }

  val expressionConstraints = ConstrainedClassifier.constraint[Sentence]{

    x: Sentence => uniqueExpressionSegmentPairs(x)
  }
}
