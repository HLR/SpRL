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
      val expSegs = expressionSegmentPairs().filter(es => es.getArgumentId(1).split("_")(0)==s.getId.split("_")(0)).toList

      // The segment should be assigned to one expression
      val g1 = expSegs.groupBy(_.getArgumentId(0))

      g1.foreach(segExp => {
        a = a and segExp._2._atmost(1)(x => ExpressionasClassifer on x is "true")
      })

      // The expression should be assigned to one segment
      val g2 = expSegs.groupBy(_.getArgumentId(1))
      g2.foreach(expSeg => {
        a = a and expSeg._2._atmost(1)(x => ExpressionasClassifer on x is "true")
      })
      a
  }

  val integrityExpression = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToSegmentSentencePairs).foreach {
        x =>
          val words = s.getText.split(" ").toList.filter(w => trainedWordClassifier.contains(w))

          a = a and words._atleast(1){
            w =>
              val seg = expressionSegmentPairs(x) ~> expsegpairToSecondArg head
              val isRel = if (x.getArgumentId(2) == "isRel") true else false
              val ws = new WordSegment(w, seg, isRel, false, "")
              val c = trainedWordClassifier(w)
              c on ws is "true"
          } ==> ((ExpressionasClassifer on x) is "true")
      }
      a
  }

  val expressionConstraints = ConstrainedClassifier.constraint[Sentence]{

    x: Sentence => uniqueExpressionSegmentPairs(x) and integrityExpression(x)
  }
}
