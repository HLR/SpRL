package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import edu.illinois.cs.cogcomp.lbjava.infer.{FirstOrderConstant, FirstOrderConstraint}
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.constraint.ConstraintTypeConversion._
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierDataModel._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers._

object WordExpressionSegmentConstraints {
  val roleIntegrity = ConstrainedClassifier.constraint[ExpressionSegment] {
    var a: FirstOrderConstraint = null
    s: ExpressionSegment =>
      a = new FirstOrderConstant(true)
      expressionsegments(s).foreach {
        x =>
          a = a and (
            (
              (ExpressionasClassifer on x) is "true") ==>
              (
                (TrajectorRoleClassifier on (triplets(x) ~> tripletToFirstArg).head is "Trajector")
                )
            )
      }
      a
  }
}
