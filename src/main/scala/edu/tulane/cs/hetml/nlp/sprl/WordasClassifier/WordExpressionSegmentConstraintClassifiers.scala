package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierDataModel._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers._
import edu.illinois.cs.cogcomp.infer.ilp.GurobiHook
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.tulane.cs.hetml.vision._

object WordExpressionSegmentConstraintClassifiers {

  val erSolver = new GurobiHook()

  object LMConstraintClassifier extends ConstrainedClassifier[ExpressionSegment, Segment](ExpressionasClassifer) {
    def subjectTo = roleConstraints

    override val solver = erSolver
    override val pathToHead = Some(-expressionSegmentToSegment)
  }
}
