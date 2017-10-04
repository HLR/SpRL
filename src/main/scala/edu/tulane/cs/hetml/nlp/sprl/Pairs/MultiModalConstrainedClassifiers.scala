package edu.tulane.cs.hetml.nlp.sprl.Pairs

import edu.illinois.cs.cogcomp.infer.ilp.OJalgoHook
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel.{pairToFirstArg, pairToSecondArg}
import edu.tulane.cs.hetml.nlp.sprl.Pairs.MultiModalSpRLConstraints._
import edu.tulane.cs.hetml.nlp.sprl.Pairs.MultiModalSpRLPairClassifiers._
/** Created by parisakordjamshidi on 2/4/17.
  */
object MultiModalConstrainedClassifiers {

  val erSolver = new OJalgoHook

  object TRPairConstraintClassifier extends ConstrainedClassifier[Relation, Relation](TrajectorPairClassifier) {
    def subjectTo = allConstraints
    override val solver = erSolver
    //override val pathToHead = Some()
  }
  object LMPairConstraintClassifier extends ConstrainedClassifier[Relation, Relation](LandmarkPairClassifier) {
    def subjectTo = allConstraints
    override val solver = erSolver
    //override val pathToHead = Some()
  }

  object LMConstraintClassifier extends ConstrainedClassifier[Phrase, Relation](LandmarkRoleClassifier) {
    def subjectTo = allConstraints
    override val solver = erSolver
    override val pathToHead = Some(-pairToFirstArg)
  }

  object TRConstraintClassifier extends ConstrainedClassifier[Phrase, Relation](TrajectorRoleClassifier) {
    def subjectTo = allConstraints
    override val solver = erSolver
    override val pathToHead = Some(-pairToFirstArg)
  }

  object IndicatorConstraintClassifier extends ConstrainedClassifier[Phrase, Relation](IndicatorRoleClassifier) {
    def subjectTo = allConstraints
    override val solver = erSolver
    override val pathToHead = Some(-pairToSecondArg)
  }
}
