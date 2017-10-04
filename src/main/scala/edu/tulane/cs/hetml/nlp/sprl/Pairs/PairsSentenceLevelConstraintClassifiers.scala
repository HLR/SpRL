package edu.tulane.cs.hetml.nlp.sprl.Pairs

import edu.illinois.cs.cogcomp.infer.ilp.OJalgoHook
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation, Sentence}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Pairs.MultiModalSpRLPairClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.Pairs.PairsSentenceLevelConstraints._

/** Created by parisakordjamshidi on 2/9/17.
  */
object PairsSentenceLevelConstraintClassifiers {

  val erSolver = new OJalgoHook

  object TRPairConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TrajectorPairClassifier) {
    def subjectTo = allConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToPairs)
  }

  object LMPairConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](LandmarkPairClassifier) {
    def subjectTo = allConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToPairs)
  }

  object LMConstraintClassifier extends ConstrainedClassifier[Phrase, Sentence](LandmarkRoleClassifier) {
    def subjectTo = allConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToPhrase)
  }

  object TRConstraintClassifier extends ConstrainedClassifier[Phrase, Sentence](TrajectorRoleClassifier) {
    def subjectTo = allConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToPhrase)
  }

  object IndicatorConstraintClassifier extends ConstrainedClassifier[Phrase, Sentence](IndicatorRoleClassifier) {
    def subjectTo = allConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToPhrase)
  }
}
