package edu.tulane.cs.hetml.nlp.sprl

import edu.illinois.cs.cogcomp.infer.ilp.{GurobiHook, OJalgoHook}
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.SentenceLevelConstraints._
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation, Sentence}

/** Created by parisakordjamshidi on 2/9/17.
  */
object SentenceLevelConstraintClassifiers {

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

  object TripletRelationTypeConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletRelationClassifier) {
    def subjectTo = tripletsConstraint

    override val solver = new GurobiHook
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletGeneralTypeConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletGeneralTypeClassifier) {
    def subjectTo = tripletsConstraint

    override val solver = new GurobiHook
    override val pathToHead = Some(-sentenceToTriplets)
  }
}
