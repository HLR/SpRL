package edu.tulane.cs.hetml.nlp.sprl.triplets

import edu.illinois.cs.cogcomp.infer.ilp.{GurobiHook, OJalgoHook}
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation, Sentence}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.triplets.TripletSentenceLevelConstraints._

object TripletSentenceLevelConstraintClassifiers {

  val erSolver = new GurobiHook()

  object LMConstraintClassifier extends ConstrainedClassifier[Phrase, Sentence](LandmarkRoleClassifier) {
    def subjectTo = roleConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToPhrase)
  }

  object TRConstraintClassifier extends ConstrainedClassifier[Phrase, Sentence](TrajectorRoleClassifier) {
    def subjectTo = roleConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToPhrase)
  }

  object IndicatorConstraintClassifier extends ConstrainedClassifier[Phrase, Sentence](IndicatorRoleClassifier) {
    def subjectTo = roleConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToPhrase)
  }

  object TripletRelationConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletRelationClassifier) {
    def subjectTo = boostTriplet

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletGeneralTypeConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletGeneralTypeClassifier) {
    def subjectTo = generalConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletDirectionConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletDirectionClassifier) {
    def subjectTo = directionConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletRegionConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletRegionClassifier) {
    def subjectTo = regionConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

}
