package edu.tulane.cs.hetml.nlp.sprl.Triplets

import edu.illinois.cs.cogcomp.infer.ilp.GurobiHook
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation, Sentence}
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletBinaryClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.TripletSentenceLevelConstraints._
import edu.tulane.cs.hetml.vision.ImageTriplet

object TripletSentenceLevelConstraintBinaryClassifiers {

  val erSolver = new GurobiHook()

  object TripletDirectionAboveConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletDirectionAboveClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletDirectionBelowConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletDirectionBelowClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletDirectionBehindConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletDirectionBehindClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletDirectionFrontConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletDirectionFrontClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletDirectionLeftConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletDirectionLeftClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletDirectionRightConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletDirectionRightClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletRegionTPPConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletRegionTPPClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletRegionECConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletRegionECClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletRegionDCConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletRegionDCClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletRegionEQConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletRegionEQClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object TripletRegionPOConstraintClassifier extends ConstrainedClassifier[Relation, Sentence](TripletRegionPOClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToTriplets)
  }

  object PrepositionAboveConstraintClassifier extends ConstrainedClassifier[ImageTriplet, Sentence](PrepositionAboveClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToVisualTriplet)
  }

  object PrepositionOnConstraintClassifier extends ConstrainedClassifier[ImageTriplet, Sentence](PrepositionOnClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToVisualTriplet)
  }

  object PrepositionInConstraintClassifier extends ConstrainedClassifier[ImageTriplet, Sentence](PrepositionInClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToVisualTriplet)
  }

  object PrepositionInFrontOfConstraintClassifier extends ConstrainedClassifier[ImageTriplet, Sentence](PrepositionInFrontOfClassifier) {
    def subjectTo = tripletConstraints

    override val solver = erSolver
    override val pathToHead = Some(-sentenceToVisualTriplet)
  }


}
