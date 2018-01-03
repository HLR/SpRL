package edu.tulane.cs.hetml.nlp.sprl.Triplets

import edu.illinois.cs.cogcomp.lbjava.learn.{SparseAveragedPerceptron, SparseNetworkLearner, SparsePerceptron}
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLDataModel._
import  edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._

object MultiModalSpRLTripletBinaryClassifiers {

  object TripletGeneralDirectionClassifier extends Learnable(triplets) {
    def label = tripletGeneralType is "direction"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletLmVector, tripletMatchingSegmentRelationFeatures))
  }

  object TripletGeneralRegionClassifier extends Learnable(triplets) {
    def label = tripletGeneralType is "region"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletLmVector, tripletMatchingSegmentRelationFeatures))
  }

  object TripletDirectionBehindClassifier extends Learnable(triplets) {
    def label = tripletDirection is "behind"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletMatchingSegmentRelationFeatures))
  }

  object TripletDirectionBelowClassifier extends Learnable(triplets) {
    def label = tripletDirection is "below"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletMatchingSegmentRelationFeatures))
  }
  object TripletDirectionLeftClassifier extends Learnable(triplets) {
    def label = tripletDirection is "left"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletMatchingSegmentRelationFeatures))
  }
  object TripletDirectionAboveClassifier extends Learnable(triplets) {
    def label = tripletDirection is "above"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletMatchingSegmentRelationFeatures))
  }
  object TripletDirectionRightClassifier extends Learnable(triplets) {
    def label = tripletDirection is "right"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List())
  }
  object TripletDirectionFrontClassifier extends Learnable(triplets) {
    def label = tripletDirection is "front"

    override lazy val classifier = new SparsePerceptron()

    override def feature = (tripletFeatures)
      .diff(List(tripletMatchingSegmentRelationFeatures))
  }

  object TripletRegionTPPClassifier extends Learnable(triplets) {
    def label = tripletRegion is "TPP"

    override lazy val classifier = new SparsePerceptron()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector))
  }

  object TripletRegionEQClassifier extends Learnable(triplets) {
    def label = tripletRegion is "EQ"

    override lazy val classifier = new SparsePerceptron()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector))
  }

  object TripletRegionECClassifier extends Learnable(triplets) {
    def label = tripletRegion is "EC"

    override lazy val classifier = new SparsePerceptron()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector))
  }

  object TripletRegionDCClassifier extends Learnable(triplets) {
    def label = tripletRegion is "DC"

    override lazy val classifier = new SparsePerceptron()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector, tripletMatchingSegmentRelationFeatures))
  }

  object TripletRegionPOClassifier extends Learnable(triplets) {
    def label = tripletRegion is "PO"

    override lazy val classifier = new SparsePerceptron()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector, tripletMatchingSegmentRelationFeatures))
  }

  /* Below is the the list of binary classifiers per prepositions
  * */

  object PrepositionOnClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "on"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionInFrontOfClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "in_front_of"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionInClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "in"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionAboveClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "above"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionBetweenClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "between"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionOverClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "over"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionWithClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "with"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionSittingAroundClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "sitting_around"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionNextToClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "next_to"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionOnEachSideClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "on_each_side"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionNearClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "near"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionLeaningOnClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "leaning_on"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionInTheMiddleOfClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "in_the_middle_of"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionInBetweenClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "in_between"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionBehindClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "behind"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionAtClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "at"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }

  object PrepositionAroundClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "around"

    override lazy val classifier = new SparsePerceptron()

    override def feature = prepositionFeatures
  }
}
