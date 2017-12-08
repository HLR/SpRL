package edu.tulane.cs.hetml.nlp.sprl.Triplets

import edu.illinois.cs.cogcomp.lbjava.learn.{SparseAveragedPerceptron, SparseNetworkLearner, SparsePerceptron, SupportVectorMachine}

import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import edu.illinois.cs.cogcomp.saul.datamodel.property.Property
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.Helpers.FeatureSets
import edu.tulane.cs.hetml.nlp.sprl.Helpers.FeatureSets.FeatureSets
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.vision.WordSegment

object MultiModalSpRLTripletClassifiers {
  var featureSet = FeatureSets.WordEmbeddingPlusImage

  def phraseFeatures: List[Property[Phrase]] = phraseFeatures(featureSet)

  def phraseFeatures(featureSet: FeatureSets): List[Property[Phrase]] =
    List(wordForm, headWordFrom, pos, headWordPos, phrasePos, semanticRole, dependencyRelation, subCategorization,
      spatialContext, headSpatialContext, headDependencyRelation, headSubCategorization) ++
      (featureSet match {
        case FeatureSets.BaseLineWithImage => List(similarityToMatchingSegment)
        case FeatureSets.WordEmbedding => List(headVector)
        case FeatureSets.WordEmbeddingPlusImage => List(headVector, similarityToMatchingSegment)
        case _ => List[Property[Phrase]]()
      })

  def tripletFeatures: List[Property[Relation]] = tripletFeatures(featureSet)

  def tripletFeatures(featureSet: FeatureSets): List[Property[Relation]] =
    List(JF2_1, JF2_2, JF2_3, JF2_4, JF2_5, JF2_6, JF2_8, JF2_9, JF2_10, JF2_11, JF2_13, JF2_14, JF2_15,
      tripletSpWithoutLandmark,
      tripletPhrasePos, tripletDependencyRelation, tripletHeadWordPos,
      tripletLmBeforeSp, tripletTrBeforeLm, tripletTrBeforeSp,
      tripletDistanceTrSp, tripletDistanceLmSp
    ) ++
      (featureSet match {
        case FeatureSets.BaseLineWithImage => List(tripletLmMatchingSegmentSimilarity,
          tripletTrMatchingSegmentSimilarity, tripletMatchingSegmentRelationFeatures)
        case FeatureSets.WordEmbedding => List(tripletTrVector, tripletLmVector)
        case FeatureSets.WordEmbeddingPlusImage => List(tripletTrVector, tripletLmVector,
          tripletLmMatchingSegmentSimilarity, tripletTrMatchingSegmentSimilarity)
        case _ => List[Property[Relation]]()
      })

  val prepositionFeatures = List(visualTripletTrajector, visualTripletlandmark,
    visualTripletTrVector, visualTripletTrajectorAreaWRTLanmark, visualTripletTrajectorAspectRatio,
    visualTripletLandmarkAspectRatio, visualTripletTrajectorAreaWRTBbox, visualTripletLandmarkAreaWRTBbox, visualTripletIOU,
    visualTripletEuclideanDistance, visualTripletTrajectorAreaWRTImage, visualTripletLandmarkAreaWRTImage,
    visualTripletBelow, visualTripletAbove, visualTripletLeft, visualTripletRight, visualTripletUnion, visualTripletIntersection,
    visualTripletTrajectorW2V, visualTripletlandmarkW2V)

  object SpatialRoleClassifier extends Learnable(phrases) {
    def label = spatialRole

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = phraseFeatures
  }

  val p = new SparsePerceptron.Parameters()
  p.learningRate = .1
  p.thickness = 2

  object TrajectorRoleClassifier extends Learnable(phrases) {
    def label = trajectorRole is "Trajector"

    override lazy val classifier = new SparsePerceptron(p)

    override def feature = phraseFeatures
  }

  object LandmarkRoleClassifier extends Learnable(phrases) {
    def label = landmarkRole is "Landmark"

    override lazy val classifier = new SparsePerceptron(p)

    override def feature = (phraseFeatures ++ List(lemma, headWordLemma))
      .diff(List())
  }

  object IndicatorRoleClassifier extends Learnable(phrases) {
    def label = indicatorRole is "Indicator"

    override lazy val classifier = new SparsePerceptron(p)

    override def feature = (phraseFeatures(FeatureSets.BaseLine) ++ List(headSubCategorization))
      .diff(List(headWordPos, headWordFrom, headDependencyRelation))
  }

  object TripletRelationClassifier extends Learnable(triplets) {
    def label = tripletIsRelation is "Relation"

    override lazy val classifier = new SparseNetworkLearner {
      val p = new SparseAveragedPerceptron.Parameters()
      p.learningRate = .1
      p.thickness = 2
      baseLTU = new SparseAveragedPerceptron(p)
    }

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector, tripletMatchingSegmentRelationFeatures))
  }

  object TripletGeneralTypeClassifier extends Learnable(triplets) {
    def label = tripletGeneralType

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = (tripletFeatures)
      .diff(List(tripletLmVector, tripletMatchingSegmentRelationFeatures))
  }

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

  object TripletSpecificTypeClassifier extends Learnable(triplets) {
    def label = tripletSpecificType

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = tripletFeatures
  }

  object TripletDirectionClassifier extends Learnable(triplets) {
    def label = tripletDirection

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = (tripletFeatures)
      .diff(List(tripletMatchingSegmentRelationFeatures))
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

  object TripletRegionClassifier extends Learnable(triplets) {
    def label = tripletRegion

    override lazy val classifier = new SparseNetworkLearner()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector, tripletMatchingSegmentRelationFeatures))
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

  object TripletImageRegionClassifier extends Learnable(triplets) {
    def label = tripletRegion

    override lazy val classifier = new SparseNetworkLearner()

    override def feature =  List(tripletMatchingSegmentRelationFeatures)
  }

  object PrepositionClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = prepositionFeatures
  }

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
