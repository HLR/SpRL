package edu.tulane.cs.hetml.nlp.sprl.Triplets

import edu.illinois.cs.cogcomp.lbjava.learn.{SparseAveragedPerceptron, SparseNetworkLearner}
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import edu.illinois.cs.cogcomp.saul.datamodel.property.Property
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.Helpers.FeatureSets
import edu.tulane.cs.hetml.nlp.sprl.Helpers.FeatureSets.FeatureSets
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._

object MultiModalSpRLTripletClassifiers {
  var featureSet = FeatureSets.WordEmbeddingPlusImage

  def phraseFeatures: List[Property[Phrase]] = phraseFeatures(featureSet)

  def phraseFeatures(featureSet: FeatureSets): List[Property[Phrase]] =
    List(wordForm, headWordFrom, pos, headWordPos, phrasePos, semanticRole, dependencyRelation, subCategorization,
      spatialContext, headSpatialContext, headDependencyRelation, headSubCategorization) ++
      (featureSet match {
        case FeatureSets.BaseLineWithImage => List()
        case FeatureSets.WordEmbedding => List(headVector)
        case FeatureSets.WordEmbeddingPlusImage => List(headVector)
        case _ => List[Property[Phrase]]()
      })

  def tripletFeatures: List[Property[Relation]] = tripletFeatures(featureSet)

  def tripletFeatures(featureSet: FeatureSets): List[Property[Relation]] =
    List(JF2_1, JF2_2, JF2_3, JF2_4, JF2_5, JF2_6, JF2_8, JF2_9, JF2_10, JF2_11, JF2_13, JF2_14, JF2_15,
      tripletPhrasePos, tripletDependencyRelation, tripletHeadWordPos) ++
      (featureSet match {
        case FeatureSets.BaseLineWithImage => List(tripletImageConfirms)
        case FeatureSets.WordEmbedding => List(tripletTrVector, tripletLmVector)
        case FeatureSets.WordEmbeddingPlusImage => List(tripletTrVector, tripletLmVector)
        case _ => List[Property[Relation]]()
      })

  object SpatialRoleClassifier extends Learnable(phrases) {
    def label = spatialRole

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = phraseFeatures
  }

  object TrajectorRoleClassifier extends Learnable(phrases) {
    def label = trajectorRole

    override lazy val classifier = new SparseNetworkLearner {
      val p = new SparseAveragedPerceptron.Parameters()
      p.learningRate = .1
      p.thickness = 2
      baseLTU = new SparseAveragedPerceptron(p)
    }

    override def feature = phraseFeatures
  }

  object LandmarkRoleClassifier extends Learnable(phrases) {
    def label = landmarkRole

    override lazy val classifier = new SparseNetworkLearner {
      val p = new SparseAveragedPerceptron.Parameters()
      p.learningRate = .1
      p.thickness = 2
      baseLTU = new SparseAveragedPerceptron(p)

    }

    override def feature = (phraseFeatures ++ List(lemma, headWordLemma))
      .diff(List(isImageConceptExactMatch))
  }

  object IndicatorRoleClassifier extends Learnable(phrases) {
    def label = indicatorRole

    override lazy val classifier = new SparseNetworkLearner {
      val p = new SparseAveragedPerceptron.Parameters()
      p.learningRate = .1
      p.thickness = 2
      baseLTU = new SparseAveragedPerceptron(p)
    }

    override def feature = (phraseFeatures(FeatureSets.BaseLine) ++ List(headSubCategorization))
      .diff(List(headWordPos, headWordFrom, headDependencyRelation, isImageConceptExactMatch))
  }

  object TripletRelationClassifier extends Learnable(triplets) {
    def label = tripletIsRelation

    override lazy val classifier = new SparseNetworkLearner()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector))
  }

  object TripletGeneralTypeClassifier extends Learnable(triplets) {
    def label = tripletGeneralType

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = (tripletFeatures)
      .diff(List(tripletLmVector))
  }

  object TripletSpecificTypeClassifier extends Learnable(triplets) {
    def label = tripletSpecificType

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = tripletFeatures
  }

  object TripletRegionClassifier extends Learnable(triplets) {
    def label = tripletRegion

    override lazy val classifier = new SparseNetworkLearner()

    override def feature =  (tripletFeatures)
      .diff(List(tripletLmVector))
  }


  object TripletDirectionClassifier extends Learnable(triplets) {
    def label = tripletDirection

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = tripletFeatures
  }
}
