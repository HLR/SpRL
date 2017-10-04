package edu.tulane.cs.hetml.nlp.sprl.Pairs

import edu.illinois.cs.cogcomp.lbjava.learn.{SparseAveragedPerceptron, SparseNetworkLearner}
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import edu.illinois.cs.cogcomp.saul.datamodel.property.Property
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.Helpers.FeatureSets
import edu.tulane.cs.hetml.nlp.sprl.Helpers.FeatureSets.FeatureSets
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._

object MultiModalSpRLPairClassifiers {
  var featureSet = FeatureSets.WordEmbeddingPlusImage

  def phraseFeatures: List[Property[Phrase]] = phraseFeatures(featureSet)

  def phraseFeatures(featureSet: FeatureSets): List[Property[Phrase]] =
    List(wordForm, headWordFrom, pos, headWordPos, phrasePos, semanticRole, dependencyRelation, subCategorization,
      spatialContext, headSpatialContext, headDependencyRelation, headSubCategorization) ++
      (featureSet match {
        case FeatureSets.BaseLineWithImage => List()
        case FeatureSets.WordEmbedding => List(headVector)
        case FeatureSets.WordEmbeddingPlusImage => List(headVector, nearestSegmentConceptToHeadVector)
        case _ => List[Property[Phrase]]()
      })

  def pairFeatures: List[Property[Relation]] = pairFeatures(featureSet)

  def pairFeatures(featureSet: FeatureSets): List[Property[Relation]] =
    List(pairWordForm, pairHeadWordForm, pairPos, pairHeadWordPos, pairPhrasePos,
      pairSemanticRole, pairDependencyRelation, pairSubCategorization, pairHeadSpatialContext,
      distance, before, isTrajectorCandidate, isLandmarkCandidate, isIndicatorCandidate) ++
      (featureSet match {
        case FeatureSets.BaseLineWithImage => List(pairIsImageConcept)
        case FeatureSets.WordEmbedding => List(pairTokensVector)
        case FeatureSets.WordEmbeddingPlusImage => List(pairTokensVector, pairNearestSegmentConceptToHeadVector,
          pairNearestSegmentConceptToPhraseVector, pairIsImageConcept)
        case _ => List[Property[Relation]]()
      })

  def tripletFeatures: List[Property[Relation]] = tripletFeatures(featureSet)

  def tripletFeatures(featureSet: FeatureSets): List[Property[Relation]] =
    List(tripletPhrasePos, tripletDependencyRelation, tripletHeadWordPos) ++
      (featureSet match {
        case FeatureSets.BaseLineWithImage => List(tripletImageConfirms)
        case FeatureSets.WordEmbedding => List(tripletTrVector, tripletSpVector, tripletLmVector)
        case FeatureSets.WordEmbeddingPlusImage => List()
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

  object TrajectorPairClassifier extends Learnable(pairs) {
    def label = isTrajectorRelation

    override lazy val classifier = new SparseNetworkLearner {
      val p = new SparseAveragedPerceptron.Parameters()
      p.learningRate = .1
      p.positiveThickness = 2
      p.negativeThickness = 1
      baseLTU = new SparseAveragedPerceptron(p)
    }

    override def feature = (pairFeatures ++ List(relationHeadDependencyRelation, relationHeadSubCategorization))
      .diff(List(pairNearestSegmentConceptToHeadVector))
  }

  object LandmarkPairClassifier extends Learnable(pairs) {
    def label = isLandmarkRelation

    override lazy val classifier = new SparseNetworkLearner {
      val p = new SparseAveragedPerceptron.Parameters()
      p.learningRate = .1
      p.positiveThickness = 4
      p.negativeThickness = 1
      baseLTU = new SparseAveragedPerceptron(p)
    }

    override def feature = (pairFeatures ++ List(relationSpatialContext))
      .diff(List(pairIsImageConcept, pairNearestSegmentConceptToPhraseVector))
  }

  object TripletGeneralTypeClassifier extends Learnable(triplets) {
    def label = tripletGeneralType

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = tripletFeatures
  }

  object TripletSpecificTypeClassifier extends Learnable(triplets) {
    def label = tripletSpecificType

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = tripletFeatures
  }

  object TripletRCC8Classifier extends Learnable(triplets) {
    def label = tripletRegion

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = tripletFeatures
  }


  object TripletDirectionClassifier extends Learnable(triplets) {
    def label = tripletDirection

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = tripletFeatures
  }
}
