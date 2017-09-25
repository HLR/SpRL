package edu.tulane.cs.hetml.nlp.sprl

import edu.illinois.cs.cogcomp.lbjava.learn.{SparseAveragedPerceptron, SparseNetworkLearner, SupportVectorMachine}
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import edu.illinois.cs.cogcomp.saul.datamodel.property.Property
import edu.tulane.cs.hetml.nlp.sprl.Helpers.FeatureSets
import edu.tulane.cs.hetml.nlp.sprl.Helpers.FeatureSets.FeatureSets
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.BaseTypes._

object MultiModalSpRLClassifiers {
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
    List(//JF2_1, JF2_2, JF2_3, JF2_4, JF2_5, JF2_6, JF2_8, JF2_9, JF2_10, JF2_11, JF2_13, JF2_14, JF2_15,
      tripletPhrasePos, tripletDependencyRelation, tripletHeadWordPos) ++
      (featureSet match {
        case FeatureSets.BaseLineWithImage => List(tripletImageConfirms)
        case FeatureSets.WordEmbedding => List(tripletTRSPPairVector, tripletSPLMPairVector)
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

  object TripletRelationClassifier extends Learnable(triplets) {
    def label = tripletIsRelation

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = tripletFeatures
  }

  object TripletRelationClassifierWithImage extends Learnable(triplets) {
    def label = tripletIsRelation

    override lazy val classifier = new SparseNetworkLearner {
      val p = new SparseAveragedPerceptron.Parameters()
      p.learningRate = .1
      p.positiveThickness = 4
      p.negativeThickness = 1
      baseLTU = new SparseAveragedPerceptron(p)
    }

    override def feature = List(JF2_1, JF2_2, JF2_3, JF2_4, JF2_5, JF2_6, JF2_8, JF2_9, JF2_10, JF2_11, JF2_13, JF2_14, JF2_15,
      tripletPhrasePos, tripletDependencyRelation, tripletHeadWordPos, tripletTRIsImageConceptExactMatch, tripletLMIsImageConceptExactMatch,
      tripletTRNearestSegmentConceptToHeadVector, tripletTRNearestSegmentConceptToPhraseVector, tripletTRIsImageConceptApproxMatch,
      tripletLMIsImageConceptApproxMatch, tripletLMNearestSegmentConceptToHeadVector,
      tripletLMNearestSegmentConceptToPhraseVector, tripletImageConfirms)
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
    def label = tripletRCC8

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = tripletFeatures
  }


  object TripletFoRClassifier extends Learnable(triplets) {
    def label = tripletFoR

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = tripletFeatures
  }
}
