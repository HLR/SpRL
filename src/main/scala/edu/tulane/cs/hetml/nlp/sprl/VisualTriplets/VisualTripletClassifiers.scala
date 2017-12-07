package edu.tulane.cs.hetml.nlp.sprl.VisualTriplets

import edu.illinois.cs.cogcomp.lbjava.learn.{SparseNetworkLearner, SupportVectorMachine}
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletsDataModel._

object VisualTripletClassifiers {

  object VisualTripletClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = List(visualTripletTrajector, visualTripletlandmark,
      visualTripletTrVector, visualTripletTrajectorAreaWRTLanmark, visualTripletTrajectorAspectRatio,
      visualTripletLandmarkAspectRatio, visualTripletTrajectorAreaWRTBbox, visualTripletLandmarkAreaWRTBbox, visualTripletIOU,
      visualTripletEuclideanDistance, visualTripletTrajectorAreaWRTImage, visualTripletLandmarkAreaWRTImage,
      visualTripletBelow, visualTripletAbove, visualTripletLeft, visualTripletRight, visualTripletUnion, visualTripletIntersection,
      visualTripletTrajectorW2V, visualTripletlandmarkW2V
    )
  }

  val binaryFeatures = List(visualTripletTrajector, visualTripletlandmark,
    visualTripletTrVector, visualTripletTrajectorAreaWRTLanmark, visualTripletTrajectorAspectRatio,
    visualTripletLandmarkAspectRatio, visualTripletTrajectorAreaWRTBbox, visualTripletLandmarkAreaWRTBbox, visualTripletIOU,
    visualTripletEuclideanDistance, visualTripletTrajectorAreaWRTImage, visualTripletLandmarkAreaWRTImage,
    visualTripletBelow, visualTripletAbove, visualTripletLeft, visualTripletRight, visualTripletUnion, visualTripletIntersection,
    visualTripletTrajectorW2V, visualTripletlandmarkW2V)

  object VisualTripletOnClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "on"

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = binaryFeatures
  }

  object VisualTripletInFrontOfClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "in_front_of"

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = binaryFeatures
  }

  object VisualTripletInClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "in"

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = binaryFeatures
  }

  object VisualTripletAboveClassifier extends Learnable(visualTriplets) {
    def label = visualTripletLabel is "above"

    override lazy val classifier = new SparseNetworkLearner()

    override def feature = binaryFeatures
  }

}
