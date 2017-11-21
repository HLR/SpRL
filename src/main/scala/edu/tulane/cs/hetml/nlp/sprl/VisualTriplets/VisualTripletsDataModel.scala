package edu.tulane.cs.hetml.nlp.sprl.VisualTriplets

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors._

/** Created by Umar on 2017-11-09.
  */
object VisualTripletsDataModel extends DataModel {

  val visualTriplets = node[ImageTriplet]

  val visualTripletLabel = property(visualTriplets) {
    t: ImageTriplet =>
      t.getSp.toLowerCase
  }

  val visualTripletTrajector = property(visualTriplets) {
    t: ImageTriplet =>
      t.getTrajector.toLowerCase
  }

  val visualTripletlandmark = property(visualTriplets) {
    t: ImageTriplet =>
      t.getLandmark.toLowerCase
  }

  val visualTripletTrajectorW2V = property(visualTriplets, ordered = true) {
    t: ImageTriplet =>
      getGoogleWordVector(t.getTrajector)
  }

  val visualTripletlandmarkW2V = property(visualTriplets, ordered = true) {
    t: ImageTriplet =>
      getGoogleWordVector(t.getLandmark)
  }

  val visualTripletTrVector = property(visualTriplets, ordered = true) {
    t: ImageTriplet =>
      t.getTrVector.split("-").map(t=> t.toDouble).toList
  }

  val visualTripletTrajectorAreaWRTLanmark = property(visualTriplets) {
    t: ImageTriplet =>
      t.getTrAreawrtLM
  }

  val visualTripletTrajectorAspectRatio = property(visualTriplets) {
    t: ImageTriplet =>
      t.getTrAspectRatio
  }

  val visualTripletLandmarkAspectRatio = property(visualTriplets) {
    t: ImageTriplet =>
      t.getLmAspectRatio
  }

  val visualTripletTrajectorAreaWRTBbox = property(visualTriplets) {
    t: ImageTriplet =>
      t.getTrAreaBbox
  }

  val visualTripletLandmarkAreaWRTBbox = property(visualTriplets) {
    t: ImageTriplet =>
      t.getLmAreaBbox
  }

  val visualTripletIOU = property(visualTriplets) {
    t: ImageTriplet =>
      t.getIou
  }

  val visualTripletEuclideanDistance = property(visualTriplets) {
    t: ImageTriplet =>
      t.getEuclideanDistance
  }

  val visualTripletTrajectorAreaWRTImage = property(visualTriplets) {
    t: ImageTriplet =>
      t.getTrAreaImage
  }

  val visualTripletLandmarkAreaWRTImage = property(visualTriplets) {
    t: ImageTriplet =>
      t.getLmAreaImage
  }
}
