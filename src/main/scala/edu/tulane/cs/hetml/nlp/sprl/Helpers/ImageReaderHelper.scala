package edu.tulane.cs.hetml.nlp.sprl.Helpers

import edu.tulane.cs.hetml.vision._

import scala.collection.JavaConversions._

/** Created by taher on 2017-02-28.
  */
class ImageReaderHelper(dataDir: String, trainFileName: String, testFileName: String, isTrain: Boolean) {

  lazy val reader = new CLEFImageReader(dataDir, trainFileName, testFileName, false)

  def getImageRelationList: List[SegmentRelation] = {

    if (isTrain) {
      reader.trainingRelations.toList
    } else {
      reader.testRelations.toList
    }
  }

  def getVisualTripletList: List[ImageTriplet] = {

    if (isTrain) {
      reader.trainImageTriplets.toList
    } else {
      reader.testImageTriplets.toList
    }
  }

  def getSegmentList: List[Segment] = {

    if (isTrain) {
      reader.trainingSegments.toList
    } else {
      reader.testSegments.toList
    }
  }

  def getImageList: List[Image] = {

    if (isTrain) {
      reader.trainingImages.toList
    } else {
      reader.testImages.toList
    }
  }
}
