package edu.tulane.cs.hetml.nlp.sprl.Helpers

import edu.tulane.cs.hetml.vision._

import scala.collection.JavaConversions._

/** Created by taher on 2017-02-28.
  */
class ImageReaderHelper(dataDir: String, trainFileName: String, testFileName: String, isTrain: Boolean) {

  val ClefSegReader = new CLEFNewSegmentCNNFeaturesReader()
  val xmlFile = if(isTrain) trainFileName else testFileName
  ClefSegReader.loadFeatures(dataDir,xmlFile, isTrain)

  def getSegmentList: List[Segment] = {

    ClefSegReader.clefUniqueSegments.toList
  }

  def getImageList: List[Image] = {

    ClefSegReader.clefImages.toList
  }
}
