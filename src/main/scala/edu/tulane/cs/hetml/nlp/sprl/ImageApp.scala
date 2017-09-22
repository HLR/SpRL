package edu.tulane.cs.hetml.nlp.sprl

import edu.tulane.cs.hetml.vision.CLEFImageReader
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.transflowExamples.{HelloTF, LabelImage}

import scala.collection.JavaConversions._

/** Created by Taher on 2017-02-20.
  */
object ImageApp extends App {

  val readFullData = false
  /* Example 1 */
  val tensorflowExample1 = new HelloTF
  tensorflowExample1.TestGraph()

  /* Example 2 */
  val tensorflowLableImage = new LabelImage("data/imageModel","data/imageModel/28.jpg")

/*  val CLEFDataset = new CLEFImageReader("data/mSprl/saiapr_tc-12", "newSprl2017_train", "newSprl2017_gold", readFullData)

  val imageListTrain = CLEFDataset.trainingImages
  val segmentListTrain = CLEFDataset.trainingSegments
  val relationListTrain = CLEFDataset.trainingRelations

  images.populate(imageListTrain)
  segments.populate(segmentListTrain)
  segmentRelations.populate(relationListTrain)

  val imageListTest = CLEFDataset.testImages
  val segementListTest = CLEFDataset.testSegments
  val relationListTest = CLEFDataset.testRelations

  images.populate(imageListTest, false)
  segments.populate(segementListTest, false)
  segmentRelations.populate(relationListTest, false)
*/
}

