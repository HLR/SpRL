package edu.tulane.cs.hetml.nlp.sprl.VisualTriplets

import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletsDataModel._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletClassifiers._
import scala.collection.JavaConversions._
/** Created by Umar on 2017-11-09.
  */
object VisualTripletsApp extends App {

  val visualTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets")
  val isTrain = false
  val classifierDirectory = s"models/mSpRL/VisualTriplets/"
  val classifierSuffix = "SVM-MSCOCO-original"
  if(isTrain) {


    val trainTriplets = visualTripletReader.trainImageTriplets

    visualTriplets.populate(trainTriplets)

    VisualTripletClassifier.modelSuffix = classifierSuffix
    VisualTripletClassifier.modelDir = classifierDirectory
    VisualTripletClassifier.learn(50)
    VisualTripletClassifier.save()
  }
  else {
    VisualTripletClassifier.modelSuffix = classifierSuffix
    VisualTripletClassifier.modelDir = classifierDirectory
    VisualTripletClassifier.load()

    val testTriplets = visualTripletReader.testImageTriplets

    visualTriplets.populate(testTriplets, false)

    VisualTripletClassifier.test()
  }
}

