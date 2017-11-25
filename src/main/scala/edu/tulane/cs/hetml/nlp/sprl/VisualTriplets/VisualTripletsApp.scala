package edu.tulane.cs.hetml.nlp.sprl.VisualTriplets

import java.io.FileOutputStream

import edu.tulane.cs.hetml.nlp.sprl.Helpers.ReportHelper
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletsDataModel._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletClassifiers._

import scala.collection.JavaConversions._

/** Created by Umar on 2017-11-09.
  */
object VisualTripletsApp extends App {

  val flickerTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "Flickr30k.majorityhead")
  val msCocoTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "MSCOCO.originalterm")
  val isTrain = true
  val classifierDirectory = s"models/mSpRL/VisualTriplets/"
  val classifierSuffix = "combined_perceptron"
  val trainTriplets = flickerTripletReader.trainImageTriplets ++ msCocoTripletReader.trainImageTriplets
  val testTriplets = flickerTripletReader.testImageTriplets ++ msCocoTripletReader.testImageTriplets
  val frequent = trainTriplets.groupBy(_.getSp.toLowerCase).filter(_._2.size > 0)
  trainTriplets.filter(x=> !frequent.contains(x.getSp.toLowerCase()))
    .foreach(_.setSp("none"))

  testTriplets.filter(x=> !frequent.contains(x.getSp.toLowerCase()))
    .foreach(_.setSp("none"))


  if (isTrain) {

    visualTriplets.populate(trainTriplets)

    VisualTripletClassifier.modelSuffix = classifierSuffix
    VisualTripletClassifier.modelDir = classifierDirectory
    VisualTripletClassifier.learn(50)
    VisualTripletClassifier.save()
    VisualTripletClassifier.test(visualTriplets())
  }
  else {
    VisualTripletClassifier.modelSuffix = classifierSuffix
    VisualTripletClassifier.modelDir = classifierDirectory
    VisualTripletClassifier.load()


    visualTriplets.populate(testTriplets, isTrain)

    val results = VisualTripletClassifier.test()
    val outStream = new FileOutputStream(s"data/mSprl/results/preposition-prediction-${classifierSuffix}.txt")
    ReportHelper.saveEvalResults(outStream, "preposition", results)
  }
}

