package edu.tulane.cs.hetml.nlp.sprl.VisualTriplets

import java.io.FileOutputStream

import scala.io.Source
import edu.tulane.cs.hetml.nlp.sprl.Helpers.ReportHelper
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletsDataModel._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierConfigurator.resultsDir
import scala.util.Random
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

/** Created by Umar on 2017-11-09.
  */
object VisualTripletsApp extends App {

//  val flickerTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "Flickr30k.majorityhead")
//  val msCocoTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "MSCOCO.originalterm")
//  val externalTrainTriplets = flickerTripletReader.trainImageTriplets //msCocoTripletReader.trainImageTriplets   ++
//  val testTriplets = flickerTripletReader.testImageTriplets //flickerTripletReader.testImageTriplets ++

  val frequentPrepositions = List("in", "on", "between", "in front of", "behind", "above", "in between", "around", "over")

  val isTrain = true
  val classifierDirectory = s"models/mSpRL/VisualTriplets/"
  val classifierSuffix = "vgg_perceptron"


  if (isTrain) {

    val VGGJsonReader = new JSONReader()
    VGGJsonReader.readJsonFile("data/mSprl/saiapr_tc-12/VggData/")

    val vggTriplets = VGGJsonReader.trainImageTriplets.take(2000)

    val vggFrequentTriplets = vggTriplets.filter(x => frequentPrepositions.contains(x.getSp))

    vggFrequentTriplets.groupBy(_.getSp).foreach(x => {
      println(x._1, x._2.size)
    })

//    val clefTrainInstances = new ListBuffer[ImageTriplet]()
//    val filename = s"$resultsDir/clefprepdata.txt"
//    for (line <- Source.fromFile(filename).getLines) {
//      val parts = line.split(",")
//      val trBox = RectangleHelper.parseRectangle(parts(3), "-")
//      val lmBox = RectangleHelper.parseRectangle(parts(4), "-")
//      val imageWidth = parts(5).toDouble
//      val imageHeight = parts(6).toDouble
//      clefTrainInstances += new ImageTriplet(parts(0), parts(1), parts(2), trBox, lmBox
//        , imageWidth, imageHeight, parts(7).toDouble, parts(8).toDouble,
//        parts(9).toDouble, parts(10).toDouble, parts(11).toDouble, parts(12).toDouble, parts(13).toDouble, parts(14).toDouble,
//        parts(15).toDouble, parts(16).toDouble, parts(17).toDouble, parts(18).toDouble, parts(19).toDouble, parts(20).toDouble,
//        parts(21).toDouble, RectangleHelper.getIntersectionArea(trBox, lmBox, imageWidth * imageHeight), RectangleHelper.getUnionArea(trBox, lmBox, imageWidth * imageHeight)
//      )
//    }

    visualTriplets.populate(vggFrequentTriplets)
//    VisualTripletClassifier.learn(50)
//    VisualTripletClassifier.save()
    VisualTripletClassifier.load()
    VisualTripletClassifier.test(visualTriplets())

    //fine tune
//    visualTriplets.clear()
//    visualTriplets.populate(clefTrainInstances)
//    VisualTripletOnClassifier.learn(10)
//    VisualTripletInClassifier.learn(10)
//    VisualTripletInFrontOfClassifier.learn(50)
//    VisualTripletAboveClassifier.learn(50)
//
//    visualClassifier.save()
  }
  else {
      val testInstances = new ListBuffer[ImageTriplet]()
      val filename = s"$resultsDir/clefprepdata-test.txt"
      for (line <- Source.fromFile(filename).getLines) {
        val parts = line.split(",")
        if(parts(0)!="-") {
          val trBox = RectangleHelper.parseRectangle(parts(3), "-")
          val lmBox = RectangleHelper.parseRectangle(parts(4), "-")
          val imageWidth = parts(5).toDouble
          val imageHeight = parts(6).toDouble

          testInstances += new ImageTriplet(parts(0), parts(1), parts(2), trBox, lmBox, imageWidth, imageHeight)
        }
      }

    VisualTripletClassifier.load()
    visualTriplets.populate(testInstances, isTrain) //testTriplets, isTrain)

    val results = VisualTripletClassifier.test()

    val outStream = new FileOutputStream(s"$resultsDir/VisualClassifier-Mscoco-test.txt", false)

    ReportHelper.saveEvalResults(outStream, "Visual triplet(within data model)", results)

  }
}

