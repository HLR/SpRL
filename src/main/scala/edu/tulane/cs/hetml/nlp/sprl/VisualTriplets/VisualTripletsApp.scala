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

  val frequentPrepositions = List("in", "on", "between", "in front of", "behind", "above", "in between", "around",
    "over", "at", "next to")

  val isTrain = false
  val useVGdata = false
  val finetunePrep = false
  val classifierDirectory = s"models/mSpRL/VisualTriplets/"
  val classifierSuffix = "vg_perceptron"
  val cleftestInstances = new ListBuffer[ImageTriplet]()
  val clefTrainInstances = new ListBuffer[ImageTriplet]()

  val VGJsonReader = new JSONReader()

  val (trainingInstances, testInstances) =
    if(useVGdata) {
    VGJsonReader.readJsonFile("data/mSprl/saiapr_tc-12/VGData/")
    val vgFrequentSPTriplets = VGJsonReader.allImageTriplets.filter(x => frequentPrepositions.contains(x.getSp))
    val trainingSize = vgFrequentSPTriplets.size() * 0.8
    vgFrequentSPTriplets.splitAt(trainingSize.toInt)
    }
    else
      (List(), List())

  if (isTrain) {

    visualTriplets.populate(trainingInstances)

    VisualTripletClassifier.learn(50)
    VisualTripletClassifier.save()
    VisualTripletClassifier.test(visualTriplets())

    //fine tune
    if(finetunePrep) {
      visualTriplets.clear()
      visualTriplets.populate(clefTrainInstances)
      VisualTripletClassifier.learn(10)
      VisualTripletClassifier.save()
    }
  }
  else {

    VisualTripletClassifier.load()
    loadClefTestData()
    visualTriplets.populate(cleftestInstances, isTrain) //testTriplets, isTrain)
    val results = VisualTripletClassifier.test()
    val outStream = new FileOutputStream(s"$resultsDir/VisualClassifier-VG-test.txt", false)
    ReportHelper.saveEvalResults(outStream, "Visual triplet(within data model)", results)
  }

  def loadClefTestData() = {

    val filename = s"$resultsDir/clef_prep_test.txt"
    for (line <- Source.fromFile(filename).getLines) {
      val parts = line.split(",")
      if(parts(0)!="-") {
        val trBox = RectangleHelper.parseRectangle(parts(3), "-")
        val lmBox = RectangleHelper.parseRectangle(parts(4), "-")
        val imageWidth = parts(5).toDouble
        val imageHeight = parts(6).toDouble
        parts(0) = parts(0).replaceAll("_", " ")
        cleftestInstances += new ImageTriplet(parts(0), parts(1), parts(2), trBox, lmBox, imageWidth, imageHeight)
      }
    }
  }

  def clefTrainData() = {

        val filename = s"$resultsDir/clefprepdata.txt"
        for (line <- Source.fromFile(filename).getLines) {
          val parts = line.split(",")
          val trBox = RectangleHelper.parseRectangle(parts(3), "-")
          val lmBox = RectangleHelper.parseRectangle(parts(4), "-")
          val imageWidth = parts(5).toDouble
          val imageHeight = parts(6).toDouble
          clefTrainInstances += new ImageTriplet(parts(0), parts(1), parts(2), trBox, lmBox, imageWidth, imageHeight)
        }
  }

  def getInstances() : (List[ImageTriplet], List[ImageTriplet]) = {
      val flickerTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "Flickr30k.majorityhead")
      val msCocoTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "MSCOCO.originalterm")
      val trainTriplets = flickerTripletReader.trainImageTriplets ++ msCocoTripletReader.trainImageTriplets
      val testTriplets = flickerTripletReader.testImageTriplets ++ flickerTripletReader.testImageTriplets
    (trainTriplets.toList, testTriplets.toList)
  }
}

