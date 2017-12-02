package edu.tulane.cs.hetml.nlp.sprl.VisualTriplets

import java.io.FileOutputStream

import scala.io.Source
import edu.tulane.cs.hetml.nlp.sprl.Helpers.ReportHelper
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletsDataModel._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator.resultsDir
import scala.util.Random
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

/** Created by Umar on 2017-11-09.
  */
object VisualTripletsApp extends App {

  val flickerTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "Flickr30k.majorityhead")
  val msCocoTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "MSCOCO.originalterm")
  val isTrain = true
  val useBinaryClassifier = false
  val classifierDirectory = if (useBinaryClassifier) s"models/mSpRL/VisualTripletsBinarySPClassifier/" else
    s"models/mSpRL/VisualTriplets/"
  val classifierSuffix = "combined_perceptron"
  val classifierOnSuffix = "combined_on_perceptron"
  val trainTriplets = flickerTripletReader.trainImageTriplets ++ msCocoTripletReader.trainImageTriplets
  val testTriplets = flickerTripletReader.testImageTriplets ++ msCocoTripletReader.testImageTriplets
  val spList = trainTriplets.groupBy(_.getSp.toLowerCase).filter(_._2.size > 0).keys.toList
  //  trainTriplets.filter(x=> !frequent.contains(x.getSp.toLowerCase()))
  //    .foreach(_.setSp("none"))
  //
  //  testTriplets.filter(x=> !frequent.contains(x.getSp.toLowerCase()))
  //    .foreach(_.setSp("none"))


  val onClassifier = new VisualTripletClassifiers.VisualTripletOnClassifier()
  val visualClassifier = new VisualTripletClassifiers.VisualTripletClassifier()

  visualClassifier.modelSuffix = classifierSuffix
  visualClassifier.modelDir = classifierDirectory
  onClassifier.modelSuffix = classifierOnSuffix
  onClassifier.modelDir = classifierDirectory

  if (isTrain) {

    val trainInstances = new ListBuffer[ImageTriplet]()
    val filename = s"$resultsDir/clefprepdata.txt"
    for (line <- Source.fromFile(filename).getLines) {
      val parts = line.split(",")
      val trBox = RectangleHelper.parseRectangle(parts(3), "-")
      val lmBox = RectangleHelper.parseRectangle(parts(4), "-")
      val imageWidth = parts(5).toDouble
      val imageHeight = parts(6).toDouble
      trainInstances += new ImageTriplet(parts(0), parts(1), parts(2), trBox, lmBox
        , imageWidth, imageHeight, parts(7).toDouble, parts(8).toDouble,
        parts(9).toDouble, parts(10).toDouble, parts(11).toDouble, parts(12).toDouble, parts(13).toDouble, parts(14).toDouble,
        parts(15).toDouble, parts(16).toDouble, parts(17).toDouble, parts(18).toDouble,
        RectangleHelper.getBelow(trBox, lmBox, imageHeight), parts(20).toDouble,
        parts(21).toDouble, RectangleHelper.getIntersectionArea(trBox, lmBox, imageWidth * imageHeight), RectangleHelper.getUnionArea(trBox, lmBox, imageWidth * imageHeight)
      )
    }
    if (!useBinaryClassifier) {
      visualTriplets.populate(trainInstances)
      val t = trainInstances.filter(x => x.getSp !="on" && x.getAbove > 0 && x.getBelow == 0&& x.getIntersectionArea>0).toList
      println(trainInstances.count(x => x.getSp =="on" && x.getAbove > 0 && x.getBelow == 0 && x.getIntersectionArea>0))
      println(trainInstances.count(x => x.getSp !="on" && x.getAbove > 0 && x.getBelow== 0&& x.getIntersectionArea>0))
      visualClassifier.learn(50)
      onClassifier.learn(50)
      visualClassifier.save()
      onClassifier.save()
      visualClassifier.test(visualTriplets())
      onClassifier.test(visualTriplets())
    }
    else {
      visualTriplets.populate(trainTriplets, isTrain)
      trainTriplets.groupBy(_.getSp.toLowerCase).filter(x => spList.contains(x._1)).foreach(t => {
        val sp = t._1
        // Training SP classifiers
        val s = new VisualTripletBinarySPClassifier(sp)
        s.modelSuffix = sp
        s.modelDir = classifierDirectory
        s.learn(50)
        println(sp)
        s.test(visualTriplets())
        s.save()
      })
    }
  }
  else {
    if (!useBinaryClassifier) {

      val testInstances = new ListBuffer[ImageTriplet]()
      val filename = s"$resultsDir/clefprepdata-test.txt"
      for (line <- Source.fromFile(filename).getLines) {
        val parts = line.split(",")
        if(parts(0)!="-") {
          val trBox = RectangleHelper.parseRectangle(parts(3), "-")
          val lmBox = RectangleHelper.parseRectangle(parts(4), "-")
          val imageWidth = parts(5).toDouble
          val imageHeight = parts(6).toDouble

          testInstances += new ImageTriplet(parts(0), parts(1), parts(2), trBox,
            lmBox, imageWidth, imageHeight, parts(7).toDouble, parts(8).toDouble,
                      parts(9).toDouble, parts(10).toDouble, parts(11).toDouble, parts(12).toDouble, parts(13).toDouble, parts(14).toDouble,
                      parts(15).toDouble, parts(16).toDouble, parts(17).toDouble, parts(18).toDouble, parts(19).toDouble, parts(20).toDouble,
                      parts(21).toDouble, RectangleHelper.getIntersectionArea(trBox, lmBox, imageWidth * imageHeight),
            RectangleHelper.getUnionArea(trBox, lmBox, imageWidth * imageHeight))
        }
      }

      visualClassifier.load()
      onClassifier.load()
      visualTriplets.populate(testInstances, isTrain)

      val results = visualClassifier.test()
      onClassifier.test()
//      val outStream = new FileOutputStream(s"data/mSprl/results/preposition-prediction-${classifierSuffix}.txt")
//      ReportHelper.saveEvalResults(outStream, "preposition", results)
    }
    else {
      visualTriplets.populate(testTriplets, isTrain)
      testTriplets.groupBy(_.getSp.toLowerCase).foreach(t => {
        val sp = t._1
        if (spList.contains(sp.toLowerCase())) {
          //visualTriplets.populate(t._2, isTrain)
          val s = new VisualTripletBinarySPClassifier(sp)
          s.modelSuffix = sp
          s.modelDir = classifierDirectory
          s.load()
          println(sp)
          val results = s.test()
          val outStream = new FileOutputStream(s"data/mSprl/results/preposition-binary-prediction-${sp}.txt")
          ReportHelper.saveEvalResults(outStream, "binary-preposition", results)
          //visualTriplets.clear()
        }
      })
    }
  }
}

