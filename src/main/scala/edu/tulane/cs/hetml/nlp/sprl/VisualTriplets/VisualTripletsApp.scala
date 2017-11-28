package edu.tulane.cs.hetml.nlp.sprl.VisualTriplets

import java.io.FileOutputStream

import edu.tulane.cs.hetml.nlp.sprl.Helpers.ReportHelper
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletsDataModel._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletClassifiers._
import scala.util.Random
import scala.collection.JavaConversions._

/** Created by Umar on 2017-11-09.
  */
object VisualTripletsApp extends App {

  val flickerTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "Flickr30k.majorityhead")
  val msCocoTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "MSCOCO.originalterm")
  val isTrain = false
  val useBinaryClassifier = true
  val classifierDirectory = if (useBinaryClassifier) s"models/mSpRL/VisualTripletsBinarySPClassifier/" else
    s"models/mSpRL/VisualTriplets/"
  val classifierSuffix = "combined_perceptron"
  val trainTriplets = flickerTripletReader.trainImageTriplets ++ msCocoTripletReader.trainImageTriplets
  val testTriplets = flickerTripletReader.testImageTriplets ++ msCocoTripletReader.testImageTriplets
  val spList = trainTriplets.groupBy(_.getSp.toLowerCase).filter(_._2.size > 0)
//  trainTriplets.filter(x=> !frequent.contains(x.getSp.toLowerCase()))
//    .foreach(_.setSp("none"))
//
//  testTriplets.filter(x=> !frequent.contains(x.getSp.toLowerCase()))
//    .foreach(_.setSp("none"))


  if (isTrain) {

    if(!useBinaryClassifier) {
      visualTriplets.populate(trainTriplets)
      VisualTripletClassifier.modelSuffix = classifierSuffix
      VisualTripletClassifier.modelDir = classifierDirectory
      VisualTripletClassifier.learn(50)
      VisualTripletClassifier.save()
      VisualTripletClassifier.test(visualTriplets())
    }
    else {
      val r = new Random(5)
      trainTriplets.groupBy(_.getSp.toLowerCase).foreach(t => {
        val sp = t._1
        // Negative Examples
        val negTriplets = trainTriplets.filter(t=> !t.getSp.equalsIgnoreCase(sp))
        val negExamples = r.shuffle(negTriplets.toList).take(t._2.size * 5)

        // Training SP classifiers
        val examples = t._2 ++ negExamples
        visualTriplets.populate(examples, isTrain)

        val s = new VisualTripletBinarySPClassifier(sp)
        s.modelSuffix = sp
        s.modelDir = classifierDirectory
        s.learn(50)
        println(sp)
        s.test(examples)
        s.save()
        visualTriplets.clear()
      })
    }
  }
  else {
    if(!useBinaryClassifier) {
      VisualTripletClassifier.modelSuffix = classifierSuffix
      VisualTripletClassifier.modelDir = classifierDirectory
      VisualTripletClassifier.load()

      visualTriplets.populate(testTriplets, isTrain)

      val results = VisualTripletClassifier.test()
      val outStream = new FileOutputStream(s"data/mSprl/results/preposition-prediction-${classifierSuffix}.txt")
      ReportHelper.saveEvalResults(outStream, "preposition", results)
    }
    else {
      //visualTriplets.populate(testTriplets, isTrain)
      testTriplets.groupBy(_.getSp.toLowerCase).foreach(t => {
        val sp = t._1
        if(spList.contains(sp.toLowerCase())) {
          visualTriplets.populate(t._2, isTrain)
          val s = new VisualTripletBinarySPClassifier(sp)
          s.modelSuffix = sp
          s.modelDir = classifierDirectory
          s.load()
          println(sp)
          val results = s.test()
          val outStream = new FileOutputStream(s"data/mSprl/results/preposition-binary-prediction-${sp}.txt")
          ReportHelper.saveEvalResults(outStream, "binary-preposition", results)
          visualTriplets.clear()
        }
      })
    }
  }
}

