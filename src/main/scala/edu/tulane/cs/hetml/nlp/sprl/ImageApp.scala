package edu.tulane.cs.hetml.nlp.sprl

import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator.imageDataPath

import scala.collection.JavaConversions._

/** Created by Taher on 2017-02-20.
  */
object ImageApp extends App {

val tripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets")

//  val languageHelper = new LanguageHelper()
//  val a =languageHelper.wordSpellVerifier("buildinng")
//
//  val sim = getGoogleSimilarity("", "")
//  println(a)
//
//val expData = new RefExpTrainedWordReader()
//  expData.ExpClsOutput("data/mSprl/saiapr_tc-12")
//  expData.WordClsOutput("data/mSprl/saiapr_tc-12")
//  expData.findDiff()

  //  val readFullData = true
//
//  val CLEFDataset = new CLEFImageReader("data/mSprl/saiapr_tc-12", "data/mSprl/saiapr_tc-12/newSprl2017_train.xml",
//    "data/mSprl/saiapr_tc-12/newSprl2017_gold.xml", readFullData)
////  val CLEFAnnotations = new CLEFAnnotationReader("data/annotatedFiles")
//
//
//  val imageListTrain = CLEFDataset.trainingImages
//  val segmentListTrain = CLEFDataset.trainingSegments
//  val relationListTrain = CLEFDataset.trainingRelations
//
////  images.populate(imageListTrain)
////  segments.populate(segmentListTrain)
////  segmentRelations.populate(relationListTrain)
//
//  val imageListTest = CLEFDataset.testImages
//  val segementListTest = CLEFDataset.testSegments
//  val relationListTest = CLEFDataset.testRelations

//  images.populate(imageListTest, false)
//  segments.populate(segementListTest, false)
//  segmentRelations.populate(relationListTest, false)

}

