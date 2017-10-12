package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import java.io.FileOutputStream

import edu.illinois.cs.cogcomp.saul.classifier.Results
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.getPos
import edu.tulane.cs.hetml.nlp.sprl.Eval.SpRLEvaluation
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{CandidateGenerator, ReportHelper}
import edu.tulane.cs.hetml.nlp.sprl.Helpers.ReportHelper.convertToEval
import edu.tulane.cs.hetml.nlp.sprl.MultiModalPopulateData.populateRoleDataFromAnnotatedCorpus
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers.SingleWordasClassifer
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import edu.tulane.cs.hetml.vision._
import me.tongfei.progressbar.ProgressBar

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.{HashMap, ListBuffer}

/** Created by Umar on 2017-10-04.
  */

object AnnotationApp extends App {

  val ClefAnnReader = new CLEFAnnotationReader(imageDataPath)
  val testImages = ClefAnnReader.testImages.toList
  val testSegments = ClefAnnReader.testSegments.toList

  populateRoleDataFromAnnotatedCorpus()

  val trCandidates = CandidateGenerator.getTrajectorCandidates(phrases().toList)
  val lmCandidates = CandidateGenerator.getLandmarkCandidates(phrases().toList)

  val trGold = phrases().filter(p => p.containsProperty("TRAJECTOR_id"))
  val lmGold = phrases().filter(p => p.containsProperty("LANDMARK_id"))

  images.populate(testImages)
  segments.populate(testSegments)
  var trcount = 0
  trGold.foreach(t => {
    val result = annotationAnalysis(t)
    if(result=="true") {
      trcount += 1
    }
  })
  println("Total Gold TR -> " + trcount)
  var lmcount = 0
  lmGold.foreach(t => {
    val result = annotationAnalysis(t)
    if(result=="true") {
      lmcount += 1
    }
  })
  println("Total Gold LM -> " + lmcount)

  var candTrcount = 0
  trCandidates.foreach(t => {
    val result = annotationAnalysis(t)
    if(result=="true") {
      candTrcount += 1
    }
  })
  println("Total Candidates TR -> " + candTrcount)

  var candLmcount = 0
  lmCandidates.foreach(t => {
    val result = annotationAnalysis(t)
    if(result=="true") {
      candLmcount += 1
    }
  })
  println("Total Candidates LM -> " + candLmcount)
}