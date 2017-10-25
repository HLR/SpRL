package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors
import edu.tulane.cs.hetml.nlp.sprl.Eval.SpRLEvaluation
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierDataModel._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import edu.tulane.cs.hetml.vision._
import me.tongfei.progressbar.ProgressBar

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

/** Created by Umar on 2017-10-20.
  */

object ExpressionClassifierApp extends App {

  // Preprocess RefExp
  val stopWords = Array("the", "an", "a")
  var combinedResults = Seq[SpRLEvaluation]()

  val relWords = Array("below", "above", "between", "not", "behind", "under", "underneath", "front of", "right of",
    "left of", "ontop of", "next to", "middle of")

  val CLEFGoogleNETReaderHelper = new CLEFGoogleNETReader(imageDataPath)
  val classifierDirectory = s"models/mSpRL/expressionClassifer/"
  println("Start Reading Data from Files...")
  val allImages =
    if(isTrain)
      CLEFGoogleNETReaderHelper.trainImages.toList
    else
      CLEFGoogleNETReaderHelper.testImages.toList

  val allsegments =
    if(!useAnntotatedClef) {
        CLEFGoogleNETReaderHelper.allSegments.filter(s => {allImages.exists(i=> i.getId==s.getAssociatedImageID)})
    } else {
      CLEFGoogleNETReaderHelper.allSegments.toList
    }

  val allDocuments = CLEFGoogleNETReaderHelper.allDocuments.filter(s => {
    val imgSegId = s.getId.split("_")
    allImages.exists(i=> i.getId==imgSegId(0))
  })

  val allSentence = CLEFGoogleNETReaderHelper.allSentences.filter(d => {
    val senID = d.getId.split("_")
    allImages.exists(i=> i.getId==senID(0))
  })

  loadWordClassifiers()

  images.populate(allImages, isTrain)
  segments.populate(allsegments, isTrain)
  documents.populate(allDocuments, isTrain)
  sentences.populate(allSentence, isTrain)

  if(isTrain) {
    println("Training...")

    ExpressionasClassifer.learn(iterations)
    ExpressionasClassifer.save()
  }

  if(!isTrain) {
    println("Testing...")
    ExpressionasClassifer.load()
    ExpressionasClassifer.test()
  }
}