package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import java.io.PrintWriter

import edu.stanford.nlp.tagger.maxent.PairsHolder
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.Eval.SpRLEvaluation
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordExpressionSegmentConstraintClassifiers.ExpressionasClassiferConstraintClassifier
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

//  val writer = new PrintWriter(s"data/mSprl/results/wordclassifier/EC-InstanceResults.txt")

  val CLEFGoogleNETReaderHelper = new CLEFGoogleNETReader(imageDataPath)
  val classifierDirectory = s"models/mSpRL/ExpressionClassiferScoreVector/"
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

  if(!useAnntotatedClef) {
    images.populate(allImages, isTrain)
    segments.populate(allsegments, isTrain)
    documents.populate(allDocuments, isTrain)
    sentences.populate(allSentence, isTrain)
  }
  else {
    val ClefAnnReader = new CLEFAnnotationReader(imageDataPath)

    val clefSegments =
      if(useAnntotatedClef) {
        ClefAnnReader.clefSegments.toList
      } else
        List()
    val clefImages = if(useAnntotatedClef) {
      ClefAnnReader.clefImages.toList
    } else
      List()

    val clefSentences = if(useAnntotatedClef) {
      ClefAnnReader.clefSentences.toList
    } else
      List()

    val clefDocuments = if(useAnntotatedClef) {
      ClefAnnReader.clefDocuments.toList
    } else
      List()

    if(useAnntotatedClef) {
      clefSegments.foreach({
        cS =>
          val segWithFeatures = allsegments.filter(seg => seg.getUniqueId.equals(cS.getUniqueId))
          cS.segmentFeatures = segWithFeatures(0).segmentFeatures
      })
    }

    images.populate(clefImages, isTrain)
    segments.populate(clefSegments, isTrain)
    documents.populate(clefDocuments, isTrain)
    sentences.populate(clefSentences, isTrain)
  }

  if(isTrain) {
    println("Training...")
    ExpressionasClassifer.modelDir = classifierDirectory
    ExpressionasClassifer.learn(iterations)
    ExpressionasClassifer.save()
  }

  if(!isTrain) {
    println("Testing...")
    //ExpressionasClassifer.modelDir = classifierDirectory
    var count = 0;
    ExpressionasClassifer.load()
    //ExpressionasClassifer.test()

    ExpressionasClassifer.test(expressionSegmentPairs(), expressionPredictedRelation, expressionActualRelation)
    //ExpressionasClassiferConstraintClassifier.test()

    //ExpressionasClassiferConstraintClassifier.test(expressionSegmentPairs().filter(es => es.getArgumentId(2)=="isRel"))
    //ExpressionasClassifer.test(expressionSegmentPairs().filter(es => es.getArgumentId(2)=="isRel"))
  }
}