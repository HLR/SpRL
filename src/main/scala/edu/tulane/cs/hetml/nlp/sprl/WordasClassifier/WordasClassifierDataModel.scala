package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import java.io._

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers.SingleWordasClassifer
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator.imageDataPath
import edu.tulane.cs.hetml.vision._
import scala.collection.JavaConversions._
import scala.collection.mutable

/** Created by Umar on 2017-10-20.
  */
object WordasClassifierDataModel extends DataModel {

  val wordsegments = node[WordSegment]
  val expressionsegments = node[Segment]

  val trainedWordClassifier = new mutable.HashMap[String, SingleWordasClassifer]()
  val classifierDirectory = "models/mSpRL/wordclassifer/"

  val wordLabel = property(wordsegments) {
    w: WordSegment =>
      if (w.isWordAndSegmentMatching)
        w.getWord
      else
        "None"
  }

  val wordSegFeatures = property(wordsegments, ordered = true) {
    w: WordSegment =>
      w.getSegment.getSegmentFeatures.split(" ").toList.map(_.toDouble)
  }

  val expressionLabel = property(expressionsegments) {
    e: Segment => e.isExpressionAndSegmentMatching
  }

  val expressionSegFeatures = property(expressionsegments, ordered = true) {
    e: Segment =>
      e.getSegmentFeatures.split(" ").toList.map(_.toDouble)
  }

  val expressionScore = property(expressionsegments) {
    e: Segment =>
      expressionScoreArray(e).max
  }

  val expressionScoreArray = property(expressionsegments, ordered = true) {
    e: Segment =>
      val scores = e.filteredTokens.split(" ").map(w => {
        if(trainedWordClassifier.contains(w)) {
          val i = new WordSegment(w, e, e.isExpressionAndSegmentMatching, false, "")
          getWordClassifierScore(w, i)
        }
        else
          0.0
      }).toList
      val noramlizeScores = scores.map(s => {
        if(s == 0) 0.0 else s / scores.map(Math.abs).sum
      })
      noramlizeScores
  }

  def getWordClassifierScore(word: String, i: WordSegment) : Double ={
    val scores = trainedWordClassifier(word).classifier.scores(i)
    if(scores.size()>0) {
      val orgValue = scores.toArray.filter(s => s.value.equalsIgnoreCase("true"))
      orgValue(0).score
    }
    else
      0.0
  }
  def loadWordClassifiers(): Unit = {
    val refexpTrainedWords = new RefExpTrainedWordReader(imageDataPath).filteredWords

    refexpTrainedWords.foreach(word => {
      val c = new SingleWordasClassifer(word)
      c.modelSuffix = word
      c.modelDir = s"models/mSpRL/wordclassifer/"
      c.load()
      trainedWordClassifier.put(word, c)
    })
  }
}
