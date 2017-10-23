package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import java.io._

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.tulane.cs.hetml.nlp.BaseTypes.Sentence
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers.SingleWordasClassifer
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator.imageDataPath
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierSensors._

import scala.collection.JavaConversions._
import scala.collection.mutable

/** Created by Umar on 2017-10-20.
  */
object WordasClassifierDataModel extends DataModel {

  val sentences = node[Sentence]
  val images = node[Image]
  val segments = node[Segment]
  val wordsegments = node[WordSegment]
  val expressionsegments = node[ExpressionSegment]

  val trainedWordClassifier = new mutable.HashMap[String, SingleWordasClassifer]()
  val classifierDirectory = "models/mSpRL/wordclassifer/"

  /*
  Edges
   */
  val imageToSegment = edge(images, segments)
  imageToSegment.addSensor(imageToSegmentMatching _)

  val expressionSegmentToSegment = edge(expressionsegments, segments)
  expressionSegmentToSegment.addSensor(expressionToSegmentMatching _)

  val expressionSegmentToImage = edge(expressionsegments, images)
  expressionSegmentToImage.addSensor(expressionToImageMatching _)


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
    e: ExpressionSegment => e.isExpressionAndSegmentMatching
  }

  val expressionSegFeatures = property(expressionsegments, ordered = true) {
    e: ExpressionSegment =>
      e.getSegment.getSegmentFeatures.split(" ").toList.map(_.toDouble)
  }

  val expressionScore = property(expressionsegments) {
    e: ExpressionSegment =>
      expressionScoreArray(e).max
  }

  val expressionScoreArray = property(expressionsegments, ordered = true) {
    e: ExpressionSegment =>
      val scores = e.getExpression.split(" ").map(w => {
        if(trainedWordClassifier.contains(w)) {
          val i = new WordSegment(w, e.getSegment, e.isExpressionAndSegmentMatching, false, "")
          getWordClassifierScore(w, i)
        }
        else
          0.0
      }).toList
      scores
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
      c.modelDir = classifierDirectory
      c.load()
      trainedWordClassifier.put(word, c)
    })
  }
}
