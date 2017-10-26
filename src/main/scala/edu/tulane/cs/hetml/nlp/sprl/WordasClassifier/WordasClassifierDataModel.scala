package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import java.io._

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Sentence}
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers.SingleWordasClassifer
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator.imageDataPath
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierSensors._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import scala.collection.JavaConversions._
import scala.collection.mutable
import edu.tulane.cs.hetml.nlp.BaseTypes._
/** Created by Umar on 2017-10-20.
  */
object WordasClassifierDataModel extends DataModel {

  val documents = node[Document]
  val sentences = node[Sentence]((s: Sentence) => s.getId)
  val images = node[Image]((i: Image) => i.getId)
  val segments = node[Segment]((s: Segment) => s.getUniqueId)
  val wordsegments = node[WordSegment]
  val expressionSegmentPairs = node[Relation]((r: Relation) => r.getId)

  val trainedWordClassifier = new mutable.HashMap[String, SingleWordasClassifer]()
  val classifierDirectory = "models/mSpRL/wordclassifer/"

  /*
  Edges
   */
  val imageToSegment = edge(images, segments)
  imageToSegment.addSensor(imageToSegmentMatching _)

  val documentToSentence = edge(documents, sentences)
  documentToSentence.addSensor(documentToSentenceMatching _)

  val documentToImage = edge(documents, images)
  documentToImage.addSensor(documentToImageMatching _)

  val sentenceToSegmentSentencePairs = edge(sentences, expressionSegmentPairs)
  sentenceToSegmentSentencePairs.addSensor(sentenceToSegmentSentencePairsMatching _)

  val expsegpairToFirstArg = edge(expressionSegmentPairs, sentences)
  expsegpairToFirstArg.addSensor(relationToFirstArgumentMatching _)

  val expsegpairToSecondArg = edge(expressionSegmentPairs, segments)
  expsegpairToSecondArg.addSensor(relationToSecondArgumentMatching _)

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

  val expressionLabel = property(expressionSegmentPairs) {
    r: Relation =>
        if(r.getArgumentId(2)=="1") "true" else "false"
  }

  val expressionSegFeatures = property(expressionSegmentPairs, ordered = true) {
    r: Relation => val s = (expressionSegmentPairs(r) ~> expsegpairToSecondArg).head
        s.getSegmentFeatures.split(" ").toList.map(_.toDouble)
  }

  val expressionScore = property(expressionSegmentPairs) {
    r: Relation =>
//      if(isTrain)
        r.getArgumentId(3).toDouble
//      else
//        getExpressionScore(r)
  }

  val expressionScoreVector = property(expressionSegmentPairs, ordered = true) {
    r: Relation => getExpressionScore(r)
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

  def getExpressionScore(r: Relation) : Double = {
    val sen  = (expressionSegmentPairs(r) ~> expsegpairToFirstArg).head
    val seg = (expressionSegmentPairs(r) ~> expsegpairToSecondArg).head
    val allsegs = segments().filter(s => s.getAssociatedImageID==seg.getAssociatedImageID)

    val instances = sen.getText.split(" ").map(w => {
       allsegs.map(s => new WordSegment(w, s, s.getUniqueId==seg.getUniqueId, false, "")).toList
    }).toList

    val scoresMatrix = instances.flatten.groupBy(i => i.getWord).map(w => {
      computeScore(w._1, w._2)
    }).toList
    val norm = normalizeScores(scoresMatrix)
    val vector = combineScores(norm)
    val regionId = vector.indexOf(vector.max) + 1
    vector.max
  }

  def computeScore(word: String, instances: List[WordSegment]): List[Double] = {
    instances.map(i => {
      getWordClassifierScore(word, i)
    })
  }

  def getWordClassifierScore(word: String, i: WordSegment) : Double ={
    if(trainedWordClassifier.contains(word)) {
      val c = trainedWordClassifier(word)
      val scores = c.classifier.scores(i)
      if(scores.size()>0) {
        val orgValue = scores.toArray.filter(s => s.value.equalsIgnoreCase("true"))
        orgValue(0).score
      }
      else {
        0.0
      }
    }
    else
      0.0
  }

  def normalizeScores(scoreMatrix: List[List[Double]]):List[List[Double]] = {
    scoreMatrix.map(w=> w.map(s=>if(s == 0) 0.0 else s/w.map(Math.abs).sum))
  }

  def combineScores(scoreMatrix: List[List[Double]]): List[Double] = {
    scoreMatrix.transpose.map(_.sum)
  }
}
