package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import java.io._

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Sentence}
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers.SingleWordasClassifer
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator.imageDataPath
import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierSensors._

import scala.collection.JavaConversions._
import scala.collection.mutable
import edu.tulane.cs.hetml.nlp.BaseTypes._
/** Created by Umar on 2017-10-20.
  */
object WordasClassifierDataModel extends DataModel {

  val documents = node[Document]
  val sentences = node[Sentence]
  val phrases = node[Phrase]
  val images = node[Image]
  val segments = node[Segment]
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
    r: Relation => if(r.getArgumentId(1)=="1") "true" else "false"
  }

  val expressionSegFeatures = property(expressionSegmentPairs, ordered = true) {
    r: Relation => getRelationSegment(r).getSegmentFeatures.split(" ").toList.map(_.toDouble)
  }

  val expressionScore = property(expressionSegmentPairs) {
    r: Relation => expressionScoreArray(r).max
  }

  val expressionScoreArray = property(expressionSegmentPairs, ordered = true) {
    r: Relation => getExpressionScore(r)
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

  def getRelationSegment(r: Relation) : Segment = {
    val imgsegID = r.getArgumentId(0).split("_")
    val s = segments().filter(s => s.getSegmentId == Integer.parseInt(imgsegID(1)) && s.getAssociatedImageID==imgsegID(0))
    s.head
  }

  def getExpressionScore(r: Relation) : List[Double] = {
    val senID = r.getArgumentId(0)
    val sen  = (sentences().filter(s => s.getId==senID)).head
    val seg = getRelationSegment(r)
    val isRel = if(r.getArgumentId(1)=="1") true else false
    val scores = sen.getText.split(" ").map(w => {
      if(trainedWordClassifier.contains(w)) {
        val i = new WordSegment(w, seg, isRel, false, "")
        getWordClassifierScore(w, i)
      }
      else
        0.0
    }).toList
    scores
  }
}
