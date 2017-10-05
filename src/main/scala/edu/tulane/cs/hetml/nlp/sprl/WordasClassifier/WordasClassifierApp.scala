package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import java.util

import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors
import edu.tulane.cs.hetml.vision._

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConversions._
import scala.collection.mutable.HashMap
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator.suffix
/** Created by Umar on 2017-10-04.
  */
object WordasClassifierApp extends App {

  val CLEFGoogleNETReader = new CLEFGoogleNETReader("data/mSprl/saiapr_tc-12")
  // Preprocess RefExp
  val stopWords = Array ("the", "an", "a")

  val relWords = Array("below", "above", "between", "not", "behind", "under", "underneath", "front of", "right of",
    "left of", "ontop of", "next to", "middle of")

  val wordFrequency = new HashMap[String,Int]()

  val trainsegments = CLEFGoogleNETReader.trainingSegments.toList
  var i = 0
  trainsegments.foreach(s => {
    if(s.refExp!=null) {

      var refExp = s.refExp.toLowerCase.replaceAll("[^a-z]", " ").trim
      var tokenRefExp = refExp.split(" ")
      // Removing Stopwords
      stopWords.foreach(s => {
        tokenRefExp = tokenRefExp.filterNot(t => s.matches(t))
      })
      relWords.foreach(s => {
        tokenRefExp = tokenRefExp.filterNot(t => s.matches(t))
      })
      refExp = tokenRefExp.mkString(" ").trim

      // Saving filtered tokens for later use
      s.filteredTokens = refExp;

      if(refExp != "" && refExp.length > 1) {
        println(s.getAssociatedImageID + "-" + s.getSegmentId + "-" + refExp)
        val d = new Document(s.getAssociatedImageID)
        val senID = s.getAssociatedImageID + "_" + s.getSegmentId.toString
        val sen = new Sentence(d, senID , 0, refExp.length, refExp)
        val toks = LanguageBaseTypeSensors.sentenceToTokenGenerating(sen)
        //Applying postag
        val pos = LanguageBaseTypeSensors.getPos(sen)
        //Generating token-postag Pair
        val pairs = toks.zip(pos)

        //Storing pos pairs in segment
        pairs.foreach(p => {
          val tokenPair = p._1.getText + "," + p._2
          s.tagged.add(tokenPair)

          // Calculate Word Frequency
          if(wordFrequency.contains(p._1.getText)) {
            var value = wordFrequency.get(p._1.getText)
            wordFrequency.update(p._1.getText, wordFrequency(p._1.getText) + 1)
          } else {
            wordFrequency.put(p._1.getText, 1)
          }
        })
      }
    }
  })

  val wordTrainingInstances = new ListBuffer[WordSegment]()
  // Generate Training Instances for words
  wordFrequency.foreach(w => {
    if (w._2 >= 40) {
      val instances = trainsegments.filter(s => {
        if (s.filteredTokens != null) s.filteredTokens.split(" ").exists(t => t.matches(w._1)) else false
      })
      println(w._1 + "->" + w._2 + "->" + instances.size)
      if(instances.size > 0) {
        instances.foreach(i => {
          wordTrainingInstances += new WordSegment(w._1, i)
        })
      }
    }
  })

  // Populate in Data Model
  images.populate(CLEFGoogleNETReader.trainingImages)
  segments.populate(trainsegments)
  wordsegments.populate(wordTrainingInstances)

  //Training the Classifier

  WordasClassifer.modelDir = s"models/mSpRL/wordclassifer/"
  WordasClassifer.modelSuffix = "Multi"
  WordasClassifer.learn(50)
  WordasClassifer.save()
//  val c = new WordasClassifer("biker")
//  c.modelSuffix = "bike"
//  c.modelDir = "data/"
}

