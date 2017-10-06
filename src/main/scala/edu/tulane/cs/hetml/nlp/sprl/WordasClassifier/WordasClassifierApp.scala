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
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._

/** Created by Umar on 2017-10-04.
  */

object WordasClassifierApp extends App {

  val CLEFGoogleNETReaderHelper = new CLEFGoogleNETReader(imageDataPath)
  // Preprocess RefExp
  val stopWords = Array ("the", "an", "a")

  val relWords = Array("below", "above", "between", "not", "behind", "under", "underneath", "front of", "right of",
    "left of", "ontop of", "next to", "middle of")

  val wordFrequency = new HashMap[String,Int]()

  val allsegments = CLEFGoogleNETReaderHelper.allSegments.toList


  allsegments.foreach(s => {
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

  // Populate in Data Model
  images.populate(CLEFGoogleNETReaderHelper.allImages)
  segments.populate(allsegments)

  println("Generating Training Dataset")

  // word training instances
  val wordTrainingInstances = new ListBuffer[WordSegment]()

  // complete training instances
  val allTrainingInstances = new ListBuffer[WordSegment]()

  // Generate Training Instances for words
  wordFrequency.foreach(w => {
    if (w._2 >= 40) {
      val instances = allsegments.filter(s => {
        if (s.filteredTokens != null) s.filteredTokens.split(" ").exists(t => t.matches(w._1)) else false
      })
      println(w._1 + "->" + w._2 + "->" + instances.size)
      if(instances.size > 0) {
        instances.foreach(i => {
          wordTrainingInstances += new WordSegment(w._1, i, true)
          allTrainingInstances += new WordSegment(w._1, i, true)
          // Create Negative Examples - Max 5 from same Image
           val ImageSegs = allsegments.filter(t => t.getAssociatedImageID.equals(i.getAssociatedImageID) &&
             (if (t.filteredTokens != null) !t.filteredTokens.split(" ").exists(tok => tok.matches(w._1)) else false))
           if(ImageSegs.size > 0) {
             val len = if(ImageSegs.size < 5) ImageSegs.size else 5
             for (iter <- 0 to len -1) {
               val negSeg = ImageSegs(iter);
               wordTrainingInstances += new WordSegment(w._1, negSeg, false)
               allTrainingInstances += new WordSegment(w._1, negSeg, false)
             }
           }
        })
        // Training Word classifier
        wordsegments.populate(wordTrainingInstances)
        val c = new SingleWordasClassifer(w._1)
        c.modelSuffix = w._1
        c.modelDir = s"models/mSpRL/wordclassifer/"
        c.learn(iterations)
        c.save()
        wordTrainingInstances.clear()
        wordsegments.clear()
      }
    }
  })
  println("Finished Training Dataset")

  wordsegments.populate(allTrainingInstances)
  //Training the Classifier

  WordasClassifer.modelDir = s"models/mSpRL/wordclassifer/"
  WordasClassifer.modelSuffix = "Multi"
  WordasClassifer.learn(iterations)
  WordasClassifer.save()



  // Generating Testing Dataset for Word as Classifer
  val ClefAnnReader = new CLEFAnnotationReader(imageDataPath)

  val testSegments = ClefAnnReader.testSegments
  println("Generating Test Dataset")
  val wordTestInstances = new ListBuffer[WordSegment]()
  // Generate Test Instances for each word
  testSegments.foreach(s => {
    val segWithFeatures = allsegments.filter(seg => seg.getAssociatedImageID.equals(s.getAssociatedImageID))

    //Create all possible combinations M x N
    s.refExp.split(" ").foreach(tok => {
      segWithFeatures.foreach(sf => {
        wordTestInstances += new WordSegment(tok, sf, if(s.getSegmentId==sf.getSegmentId) true else false)
      })
    })
  })

  wordTestInstances.foreach(wt => {
    println(wt.getWord, wt.getSegment.getAssociatedImageID + "-" + wt.getSegment.getSegmentId,wt.getWord2Segment)
  })
}

