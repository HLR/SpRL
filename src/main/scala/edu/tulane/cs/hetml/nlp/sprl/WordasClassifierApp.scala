package edu.tulane.cs.hetml.nlp.sprl

import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors
import scala.collection.JavaConversions._
import scala.collection.mutable.HashMap
/** Created by Umar on 2017-10-04.
  */
object WordasClassifierApp extends App {

  val CLEFGoogleNETReader = new CLEFGoogleNETReader("data/mSprl/saiapr_tc-12")
  // Preprocess RefExp
  val stopWords = Array ("the", "an", "a")

  val relWords = Array("below", "above", "between", "not", "behind", "under", "underneath", "front of", "right of",
    "left of", "ontop of", "next to", "middle of")

  val wordFrequency = new HashMap[String,Int]()

  val allsegments = CLEFGoogleNETReader.allSegments.toList
  var count = 0
  allsegments.foreach(s => {
    if(s.refExp!=null) {

      var refExp = s.refExp.toLowerCase.replaceAll("[^a-z]", " ")
      var tokenRefExp = refExp.split(" ")
      // Removing Stopwords
      stopWords.foreach(s => {
        tokenRefExp = tokenRefExp.filterNot(t => s.contains(t))
      })
      relWords.foreach(s => {
        tokenRefExp = tokenRefExp.filterNot(t => s.contains(t))
      })
      refExp = tokenRefExp.mkString(" ").trim

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
    else {
      count = count + 1
    }
  })
  println("Missed" + count)

  count = 0
  wordFrequency.foreach(w => {
    if (w._2 >= 40) {
      println(w._1 + "->" + w._2)
      count = count + 1;
    }
  })
  println(count)
}

