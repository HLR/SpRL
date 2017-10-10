package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import java.io.FileOutputStream
import java.util

import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors
import edu.tulane.cs.hetml.nlp.sprl.Helpers.ReportHelper
import edu.tulane.cs.hetml.vision._

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConversions._
import scala.collection.mutable.HashMap
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers.{SingleWordasClassifer, _}
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalTripletApp.expName
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import me.tongfei.progressbar.ProgressBar

/** Created by Umar on 2017-10-04.
  */

object WordasClassifierApp extends App {

  // Preprocess RefExp
  val stopWords = Array("the", "an", "a")

  val relWords = Array("below", "above", "between", "not", "behind", "under", "underneath", "front of", "right of",
    "left of", "ontop of", "next to", "middle of")

  val wordFrequency = new HashMap[String, Int]()
  val CLEFGoogleNETReaderHelper = new CLEFGoogleNETReader(imageDataPath)

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

  val pb = new ProgressBar("Processing Data", allsegments.size)
  pb.start()

  allsegments.foreach(s => {
    if (s.refExp != null) {

      var refExp = s.refExp.toLowerCase.replaceAll("[^a-z]", " ").trim

      refExp = filterRefExpression(refExp)
      // Saving filtered tokens for later use
      s.filteredTokens = refExp

      if (refExp != "" && refExp.length > 1) {
        val d = new Document(s.getAssociatedImageID)
        val senID = s.getAssociatedImageID + "_" + s.getSegmentId.toString
        val sen = new Sentence(d, senID, 0, refExp.length, refExp)
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
          if (wordFrequency.contains(p._1.getText)) {
            var value = wordFrequency.get(p._1.getText)
            wordFrequency.update(p._1.getText, wordFrequency(p._1.getText) + 1)
          } else {
            wordFrequency.put(p._1.getText, 1)
          }
        })
      }
    }
    pb.step()
  })
  pb.stop()
  // Populate in Data Model
  //images.populate(CLEFGoogleNETReaderHelper.allImages)
  segments.populate(allsegments)

  // word-segment pair instances
  val trainInstances = new ListBuffer[WordSegment]()

  if(isTrain) {
    // Generate Training Instances for words
    val words = wordFrequency.filter(w => w._2 >= 40).keys

    words.foreach(w => {
      val filteredSegments = allsegments.filter(s => {
        if (s.filteredTokens != null) s.filteredTokens.split(" ").exists(t => t.matches(w)) else false
      })

      if (filteredSegments.nonEmpty) {
        filteredSegments.foreach(i => {
          trainInstances += new WordSegment(w, i, true)
            // Create Negative Examples - Max 5 from same Image
            val ImageSegs = allsegments.filter(t => t.getAssociatedImageID.equals(i.getAssociatedImageID) &&
              (if (t.filteredTokens != null) !t.filteredTokens.split(" ").exists(tok => tok.matches(w)) else false))
            if (ImageSegs.nonEmpty) {
              val len = if (ImageSegs.size < 5) ImageSegs.size else 5
              for (iter <- 0 until len) {
                val negSeg = ImageSegs(iter)
                trainInstances += new WordSegment(w, negSeg, false)
              }
            }
        })
        // Training Word classifier
        wordsegments.populate(trainInstances, isTrain)
        val c = new SingleWordasClassifer(w)
        c.modelSuffix = w
        c.modelDir = s"models/mSpRL/wordclassifer/"
        c.learn(iterations)
        c.save()
        trainInstances.clear()
        wordsegments.clear()
      }
    })
  }

  if(!isTrain) {

    val testInstances = new ListBuffer[WordSegment]()


    val ClefAnnReader = new CLEFAnnotationReader(imageDataPath)
    val testSegments = if (useAnntotatedClef)ClefAnnReader.testSegments.toList else allsegments

    // Generate Test Instances for each word
    testSegments.foreach(s => {
      val segWithFeatures = allsegments.filter(seg => seg.getAssociatedImageID.equals(s.getAssociatedImageID))
      if(s.refExp!=null) {
        val filterRefExp = if(useAnntotatedClef) filterRefExpression(s.refExp) else s.filteredTokens

        //Create all possible combinations M x N
        filterRefExp.split(" ").foreach(tok => {
          segWithFeatures.foreach(sf => {
            testInstances += new WordSegment(tok, sf, s.getSegmentId==sf.getSegmentId)
          })
        })
      }
    })
    val outStream = new FileOutputStream(s"$resultsDir/WordasClassifier.txt", false)
    val outStreamCombined = new FileOutputStream(s"$resultsDir/WordasClassifierCombined.txt", false)
    testInstances.groupBy(t=> t.getWord).map({
      w =>
        try {
          val c = new SingleWordasClassifer(w._1)
          c.modelSuffix = w._1
          c.modelDir = s"models/mSpRL/wordclassifer/"
          wordsegments.populate(w._2, false)
          c.load()
          val result = c.test()

          ReportHelper.saveEvalResults(outStream, w._1, result)
          ReportHelper.combineResults(result)

          wordsegments.clear()
        }
        catch {
          case _: Throwable => println("Word " + w + "Classifier not found")
        }
    })
    ReportHelper.saveallResults(outStreamCombined, "Combined Results", ReportHelper.allResults)
    outStream.close()
    outStreamCombined.close()
  }

  def filterRefExpression(refExp: String): String = {
    var tokenRefExp = refExp.split(" ")
    // Removing Stopwords
    stopWords.foreach(s => {
      tokenRefExp = tokenRefExp.filterNot(t => s.matches(t))
    })
    relWords.foreach(s => {
      tokenRefExp = tokenRefExp.filterNot(t => s.matches(t))
    })
    tokenRefExp.mkString(" ").trim
  }
}