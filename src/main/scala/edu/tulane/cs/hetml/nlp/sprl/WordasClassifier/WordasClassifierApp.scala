package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import java.io.{FileOutputStream, PrintWriter}
import java.util

import edu.illinois.cs.cogcomp.saul.classifier.Results
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors
import edu.tulane.cs.hetml.nlp.sprl.Eval.SpRLEvaluation
import edu.tulane.cs.hetml.nlp.sprl.Helpers.ReportHelper._
import edu.tulane.cs.hetml.vision._

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConversions._
import scala.collection.mutable.HashMap
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierDataModel._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import me.tongfei.progressbar.ProgressBar
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors._
import edu.tulane.cs.hetml.nlp.sprl.Helpers.ReportHelper
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors._

import scala.collection.mutable

/** Created by Umar on 2017-10-04.
  */

object WordasClassifierApp extends App {

  // Preprocess RefExp
  val stopWords = Array("the", "an", "a")
  var combinedResults = Seq[SpRLEvaluation]()
  var allResults = Seq[SpRLEvaluation]()

  val relWords = Array("below", "above", "between", "not", "behind", "under", "underneath", "front of", "right of",
    "left of", "ontop of", "next to", "middle of")

  val wordFrequency = new HashMap[String, Int]()
  val CLEFGoogleNETReaderHelper = new CLEFGoogleNETReader(imageDataPath)
  val refexpTrainedWords = new RefExpTrainedWordReader(imageDataPath).filteredWords

  val languageHelper = new LanguageHelper()
  val wordToClosetClassifier = new mutable.HashMap[String, String]()
  val trainedWordClassifier = new mutable.HashMap[String, SingleWordasClassifer]()
  val classifierDirectory = s"models/mSpRL/wordclassifer/"

  val writer = new PrintWriter(s"data/mSprl/results/wordclassifier/W2C-Output.txt")
//  val writer = new PrintWriter(s"data/mSprl/results/wordclassifier/missedWordsAfterReplacing.txt")
//  val writerGoogleW2V = new PrintWriter(s"data/mSprl/results/wordclassifier/googleW2CPredictions.txt")
//  val replacedWords = new PrintWriter(s"data/mSprl/results/wordclassifier/replacedWords.txt")

  var missedWords = 0
  var predictedWordCount = 0
  var repWords = 0

  val allImages =
    if(isTrain)
      CLEFGoogleNETReaderHelper.trainImages.toList
    else
      CLEFGoogleNETReaderHelper.testImages.toList

  val allsegments =
    if(!useAnntotatedClef) {
        CLEFGoogleNETReaderHelper.allSegments.filter(s => {allImages.exists(i=> i.getId==s.getAssociatedImageID)})
    } else {
      CLEFGoogleNETReaderHelper.allSegments.take(2).toList
    }

  val allRefExp = CLEFGoogleNETReaderHelper.segRefExp

  val pb = new ProgressBar("Processing Data", allsegments.size)
  pb.start()

  allsegments.foreach(s => {
    if (s.referItExpression != null) {

      var refExp = s.referItExpression.toLowerCase.replaceAll("[^a-z]", " ").replaceAll("( )+", " ").trim

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
          if(isTrain) {
            // Calculate Word Frequency
            if (wordFrequency.contains(p._1.getText)) {
              var value = wordFrequency.get(p._1.getText)
              wordFrequency.update(p._1.getText, wordFrequency(p._1.getText) + 1)
            } else {
              wordFrequency.put(p._1.getText, 1)
            }
          }
        })
      }
    }
    pb.step()
  })
  pb.stop()

  // Populate in Data Model
  //images.populate(CLEFGoogleNETReaderHelper.allImages)
  //segments.populate(allsegments)

  // word-segment pair instances
  val trainInstances = new ListBuffer[WordSegment]()

  if(isTrain) {
    println("Training...")
    // Generate Training Instances for words
    val words = wordFrequency.filter(w => w._2 >= 40).keys

    words.foreach(w => {
      val filteredSegments = allsegments.filter(s => {
        if (s.filteredTokens != null) s.filteredTokens.split(" ").exists(t => t.matches(w)) else false
      })

      if (filteredSegments.nonEmpty) {
        filteredSegments.foreach(i => {
          trainInstances += new WordSegment(w, i, true, false, "")
            // Create Negative Examples - Max 5 from same Image
            val ImageSegs = allsegments.filter(t => t.getAssociatedImageID.equals(i.getAssociatedImageID) &&
              (if (t.filteredTokens != null) !t.filteredTokens.split(" ").exists(tok => tok.matches(w)) else false))
            if (ImageSegs.nonEmpty) {
              val len = if (ImageSegs.size < 5) ImageSegs.size else 5
              for (iter <- 0 until len) {
                val negSeg = ImageSegs(iter)
                trainInstances += new WordSegment(w, negSeg, false, false, "")
              }
            }
        })
        // Training Word classifier
        wordsegments.populate(trainInstances, isTrain)
        val c = new SingleWordasClassifer(w)
        c.modelSuffix = w
        c.modelDir = classifierDirectory
        c.learn(iterations)
        c.save()
        trainInstances.clear()
        wordsegments.clear()
      }
    })
  }

  if(!isTrain) {
    println("Testing...")

    //load Trained classifiers
    loadAllTrainedClassifiers()
    val testInstances = new ListBuffer[WordSegment]()

    val testSegments =
      if (useAnntotatedClef) {
        val ClefAnnReader = new CLEFAnnotationReader(imageDataPath)
        ClefAnnReader.clefSegments.toList
      }
      else
        allsegments

    // Generate Test Instances for each word
    val tokenPhraseMap = mutable.HashMap[String, List[WordSegment]]()
    testSegments.foreach(s => {
      val segWithFeatures = allsegments.filter(seg => seg.getAssociatedImageID.equals(s.getAssociatedImageID))
      if(s.referItExpression!=null) {

        val filterRefExp = if(useAnntotatedClef) filterRefExpression(s.referItExpression.trim) else s.filteredTokens.trim

        val d = new Document(s.getAssociatedImageID)
        val senID = s.getAssociatedImageID + "_" + s.getSegmentId.toString
        val sen = new Sentence(d, senID, 0, filterRefExp.length, filterRefExp)
        val phrases = sentenceToPhraseGenerating(sen)
        val toks = phrases.flatMap(LanguageBaseTypeSensors.phraseToTokenGenerating)
        val headWords = phrases.map(p=> p -> getHeadword(p)).toMap

        //Applying postag
//        val pos = LanguageBaseTypeSensors.getPos(sen).zip(toks).map(x=>x._2.getText->x._1).toMap

        //Create all possible combinations M x N
        val  seg_pairs = toks.groupBy(t=> t.getText).map(t=> t._2.head).flatMap(tok => {
          val tokHeadWord = headWords(tok.getPhrase)
          segWithFeatures.map(sf => {
            new WordSegment(tok.getText, sf, s.getSegmentId==sf.getSegmentId, tok.getText==tokHeadWord.getText, "")
            //new WordSegment(tok.getText, sf, s.getSegmentId==sf.getSegmentId, tok.getText==tokHeadWord.getText, pos(tok.getText))
          })
        }).toList

        if(seg_pairs.nonEmpty) {
          tokenPhraseMap.put(s.getAssociatedImageID + "_" + s.getSegmentId + "_" + s.referItExpression, seg_pairs)
          testInstances ++= seg_pairs
        }
      }
    })

//    val outStream = new FileOutputStream(s"$resultsDir/wordclassifier/WordasClassifier.txt", false)
//    val outStreamCombined = new FileOutputStream(s"$resultsDir/wordclassifier/WordasClassifierCombined.txt", false)
//    var acc = 0.0
//    var count = 0
//    testInstances.groupBy(t=> t.getWord).foreach({
//      w =>
//        try {
//          val c = new SingleWordasClassifer(w._1)
//          c.modelSuffix = w._1
//          c.modelDir = s"models/mSpRL/wordclassifer/"
//          wordsegments.populate(w._2, false)
//          c.load()
//          val result = c.test()
//          ReportHelper.saveEvalResults(outStream, w._1, result)
//          val correct = result.perLabel.map(x=>x.correctSize).sum
//          acc += correct / (w._2.size + 0.0)
//          count += w._2.size
//          println(correct / (w._2.size + 0.0))
//          combineResults(result, w._1)
//        }
//        catch {
//          case _: Throwable =>
//            println("Word " + w._1 + "Classifier not found")
//            allResults = mergeResults(List(new SpRLEvaluation(w._1, 100, 0, 0, w._2.size, 0)), allResults)
//        }
//        wordsegments.clear
//    })
//
//    println("Overall acc: " + acc/count)
//    ReportHelper.saveEvalResults(outStreamCombined, "Combined Results", combinedResults, Seq("false"))
//    outStream.close()
//    outStreamCombined.close()
    //wordsegments.populate(testInstances)


    var count = 0
    var wrong = 0
    tokenPhraseMap.foreach{
      case (uniqueId, wordSegList) =>
        val row = uniqueId.split("_")
        println(uniqueId)
        val predictedSegId = computeMatrix(wordSegList)
        writer.println(predictedSegId)
        if(row(1).toInt == predictedSegId) {
          count += 1
        } else {
          wrong += 1
        }
    }
    writer.close()

    val percentage = count * 100.00f / tokenPhraseMap.size
    println("Correct : " + count + "Wrong: " + wrong + "Percentage: " + percentage)
//
//    writer.println("Total Missed Count: " + missedWords)
//    writer.close()
//
//    writerGoogleW2V.println("Total Predicted Count: " + predictedWordCount)
//    writerGoogleW2V.close()
//
//    replacedWords.println("Total Replaced: " + repWords)
//    replacedWords.close()
  }

  def computeMatrix(instances: List[WordSegment]): Int = {

    val scoresMatrix = instances.groupBy(i => i.getWord).map(w => {
      computeScore(w._1, w._2, w._2.head.getPos)
    }).toList
    val norm = normalizeScores(scoresMatrix)
    val vector = combineScores(norm)
    writer.println(vector.map(_.toDouble))
    val regionId = vector.indexOf(vector.max) + 1
    regionId
  }

  def computeScore(word: String, instances: List[WordSegment], postag : String): List[Double] = {

    var countOnce = false

    val w = instances.map(i => {
//      val score = getWordClassifierScore(word, i)
//      if(score!=0.0)
//        score
//      else if(languageHelper.wordSpellVerifier(word)!="true") {
//        useSpellingClassifier(word, i)
//      }
//      else {
//        val predictedWord = getClosetClassifier(word, postag)
//        if(predictedWord!="") {
//          getWordClassifierScore(predictedWord, i)
//        }
//        else if(!countOnce) {
//          //writer.println(i.getWord + " Missed")
//          missedWords += 1
//          countOnce = true
//          0.0
//        }
//        else
//          0.0
//      }
      getWordClassifierScore(word, i)
    })
    w
  }

  def loadAllTrainedClassifiers(): Unit ={
    refexpTrainedWords.foreach(word => {
      val c = new SingleWordasClassifer(word)
      c.modelSuffix = word
      c.modelDir = classifierDirectory
      c.load()
      trainedWordClassifier.put(word, c)
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

  def getClosetClassifier(word: String, pos: String) : String = {
    // RefExp Trained words
    if(wordToClosetClassifier.contains(word))
      return wordToClosetClassifier(word)

    val threshold = if(pos.toUpperCase.contains("NN")) 0.50 else 0.99

    val scoreVector = refexpTrainedWords.map(r => {
      getGoogleSimilarity(r, word)
    })

    val filtered = scoreVector.filter(s => s > threshold)
    filtered.foreach(f => {
      //writerGoogleW2V.println(s"Word Classifier ${word} -> possible predictions ${refexpTrainedWords.get(scoreVector.indexOf(f))} -> score ${f}")
    })

    if(scoreVector.max > threshold) {
      val index = scoreVector.indexOf(scoreVector.max)
      val predictedWord = refexpTrainedWords.get(index)
      predictedWordCount += 1
      //writerGoogleW2V.println(s"Word Classifier ${word} -> Pos ${pos} -> predicted Classifier ${predictedWord}")
      wordToClosetClassifier.put(word, predictedWord)
      predictedWord
    }
    else {
      wordToClosetClassifier.put(word, "")
      ""
    }
  }
  def useSpellingClassifier(word: String, i: WordSegment) : Double = {
    val result = languageHelper.wordSpellVerifier(word)

    //replacedWords.println(word + " ->" + result)
    repWords += 1
    getWordClassifierScore(word, i)
  }

  def normalizeScores(scoreMatrix: List[List[Double]]):List[List[Double]] = {
    //scoreMatrix.map(w=> w.map(s=>if(s == 0) 0.0 else Math.exp(s)/w.map(x => Math.exp(x)).sum))
    scoreMatrix.map(w=> w.map(s=>if(s == 0) 0.0 else s/w.map(Math.abs).sum))
  }

  def combineScores(scoreMatrix: List[List[Double]]): List[Double] = {
    scoreMatrix.transpose.map(_.sum)
  }

  def combineResults(results: Results, w:String): Unit = {
    val r = convertToEval(results).map {
      case t if t.getLabel.equalsIgnoreCase("true") =>
        new SpRLEvaluation(w, t.getPrecision, t.getRecall, t.getF1, t.getLabeledCount, t.getPredictedCount)
      case t => t
    }
    combinedResults  = mergeResults(r, combinedResults)
  }

  def mergeResults (l1: Seq[SpRLEvaluation], l2: Seq[SpRLEvaluation]): Seq[SpRLEvaluation] = {
    val m = l1 ++ l2
    m
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

  def saveCompeleteResults(): Unit = {
//    var acc = 0.0
//    testInstances.groupBy(t=> t.getWord).foreach({
//      w =>
//        try {
//          val c = new SingleWordasClassifer(w._1)
//          c.modelSuffix = w._1
//          c.modelDir = s"models/mSpRL/wordclassifer/"
//          wordsegments.populate(w._2, false)
//          c.load()
//          val result = c.test()
//          ReportHelper.saveEvalResults(outStream, w._1, result)
//          val correct = result.perLabel.map(x=>x.correctSize).sum
//          acc += correct / (w._2.size + 0.0)
//          count += w._2.size
//          println(correct / (w._2.size + 0.0))
//          combineResults(result, w._1)
//        }
//        catch {
//          case _: Throwable =>
//            println("Word " + w._1 + "Classifier not found")
//            allResults = mergeResults(List(new SpRLEvaluation(w._1, 100, 0, 0, w._2.size, 0)), allResults)
//        }
//        wordsegments.clear
//    })
//
//    println("Overall acc: " + acc/count)
//    ReportHelper.saveEvalResults(outStreamCombined, "Combined Results", combinedResults, Seq("false"))
//    outStream.close()
//    outStreamCombined.close()
  }
}