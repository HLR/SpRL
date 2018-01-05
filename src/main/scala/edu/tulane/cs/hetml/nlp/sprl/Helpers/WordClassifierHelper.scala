package edu.tulane.cs.hetml.nlp.sprl.Helpers

import java.io.PrintWriter

import edu.tulane.cs.hetml.vision._

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConversions._
import scala.collection.mutable.HashMap
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierDataModel._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierConfigurator._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors._
import me.tongfei.progressbar.ProgressBar

import scala.collection.mutable
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors._

import scala.io.Source

class WordClassifierHelper {

  val stopWords = Array("the", "an", "a")
  val relWords = Array("below", "above", "between", "not", "behind", "under", "underneath", "front of", "right of",
    "left of", "ontop of", "next to", "middle of")

  val wordFrequency = new HashMap[String, Int]()
  val clefWords = new HashMap[String, Int]()

  //** Trained Words Classifiers (Referit and Clef)
  val trainedWords = new WordasClassifierTrainedWordsReader()
  trainedWords.loadTrainedWords(trainWordsPath)
  val refexpTrainedWords = (trainedWords.filteredWords ++ trainedWords.missingWords).toList
  val trainedWordClassifier = new mutable.HashMap[String, SingleWordasClassifer]()
  val wordToClosetClassifier = new mutable.HashMap[String, String]()

  val missedWordsList = new mutable.HashMap[String, Int]()
  val writerMissedWords = new PrintWriter(s"$resultsDir/wordclassifier/missingWords.txt")

  //** CNN Features for Referit Expression - Segments
  val CLEFGoogleNETReaderHelper = new CLEFGoogleNETReader(imageDataPath)
  val images =
    if(isTrain)
      CLEFGoogleNETReaderHelper.trainImages.toList
    else
      CLEFGoogleNETReaderHelper.testImages.toList

  val referItSegments = CLEFGoogleNETReaderHelper.allSegments.toList

  //** CNN Features for CLEF new Proposed Segments
  val ClefSegReader = new CLEFNewSegmentCNNFeaturesReader()
  val xmlFile = if (isTrain) trainFile else testFile
  ClefSegReader.loadFeatures(imageDataPath,xmlFile, isTrain)

  val clefSegments =  ClefSegReader.clefSegments.toList
  val clefUniqueSegments = ClefSegReader.clefUniqueSegments.toList

  val languageHelper = new LanguageHelper()

  def clefHeadwordSegmentsLemma(): Unit = {
    clefSegments.foreach(s => {
      if (s.getSegmentConcept!="") {
        val concept = getHeadwordLemma(s)
        if(!clefWords.contains(concept))
          clefWords.put(concept, 1)
        s.setSegmentConcept(concept)
      }
    })
  }

  def getHeadwordLemma(s : Segment) : String = {
    val d = new Document(s.getAssociatedImageID)
    val senID = s.getAssociatedImageID + "_" + s.getSegmentId.toString
    val sen = new Sentence(d, senID, 0, s.getSegmentConcept.length, s.getSegmentConcept)
    val phrases = sentenceToPhraseGenerating(sen)
    val toks = phrases.flatMap(LanguageBaseTypeSensors.phraseToTokenGenerating)
    val lemmaConcept = getLemma(toks.head).mkString("")
    lemmaConcept
  }

  def getPhraseHeadwordSegmentScore(phrase: String, segment: Segment) : Double = {

    val testInstances = new ListBuffer[WordSegment]()
    val imgSegs = clefUniqueSegments.filter(s=> s.getAssociatedImageID == segment.getAssociatedImageID)

    //Create all possible combinations M x N
    val segPairs = phrase.split(" ").distinct.flatMap(tok => {
      imgSegs.map(is => {
        new WordSegment(tok, is, false)
      })
    }).toList
    val scoreVector = getPhraseSegmentScores(segPairs)
    if (segment.getSegmentId <= scoreVector.size) {
      scoreVector(segment.getSegmentId - 1)
    } else {
      println("Warning: Mismatched Segment Id")
      0.0
    }
  }

  def preprocessReferIt() = {
    val pb = new ProgressBar("Processing Data", referItSegments.size)
    pb.start()

    referItSegments.foreach(s => {
      if (s.getSegmentConcept != null) {

        var refExp = s.getSegmentConcept.toLowerCase.replaceAll("[^a-z]", " ").replaceAll("( )+", " ").trim

        refExp = filterRefExpression(refExp)
        // Saving filtered tokens for later use
        s.filteredTokens = refExp

        if (refExp != "" && refExp.length > 1) {
          val d = new Document(s.getAssociatedImageID)
          val senID = s.getAssociatedImageID + "_" + s.getSegmentId.toString
          val sen = new Sentence(d, senID, 0, refExp.length, refExp)
          val toks = LanguageBaseTypeSensors.sentenceToTokenGenerating(sen).map(x => getLemma(x).mkString(""))
          //Applying postag
          val pos = LanguageBaseTypeSensors.getPos(sen)
          //Generating token-postag Pair
          val pairs = toks.zip(pos)

          //Storing pos pairs in segment
          pairs.foreach(p => {
          // Calculate Word Frequency
            if (wordFrequency.contains(p._1)) {
                var value = wordFrequency.get(p._1)
                wordFrequency.update(p._1, wordFrequency(p._1) + 1)
            } else {
                wordFrequency.put(p._1, 1)
              }
          })
        }
      }
      pb.step()
    })
    println("Write to file...")
    val printWriter = new PrintWriter(s"$resultsDir/newFilteredWords.txt")
    wordFrequency.filter(w => w._2 >= 5).foreach(x => {
      val s = x._1
      val c = x._2
      printWriter.println(s + "-"  + c)
    })
    printWriter.close()
    pb.stop()
  }

  def testWordClassifiers() = {
    //load Trained classifiers
    loadAllTrainedClassifiers(true)

    val testInstances = new ListBuffer[WordSegment]()

    val testSegments =
      if (useAnntotatedClef)
        clefSegments
      else
        referItSegments

    val referItReader = new ReferItExpressionReader()
    referItReader.loadReferitExpressions(imageDataPath)
    val allReferItExpressions = referItReader.referitExpressions

    // Generate Test Instances for each word
    val tokenPhraseMap = mutable.HashMap[String, List[WordSegment]]()
    val TS = testSegments.filter(t => t.getSegmentConcept.trim != "")

    var correct = 0
    var wrong = 0
    var top_correct = 0
    var top_wrong = 0

    val size = TS.size

    TS.foreach(s => {
      var segWithFeatures =  clefUniqueSegments.filter(seg => seg.getAssociatedImageID.equals(s.getAssociatedImageID))

      if(s.getSegmentConcept!=null) {

        val filterRefExp =
          if(useAnntotatedClef)
            s.getSegmentConcept
          else
            s.filteredTokens.trim

        val d = new Document(s.getAssociatedImageID)
        val senID = s.getAssociatedImageID + "_" + s.getSegmentId.toString
        val sen = new Sentence(d, senID, 0, filterRefExp.length, filterRefExp)
        val phrases = sentenceToPhraseGenerating(sen)
        val toks = phrases.flatMap(LanguageBaseTypeSensors.phraseToTokenGenerating)
        val headWords = phrases.map(p=> p -> getHeadword(p)).toMap

        val seg_pairs = toks.groupBy(t=> t.getText).map(t=> t._2.head).flatMap(tok => {
          val tokHeadWord = headWords(tok.getPhrase)
          segWithFeatures.map(sf => {
            val lemma = getLemma(tok).mkString("")
            s.setSegmentConcept(lemma)
            new WordSegment(lemma, sf, s.getSegmentId==sf.getSegmentId)
          })
        }).toList

        val predictedSegId = predictSegmentId(seg_pairs)
        val top = predictTopSegmentIds(seg_pairs, 5)

        if(s.getSegmentId == predictedSegId) {
          correct += 1
        } else {
          wrong += 1
        }

        if(top.contains(s.getSegmentId)) {
          top_correct += 1
        } else {
          top_wrong += 1
        }
      }
    })

    missedWordsList.toList.sortBy(_._2).foreach(w => {
      writerMissedWords.println(w._1 + "_" + w._2)
    })
    writerMissedWords.close()

    println(TS.count(t => trainedWordClassifier.contains(t.getSegmentConcept.trim.toLowerCase())))

    val percentage = correct * 100.00f / size
    val top_percentage = top_correct * 100.00f / size
    println("Correct : " + correct + "Wrong: " + wrong + "Percentage: " + percentage)
    println("Correct : " + top_correct + "Wrong: " + top_wrong + "Percentage: " + top_percentage)

  }

  def trainOnFrequencyWordClassifiers() = {
    // word-segment pair instances
    val trainInstances = new ListBuffer[WordSegment]()

    // Generate CLEF Headword Lemma's
    clefHeadwordSegmentsLemma()

    // Generate Training Instances for words
    val words = clefWords.keys.toList

//      if(preprocessReferitExp) {
//
//        val filename = trainWordsPath + "newFrequencyWords.txt"
//        for (line <- Source.fromFile(filename).getLines) {
//          val w = line.split("-")
//            wordFrequency.put(w(0), w(1).toInt)
//        }
//        wordFrequency.filter(w => w._2 < 40 && w._2 > 0).keys
//      }
//      else
//        refexpTrainedWords

    words.foreach(w => {

      //** Referit Examples
      val filteredSegments = referItSegments.filter(s => {
        if (s.getSegmentConcept != null) s.getSegmentConcept.split(" ").exists(t => t.matches(w)) else false
      })

      //** CLEF Examples
      val filteredClefSegments = clefSegments.filter(s => s.getSegmentConcept.equalsIgnoreCase(w))

      println(w + "ReferIt -> " + filteredSegments.size + " Clef Examples -> " + filteredClefSegments.size )
      if (filteredSegments.nonEmpty) {
        filteredSegments.foreach(i => {
          trainInstances += new WordSegment(w, i, true)
          // Create Negative Examples - Max 5 from same Image
          val ImageSegs = referItSegments.filter(t => t.getAssociatedImageID.equals(i.getAssociatedImageID) &&
            (if (t.getSegmentConcept != null) !t.getSegmentConcept.split(" ").exists(tok => tok.matches(w)) else false))
          if (ImageSegs.nonEmpty) {
            val len = if (ImageSegs.size < 5) ImageSegs.size else 5
            for (iter <- 0 until len) {
              val negSeg = ImageSegs(iter)
              trainInstances += new WordSegment(w, negSeg, false)
            }
          }
        })

        // Add CLEF Examples in Traing Set
        filteredClefSegments.foreach(s => {
          trainInstances += new WordSegment(w, s, true)
        })

        // Training Word classifier
        wordsegments.populate(trainInstances, isTrain)
        val c = new SingleWordasClassifer(w)
        c.modelSuffix = w
        c.modelDir = classifierPath
        c.learn(iterations)
        c.save()
        trainInstances.clear()
        wordsegments.clear()
      }
    })
  }

  def getPhraseSegmentScores(instances: List[WordSegment]): List[Double] = {

    val scoresMatrix = instances.groupBy(i => i.getWord).map(w => {
      computeScore(w._1, w._2, useWord2VecClassifier)
    }).toList
    val norm = normalizeScores(scoresMatrix)
    val vector = combineScores(norm)
    vector
  }

  def predictSegmentId(instances: List[WordSegment]): Int = {

    val scoresMatrix = instances.groupBy(i => i.getWord).map(w => {
      computeScore(w._1, w._2, useWord2VecClassifier)
    }).toList
    val norm = normalizeScores(scoresMatrix)
    val vector = combineScores(norm)
    if(vector.forall(x=>x == 0))
      return -1
    val regionId = vector.indexOf(vector.max) + 1
    regionId
  }

  def predictTopSegmentIds(instances: List[WordSegment], N: Int): List[Int] = {

    val scoresMatrix = instances.groupBy(i => i.getWord).map(w => {
      computeScore(w._1, w._2, useWord2VecClassifier)
    }).toList
    val norm = normalizeScores(scoresMatrix)
    val combined = combineScores(norm)
    if(combined.forall(x=>x == 0))
      return List()

    val vector = combined.zipWithIndex.sortBy(_._1).reverse
      .take(N).map(_._2+1)
    vector
  }

  def useSpellingClassifier(word: String, i: WordSegment) : Double = {
    val result = languageHelper.wordSpellVerifier(word)
    getWordClassifierScore(result, i)
  }

  def computeScore(word: String, instances: List[WordSegment], useWord2Vec: Boolean): List[Double] = {
    val w = instances.map(i => {
      if(useWord2Vec) {
        val score = getWordClassifierScore(word, i)
        if(score!=0.0)
          score
//        else if(languageHelper.wordSpellVerifier(word)!="true") {
//          useSpellingClassifier(word, i)
//        }
        else {
          val predictedWord = getClosetClassifier(word)
          if(predictedWord!="") {
            getWordClassifierScore(predictedWord, i)
          }
          else
            0.0
        }
      }
      else
        getWordClassifierScore(word, i)
    })
    w
  }

  def loadAllTrainedClassifiers(loadMissedTrained: Boolean): Unit ={
    refexpTrainedWords.foreach(word => {
      val c = new SingleWordasClassifer(word)
      c.modelSuffix = word
      c.modelDir = classifierPath
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
    else {
      missedWordsList.put(word, 0)
      0.0
    }
  }

  def getClosetClassifier(word: String) : String = {
    // RefExp Trained words
    if(wordToClosetClassifier.contains(word))
      return wordToClosetClassifier(word)

    val threshold = 0.0

    val scoreVector = refexpTrainedWords.map(r => {
      getGoogleSimilarity(r, word)
    })

    if(scoreVector.max > threshold) {
      val index = scoreVector.indexOf(scoreVector.max)
      val predictedWord = refexpTrainedWords.get(index)
      wordToClosetClassifier.put(word, predictedWord)
      predictedWord
    }
    else {
      wordToClosetClassifier.put(word, "")
      ""
    }
  }

  def normalizeScores(scoreMatrix: List[List[Double]]):List[List[Double]] = {
    scoreMatrix.map(w=> w.map(s=>if(s == 0) 0.0 else s/w.map(Math.abs).sum))
  }

  def combineScores(scoreMatrix: List[List[Double]]): List[Double] = {
    scoreMatrix.transpose.map(_.sum)
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
  }
}

