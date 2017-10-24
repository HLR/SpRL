package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors
import edu.tulane.cs.hetml.nlp.sprl.Eval.SpRLEvaluation
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierDataModel._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import edu.tulane.cs.hetml.vision._
import me.tongfei.progressbar.ProgressBar

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

/** Created by Umar on 2017-10-20.
  */

object ExpressionClassifierApp extends App {

  // Preprocess RefExp
  val stopWords = Array("the", "an", "a")
  var combinedResults = Seq[SpRLEvaluation]()

  val relWords = Array("below", "above", "between", "not", "behind", "under", "underneath", "front of", "right of",
    "left of", "ontop of", "next to", "middle of")

  val CLEFGoogleNETReaderHelper = new CLEFGoogleNETReader(imageDataPath)
  val classifierDirectory = s"models/mSpRL/expressionClassifer/"

  val allImages =
    if(isTrain)
      CLEFGoogleNETReaderHelper.trainImages.take(10).toList
    else
      CLEFGoogleNETReaderHelper.testImages.toList

  val allsegments =
    if(!useAnntotatedClef) {
        CLEFGoogleNETReaderHelper.allSegments.filter(s => {allImages.exists(i=> i.getId==s.getAssociatedImageID)})
    } else {
      CLEFGoogleNETReaderHelper.allSegments.toList
    }

  val allDocuments = CLEFGoogleNETReaderHelper.allDocuments.filter(s => {
    val imgID = s.getId.split("_")
    allImages.exists(i=> i.getId==imgID(0))
  })

  val allSentence = CLEFGoogleNETReaderHelper.allSentences.filter(s => {
    val senID = s.getId.split("_")
    allImages.exists(i=> i.getId==senID(0))
  })

  loadWordClassifiers()

  images.populate(allImages, isTrain)
  segments.populate(allsegments, isTrain)
  documents.populate(allDocuments, isTrain)
  sentences.populate(allSentence, isTrain)




/*  val pb = new ProgressBar("Processing Data", allsegments.size)
  pb.start()

  val instances = new ListBuffer[ExpressionSegment]()

  allsegments.foreach(s => {
    if (s.referItExpression != null) {

      val refExp = s.referItExpression.toLowerCase.replaceAll("[^a-z]", " ").replaceAll("( )+", " ").trim

      // Saving filtered tokens for later use
      s.filteredTokens = refExp

      if (refExp != "" && refExp.length > 1) {

        getPostags(s).foreach(p => {
          val tokenPair = p._1.getText + "," + p._2
          s.tagged.add(tokenPair)
        })

        // Create Positive Example
        instances += new ExpressionSegment(s.filteredTokens, s, true)

        // Create Negative Example(s)
        val ImageSegs = allsegments.filter(t => t.getAssociatedImageID.equals(s.getAssociatedImageID) &&
          t.getSegmentId != s.getSegmentId)

        if (ImageSegs.nonEmpty) {
          val len = if (ImageSegs.size < 5) ImageSegs.size else 5
          for (iter <- 0 until len) {
            val negSeg = ImageSegs(iter)
            if (negSeg.referItExpression != "" && negSeg.referItExpression.length > 1) {
              if(negSeg.filteredTokens!=null) {
                instances += new ExpressionSegment(negSeg.filteredTokens, negSeg, false)
              }
              else {
                val exp = negSeg.referItExpression.toLowerCase.replaceAll("[^a-z]", " ").replaceAll("( )+", " ").trim
                instances += new ExpressionSegment(exp, negSeg, false)
              }
            }
          }
        }
      }
    }
    pb.step()
  })
  pb.stop()
*/
  //expressionsegments.populate(instances, isTrain)

  if(isTrain) {
    println("Training...")
    expressionSegmentPairs().foreach(e => println(expressionLabel(e)))
    expressionSegmentPairs().foreach(e => println(expressionScoreArray(e)))

    ExpressionasClassifer.learn(iterations)

    ExpressionasClassifer.save()
  }

  if(!isTrain) {
    println("Testing...")
    ExpressionasClassifer.load()
    ExpressionasClassifer.test()
  }

  def getPostags(s: Segment): List[(Token, String)] ={
    val d = new Document(s.getAssociatedImageID)
    val senID = s.getAssociatedImageID + "_" + s.getSegmentId.toString
    val sen = new Sentence(d, senID, 0, s.filteredTokens.length, s.filteredTokens)
    val toks = LanguageBaseTypeSensors.sentenceToTokenGenerating(sen)
    //Applying postag
    val pos = LanguageBaseTypeSensors.getPos(sen)
    //Generating token-postag Pair
    toks.zip(pos).toList
  }
}