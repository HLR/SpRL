package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.awt.geom.Rectangle2D
import java.io.PrintWriter

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.documentToSentenceGenerating
import edu.tulane.cs.hetml.nlp.sprl.Helpers._
import MultiModalSpRLDataModel.{segments, _}
import edu.tulane.cs.hetml.nlp.sprl.Triplets.TripletSensors.alignmentHelper
import edu.tulane.cs.hetml.nlp.sprl.Triplets.tripletConfigurator.{isTrain, _}
import edu.tulane.cs.hetml.relations.RelationInformationReader
import edu.tulane.cs.hetml.vision.{ImageTripletReader, Segment, WordSegment}
import me.tongfei.progressbar.ProgressBar

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors.getGoogleSimilarity
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers.TripletRelationClassifier
import edu.tulane.cs.hetml.nlp.sprl.Triplets.TripletSentenceLevelConstraintClassifiers.TripletRelationConstraintClassifier

/** Created by Taher on 2017-02-12.
  */

object MultiModalPopulateData extends Logging {

  LexiconHelper.path = spatialIndicatorLex
  lazy val xmlTestReader = new SpRLXmlReader(testFile, globalSpans)
  lazy val xmlTrainReader = new SpRLXmlReader(trainFile, globalSpans)

  def xmlReader = if (isTrain) xmlTrainReader else xmlTestReader

  lazy val imageTrainReader = new ImageReaderHelper(imageDataPath, trainFile, testFile, true)
  lazy val imageTestReader = new ImageReaderHelper(imageDataPath, trainFile, testFile, false)

  def imageReader = if (isTrain) imageTrainReader else imageTestReader

  lazy val alignmentTrainReader = new AlignmentReader(alignmentAnnotationPath, true)
  lazy val alignmentTestReader = new AlignmentReader(alignmentAnnotationPath, false)

  val relationReader = new RelationInformationReader();
  relationReader.loadRelations(imageDataPath);
  val visualgenomeRelationsList = relationReader.visualgenomeRelations.toList
  val coReferenceTriplets = new ListBuffer[Relation]()

  def alignmentReader = if (isTrain) alignmentTrainReader else alignmentTestReader

  def populateRoleDataFromAnnotatedCorpus(populateNullPairs: Boolean = true): Unit = {
    logger.info("Role population started ...")
    if (isTrain && onTheFlyLexicon) {
      LexiconHelper.createSpatialIndicatorLexicon(xmlReader)
    }
    documents.populate(xmlReader.getDocuments, isTrain)
    sentences.populate(xmlReader.getSentences, isTrain)

    if (populateNullPairs) {
      phrases.populate(List(dummyPhrase), isTrain)
    }

    val phraseInstances = (if (isTrain) phrases.getTrainingInstances.toList else phrases.getTestingInstances.toList)
      .filter(_.getId != dummyPhrase.getId)

    if (globalSpans) {
      phraseInstances.foreach {
        p =>
          p.setStart(p.getSentence.getStart + p.getStart)
          p.setEnd(p.getSentence.getStart + p.getEnd)
          p.setGlobalSpan(globalSpans)
      }
    }

    xmlReader.setRoles(phraseInstances)

    if (populateImages) {
      alignmentReader.setAlignments(phraseInstances)
      images.populate(imageReader.getImageList, isTrain)
      val segs = getAdjustedSegments(imageReader.getSegmentList)
      segments.populate(segs, isTrain)
      imageSegmentsDic = getImageSegmentsDic()
      if (alignmentMethod != "topN") {
        setBestAlignment()
      }
      else {
        val ws = segmentPhrasePairs().map {
          pair =>
            val s = (segmentPhrasePairs(pair) ~> -segmentToSegmentPhrasePair).head
            val p = (segmentPhrasePairs(pair) ~> segmentPhrasePairToPhrase).head
            val segs = (segments(s) ~> -imageToSegment ~> imageToSegment).toList
            val lemma = headWordLemma(p)
            val wordSegs = segs.map(x => new WordSegment(lemma, x, false))
            val topIds = alignmentHelper.predictTopSegmentIds(wordSegs, tripletConfigurator.topAlignmentCount)
            if (topIds.contains(s.getSegmentId)) {
              val wordSegment = new WordSegment(lemma, s, false)
              wordSegment.setPhrase(p)
              wordSegment
            }
            else
              null
        }.filter(x => x != null)
        wordSegments.populate(ws)
      }

    }

    logger.info("Role population finished.")
  }

  def populateTripletDataFromAnnotatedCorpus(
                                              trFilter: (Phrase) => Boolean,
                                              spFilter: (Phrase) => Boolean,
                                              lmFilter: (Phrase) => Boolean
                                            ): Unit = {

    logger.info("Triplet population started ...")
    val candidateRelations = TripletCandidateGenerator.generateAllTripletCandidates(
      trFilter,
      spFilter,
      lmFilter,
      isTrain
    )
    xmlReader.setTripletRelationTypes(candidateRelations)

    if(useCoReference) {
      candidateRelations.foreach(r => {
        if(r.getArgument(2).toString=="it")
          r.setProperty("ImplicitLandmark","true")
        else
          r.setProperty("ImplicitLandmark","false")
      })
      println("Processing for Co-Reference...")
      //**
      // Landmark Candidates
      val instances = if (isTrain) phrases.getTrainingInstances else phrases.getTestingInstances
      val landmarks = instances.filter(t => t.getId != dummyPhrase.getId && lmFilter(t)).toList
        .sortBy(x => x.getSentence.getStart + x.getStart)

      //**
      // Headwords of triplets
      relationReader.loadClefRelationHeadwords(imageDataPath, isTrain)
      val clefHWRelation = relationReader.clefHeadwordRelations.toList

      val p = new ProgressBar("Processing Relations", candidateRelations.filter(r => r.getProperty("ImplicitLandmark")=="true").size)
      p.start()
      candidateRelations.filter(r => r.getProperty("ImplicitLandmark")=="true").foreach(r => {
        p.step()
        val headWordsTriplet = clefHWRelation.filter(c => {
          c.getId==r.getId
        })
        //**
        // get possible Landmarks from Visual Genome
        val possibleLMs = visualgenomeRelationsList.filter(v => v.getPredicate==headWordsTriplet.head.getSp
          && v.getSubject==headWordsTriplet.head.getTr)
        if(possibleLMs.size>0) {
          //Count Unique Instances
          var uniqueRelsForLM = scala.collection.mutable.Map[String, Int]()
          possibleLMs.foreach(t => {
            val key = t.getSubject + "-" + t.getPredicate + "-" + t.getObject
            if(!(uniqueRelsForLM.keySet.exists(_ == key)))
              uniqueRelsForLM += (key -> 1)
            else {
              var count = uniqueRelsForLM.get(key).get
              count = count + 1
              uniqueRelsForLM.update(key, count)
            }
          })
          //**
          // get all sentence triplets where tr and sp matches
//          val senSameRels = candidateRelations.filter(c => c.getProperty("ActualId")==r.getProperty("ActualId") &&
//            c.getArgument(0).toString==r.getArgument(0).toString && c.getArgument(1).toString==r.getArgument(1).toString)

          val rSId = r.getArgumentId(0).split("\\(")(0)
          val sentenceLMs =
            if(useCrossSentence) {
              val docId = sentences().filter(s => s.getId==rSId).head.getDocument.getId
              val sens = sentences().filter(s => s.getDocument.getId==docId)
              landmarks.filter(l => {
                sens.exists(s => {
                  s.getId==l.getSentence.getId
                }) && l.getText!=r.getArgument(0).toString && l.getText!=r.getArgument(2).toString
              })
            }
            else {
              //**
              // get all landmark candidates for the sentence
              landmarks.filter(l => {
                l.getSentence.getId == rSId && l.getText!=r.getArgument(0).toString && l.getText!=r.getArgument(2).toString
              })
            }

          //**
          // Similarity scroes for each
          val w2vVectorScores = uniqueRelsForLM.toList.map(t => {
            val rSplited = t._1.split("-")
            if(rSplited.length==3) {
              val v = getMaxW2VScore(rSplited(2), sentenceLMs)
              v
            }
            else {
              val dummyLM = new Phrase()
              dummyLM.setText("None")
              dummyLM.setId("dummy")
              (dummyLM, -1, 0.0)
            }
          })

          val ProbableLandmark = w2vVectorScores.sortBy(_._3).last
          if (ProbableLandmark._3 > 0.75 && ProbableLandmark._1.getText != "None") {
            val pLM = headWordFrom(ProbableLandmark._1)
            r.setProperty("ProbableLandmark", pLM)
          }
          else {
            r.setProperty("ProbableLandmark", "None")
          }
          if(!isTrain) {
            val ProbableLandmark = w2vVectorScores.sortBy(_._3)
            var count = 0
            ProbableLandmark.foreach(p => {
              if(p._3>0) {
                val rNew = new Relation()
                rNew.setId(r.getId + "~" + count)
                rNew.setArgument(0, r.getArgument(0))
                rNew.setArgument(1, r.getArgument(1))
                rNew.setArgument(2, p._1)

                rNew.setArgumentId(0, r.getArgumentId(0))
                rNew.setArgumentId(1, r.getArgumentId(1))
                rNew.setArgumentId(2, p._1.getId)

                rNew.setParent(r.getParent)

                rNew.setProperty("RCC8", r.getProperty("RCC8"))
                rNew.setProperty("Relation", r.getProperty("Relation"))
                rNew.setProperty("FoR", r.getProperty("FoR"))
                rNew.setProperty("SpecificType", r.getProperty("SpecificType"))
                rNew.setProperty("ActualId", r.getProperty("ActualId"))
                rNew.setProperty("GeneralType", r.getProperty("GeneralType"))

                coReferenceTriplets += rNew
                count = count + 1
              }
            })
          }
        }
        else {
          r.setProperty("ProbableLandmark", "None")
        }
      })
      p.stop()
    }

    triplets.populate(candidateRelations, isTrain)
    if(!isTrain)
      tripletsCoReference.populate(coReferenceTriplets, isTrain)
//    if(newExperiment) {
//      var correct = 0
//      var wrong = 0
//      var wrongPred = 0
//      var coRefcorrect = 0
//      var coRefwrong = 0
//      var coRefwrongPred = 0
//      triplets().filter(r => r.getProperty("ImplicitLandmark")=="true").foreach(r => {
//        val gRel = r.getProperty("Relation")
//        val pRel = TripletRelationConstraintClassifier(r)
//        if(gRel=="true" && pRel=="true")
//          correct = correct + 1
//        else if(gRel=="true" && pRel=="false")
//          wrong = wrong + 1
//        else if(gRel!="true" && pRel=="true")
//          wrongPred = wrongPred + 1
//        println("R ->" + gRel + "P ->" + pRel)
////        println("Features ->" + JF2_1(r) + " " + JF2_2(r) + " " + JF2_3(r) + " " + JF2_4(r) + " " + JF2_5(r) + " " + JF2_6(r) + " " +
////          JF2_8(r) + " " + JF2_9(r) + " " + JF2_10(r) + " " + JF2_11(r) + " " + JF2_13(r) + " " + JF2_14(r) + " " + JF2_15(r) + " " +
////          tripletSpWithoutLandmark(r) + " " + tripletPhrasePos(r) + " " + tripletDependencyRelation(r) + " " + tripletHeadWordPos(r) + " " +
////          tripletLmBeforeSp(r) + " " + tripletTrBeforeLm(r) + " " + tripletTrBeforeSp(r) + " " +
////          tripletDistanceTrSp(r) + " " + tripletDistanceLmSp(r))
//
//        val impLM = r.getProperty("ProbableLandmark")
//        var found = -1
//        coReferenceTriplets.filter(rNew => rNew.getId==r.getId).foreach(rNew => {
//
//          r.setProperty("ProbableLandmark", rNew.getArgument(2).toString)
//
////          println("Features ->" + JF2_1(r) + " " + JF2_2(r) + " " + JF2_3(r) + " " + JF2_4(r) + " " + JF2_5(r) + " " + JF2_6(r) + " " +
////            JF2_8(r) + " " + JF2_9(r) + " " + JF2_10(r) + " " + JF2_11(r) + " " + JF2_13(r) + " " + JF2_14(r) + " " + JF2_15(r) + " " +
////            tripletSpWithoutLandmark(r) + " " + tripletPhrasePos(r) + " " + tripletDependencyRelation(r) + " " + tripletHeadWordPos(r) + " " +
////            tripletLmBeforeSp(r) + " " + tripletTrBeforeLm(r) + " " + tripletTrBeforeSp(r) + " " +
////            tripletDistanceTrSp(r) + " " + tripletDistanceLmSp(r))
//          val res = TripletRelationConstraintClassifier(r)
//          if(res=="true" && gRel=="true")
//            found = 0
//          else if(res=="false" && gRel=="true")
//            found = 1
//          else if(res=="true" && gRel!="true")
//            found = 2
//        })
//        if(found ==0)
//          coRefcorrect = coRefcorrect + 1
//        else if(found==1)
//          coRefwrong = coRefwrong + 1
//        else if(found==2)
//          coRefwrongPred = coRefwrongPred + 1
//        r.setProperty("ProbableLandmark", impLM)
//      })
//      println("Correct -> " + correct + " Wrong -> " + wrong + " Wrongly Pred -> " + wrongPred)
//      println("CoRefCorrect -> " + coRefcorrect + " CoRefWrong -> " + coRefwrong + " CoRef Wrongly Pred -> " + coRefwrongPred)
//    }

    logger.info("Triplet population finished.")
  }

  def getMaxW2VScore(replacementLM: String, sentenceLMs: List[Phrase]): (Phrase, Int, Double) = {
    val scores = sentenceLMs.map(w => {
      val phraseText = w.getText.replaceAll("[^A-Za-z0-9]", " ")
      phraseText.replaceAll("\\s+", " ");
      w.setText(phraseText)
      getGoogleSimilarity(replacementLM, headWordFrom(w))
    })
    val maxScore = scores.max
    val maxIndex = scores.indexOf(scores.max)
    val possibleLM = sentenceLMs(maxIndex)
    (possibleLM, maxIndex, maxScore)
  }

  def populateDataFromPlainTextDocuments(documentList: List[Document],
                                         indicatorClassifier: Phrase => Boolean,
                                         populateNullPairs: Boolean = true
                                        ): Unit = {

    logger.info("Data population started ...")
    val isTrain = false

    documents.populate(documentList, isTrain)
    sentences.populate(documentList.flatMap(d => documentToSentenceGenerating(d)), isTrain)
    if (populateNullPairs) {
      phrases.populate(List(dummyPhrase), isTrain)
    }
    val spCandidatesTrain = TripletCandidateGenerator.getIndicatorCandidates(phrases().toList)
    val trCandidatesTrain = TripletCandidateGenerator.getTrajectorCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))
    val lmCandidatesTrain = TripletCandidateGenerator.getLandmarkCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))


    logger.info("Triplet population started ...")
    val candidateRelations = TripletCandidateGenerator.generateAllTripletCandidates(
      x => trCandidatesTrain.exists(_.getId == x.getId),
      x => indicatorClassifier(x),
      x => lmCandidatesTrain.exists(_.getId == x.getId),
      isTrain
    )

    triplets.populate(candidateRelations, isTrain)

    logger.info("Data population finished.")
  }

  def populateVisualTripletsFromExternalData(): Unit = {
    val flickerTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "Flickr30k.majorityhead")
    val msCocoTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "MSCOCO.originalterm")

    val externalTrainTriplets = flickerTripletReader.trainImageTriplets ++ msCocoTripletReader.trainImageTriplets

    if (trainPrepositionClassifier && isTrain) {
      println("Populating Visual Triplets from External Dataset...")
      visualTriplets.populate(externalTrainTriplets, isTrain)
    }
  }

  def getAdjustedSegments(segments: List[Segment]): List[Segment] = {

    val alignedPhrases = phrases().filter(_.containsProperty("goldAlignment"))
    val update = alignedPhrases
      .filter(p => segments.exists(s => s.getAssociatedImageID == p.getPropertyFirstValue("imageId") &&
        p.getPropertyValues("segId").exists(_.toInt == s.getSegmentId)))

    update.foreach {
      p =>
        segments.filter(x =>
          x.getAssociatedImageID == p.getPropertyFirstValue("imageId") &&
            p.getPropertyValues("segId").exists(_.toInt == x.getSegmentId)
        ).foreach {
          seg =>
            val im = images().find(_.getId == seg.getAssociatedImageID).get
            val x = Math.min(im.getWidth, Math.max(0, p.getPropertyFirstValue("segX").toDouble))
            val y = Math.min(im.getHeight, Math.max(0, p.getPropertyFirstValue("segY").toDouble))
            val w = Math.min(im.getWidth - x, p.getPropertyFirstValue("segWidth").toDouble)
            val h = Math.min(im.getHeight - y, p.getPropertyFirstValue("segHeight").toDouble)
            if (seg.getBoxDimensions == null)
              seg.setBoxDimensions(new Rectangle2D.Double(x, y, w, h))
            else {
              seg.getBoxDimensions.setRect(x, y, w, h)
            }
        }
    }

    segments
  }

  private def setBestAlignment() = {
    sentences().foreach(s => {
      val phraseSegments = (sentences(s) ~> sentenceToPhrase)
        .toList.flatMap(p => (phrases(p) ~> -segmentPhrasePairToPhrase).toList)
        .sortBy(x => x.getProperty("similarity").toDouble).reverse
      val usedSegments = ListBuffer[String]()
      val usedPhrases = ListBuffer[String]()
      phraseSegments.foreach(pair => {
        if (!usedPhrases.contains(pair.getArgumentId(0)) && !usedSegments.contains(pair.getArgumentId(1))) {
          usedPhrases.add(pair.getArgumentId(0))
          usedSegments.add(pair.getArgumentId(1))
          val p = (segmentPhrasePairs(pair) ~> segmentPhrasePairToPhrase).head
          if (pair.getProperty("similarity").toDouble > 0.30 || alignmentMethod == "classifier") {
            p.addPropertyValue("bestAlignment", pair.getArgumentId(1))
            p.addPropertyValue("bestAlignmentScore", pair.getProperty("similarity"))
          }
        }
      }
      )
    })
  }
}

