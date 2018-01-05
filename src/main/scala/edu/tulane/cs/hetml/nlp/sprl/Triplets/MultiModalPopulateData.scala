package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.awt.geom.Rectangle2D

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.documentToSentenceGenerating
import edu.tulane.cs.hetml.nlp.sprl.Helpers._
import MultiModalSpRLDataModel.{segments, _}
import edu.tulane.cs.hetml.nlp.sprl.Triplets.TripletSensors.alignmentHelper
import edu.tulane.cs.hetml.nlp.sprl.Triplets.tripletConfigurator.{isTrain, _}
import edu.tulane.cs.hetml.vision.{ImageTripletReader, Segment, WordSegment}

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

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

    triplets.populate(candidateRelations, isTrain)

    logger.info("Triplet population finished.")
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

