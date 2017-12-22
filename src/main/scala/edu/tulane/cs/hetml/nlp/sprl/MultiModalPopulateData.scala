package edu.tulane.cs.hetml.nlp.sprl

import java.awt.geom.Rectangle2D
import java.io.PrintStream

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.documentToSentenceGenerating
import edu.tulane.cs.hetml.nlp.sprl.Helpers._
import edu.tulane.cs.hetml.vision.{ImageTriplet, ImageTripletReader, Segment}
import mSpRLConfigurator._

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

/** Created by Taher on 2017-02-12.
  */

object MultiModalPopulateData extends Logging {

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
      setBestAlignment()
    }

    logger.info("Role population finished.")
  }

  def populatePairDataFromAnnotatedCorpus(indicatorClassifier: Phrase => Boolean,
                                          populateNullPairs: Boolean = true
                                         ): Unit = {

    logger.info("Pair population started ...")
    val phraseInstances = (if (isTrain) phrases.getTrainingInstances.toList else phrases.getTestingInstances.toList)
      .filter(_.getId != dummyPhrase.getId)

    val candidateRelations = CandidateGenerator.generatePairCandidates(phraseInstances, populateNullPairs, indicatorClassifier)
    pairs.populate(candidateRelations, isTrain)

    val relations = if (isTrain) pairs.getTrainingInstances.toList else pairs.getTestingInstances.toList
    xmlReader.setPairTypes(relations, populateNullPairs)

    logger.info("Pair population finished.")
  }

  def populateTripletDataFromAnnotatedCorpusFromPairs(
                                                       trSpFilter: (Relation) => Boolean,
                                                       spFilter: (Phrase) => Boolean,
                                                       lmSpFilter: (Relation) => Boolean
                                                     ): Unit = {

    logger.info("Triplet population started ...")
    val candidateRelations = CandidateGenerator.generateTripletCandidatesFromPairs(
      trSpFilter,
      spFilter,
      lmSpFilter,
      isTrain
    )
    triplets.populate(candidateRelations, isTrain)

    xmlReader.setTripletRelationTypes(candidateRelations)

    logger.info("Triplet population finished.")
  }

  def populateTripletDataFromAnnotatedCorpus(
                                              trFilter: (Phrase) => Boolean,
                                              spFilter: (Phrase) => Boolean,
                                              lmFilter: (Phrase) => Boolean
                                            ): Unit = {

    logger.info("Triplet population started ...")
    val candidateRelations = CandidateGenerator.generateAllTripletCandidates(
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
    val candidateRelations = CandidateGenerator.generatePairCandidates(phrases().toList, populateNullPairs, indicatorClassifier)
    pairs.populate(candidateRelations, isTrain)

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
        s.getSegmentId == p.getPropertyFirstValue("segId").toInt))
    val addNew = alignedPhrases.filter(p => !update.contains(p))

    update.foreach {
      p =>
        val seg = segments.find(x =>
          x.getAssociatedImageID == p.getPropertyFirstValue("imageId") &&
            x.getSegmentId == p.getPropertyFirstValue("segId").toInt
        ).get
        val im = images().find(_.getId == seg.getAssociatedImageID).get
        val x = Math.min(im.getWidth, Math.max(0, p.getPropertyFirstValue("segX").toDouble))
        val y = Math.min(im.getHeight, Math.max(0, p.getPropertyFirstValue("segY").toDouble))
        val w = Math.min(im.getWidth - x, p.getPropertyFirstValue("segWidth").toDouble)
        val h = Math.min(im.getHeight - y, p.getPropertyFirstValue("segHeight").toDouble)

        seg.getBoxDimensions.setRect(x, y, w, h)
    }
    val newSegs = addNew.map {
      p =>
        val imId = p.getPropertyFirstValue("imageId")
        val im = images().find(_.getId == imId).get
        val x = Math.min(im.getWidth, Math.max(0, p.getPropertyFirstValue("segX").toDouble))
        val y = Math.min(im.getHeight, Math.max(0, p.getPropertyFirstValue("segY").toDouble))
        val w = Math.min(im.getWidth - x, p.getPropertyFirstValue("segWidth").toDouble)
        val h = Math.min(im.getHeight - y, p.getPropertyFirstValue("segHeight").toDouble)
        new Segment(imId, p.getPropertyFirstValue("segId").toInt, -1, "", headWordFrom(p), new Rectangle2D.Double(x, y, w, h))
    }

    segments ++ newSegs
  }

  private def setBestAlignment() = {
    alignmentMethod match {
//      case "gold" =>
//        phrases().foreach(p => {
//          if (p.containsProperty("goldAlignment")) {
//            p.addPropertyValue("bestAlignment", p.getPropertyFirstValue("goldAlignment"))
//            p.addPropertyValue("bestAlignmentScore", "1.0")
//          }
//        })
      case _ =>
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
}

