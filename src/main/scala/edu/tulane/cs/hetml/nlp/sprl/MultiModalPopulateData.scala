package edu.tulane.cs.hetml.nlp.sprl

import java.io.PrintStream

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.documentToSentenceGenerating
import edu.tulane.cs.hetml.nlp.sprl.Helpers._
import edu.tulane.cs.hetml.vision.ImageTripletReader
import mSpRLConfigurator._

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

/** Created by Taher on 2017-02-12.
  */

object MultiModalPopulateData extends Logging {

  lazy val xmlReader = new SpRLXmlReader(if (isTrain) trainFile else testFile, globalSpans)
  lazy val imageReader = new ImageReaderHelper(imageDataPath, trainFile, testFile, isTrain)
  lazy val alignmentReader = new AlignmentReader(alignmentAnnotationPath, alignmentTextPath)

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

    //    if (globalSpans) {
    //      phraseInstances.foreach {
    //        p =>
    //          p.setStart(p.getStart - p.getSentence.getStart)
    //          p.setEnd(p.getEnd - p.getSentence.getStart)
    //      }
    //    }

    alignmentReader.setAlignments(phraseInstances)

    if (populateImages) {
      images.populate(imageReader.getImageList, isTrain)
      segments.populate(imageReader.getSegmentList, isTrain)
      //visualTripletsPairs.populate(imageReader.getVisualTripletList, isTrain)

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
    val visualTripletsFiltered =
      (triplets() ~> tripletToVisualTriplet).toList.filter(x => x.getSp != "-")
        .sortBy(x => x.getImageId + "_" + x.getFirstSegId + "_" + x.getSecondSegId)

    visualTriplets.populate(visualTripletsFiltered, isTrain)

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

  def populateVisualTripletsFromExternalData() : Unit = {
    val flickerTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "Flickr30k.majorityhead")
    val msCocoTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets", "MSCOCO.originalterm")

    val externalTrainTriplets = flickerTripletReader.trainImageTriplets ++ msCocoTripletReader.trainImageTriplets

    if(trainPrepositionClassifier && isTrain) {
      println("Populating Visual Triplets...")
      visualTriplets.populate(externalTrainTriplets, isTrain)
    }
  }

  private def setBestAlignment() = {
    alignmentMethod match {
      case "gold" =>
        phrases().foreach(p => {
          if (p.containsProperty("goldAlignment")) {
            p.addPropertyValue("bestAlignment", p.getPropertyFirstValue("goldAlignment"))
            p.addPropertyValue("bestAlignmentScore", "1.0")
          }
        })
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
              if(pair.getProperty("similarity").toDouble > 0.30 || alignmentMethod=="classifier") {
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

