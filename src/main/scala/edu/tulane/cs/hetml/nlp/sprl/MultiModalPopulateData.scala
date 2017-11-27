package edu.tulane.cs.hetml.nlp.sprl

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel.{triplets, _}
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.documentToSentenceGenerating
import edu.tulane.cs.hetml.nlp.sprl.Helpers._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletsDataModel
import mSpRLConfigurator._

import scala.collection.JavaConversions._

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
      segmentRelations.populate(imageReader.getImageRelationList, isTrain)
      visualTriplets.populate(imageReader.getVisualTripletList, isTrain)
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
    val visualTriplets = (triplets() ~> tripletToVisualTriplet).toList
    VisualTripletsDataModel.visualTriplets.populate(visualTriplets, isTrain)

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
}

