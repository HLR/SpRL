package edu.tulane.cs.hetml.nlp.sprl

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel.{triplets, _}
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.documentToSentenceGenerating
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{CandidateGenerator, ImageReaderHelper, LexiconHelper, SpRLXmlReader}
import mSpRLConfigurator._

import scala.collection.JavaConversions._

/** Created by Taher on 2017-02-12.
  */

object MultiModalPopulateData extends Logging{

  lazy val xmlReader = new SpRLXmlReader(if (isTrain) trainFile else testFile)
  lazy val imageReader = new ImageReaderHelper(imageDataPath, trainFile, testFile, isTrain)

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

    xmlReader.setRoles(phraseInstances)

    if(populateImages) {
      images.populate(imageReader.getImageList, isTrain)
      segments.populate(imageReader.getSegmentList, isTrain)
      segmentRelations.populate(imageReader.getImageRelationList, isTrain)
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

  def populateTripletDataFromAnnotatedCorpus(
                                              trSpFilter: (Relation) => Boolean,
                                              spFilter: (Phrase) => Boolean,
                                              lmSpFilter: (Relation) => Boolean
                                            ): Unit = {

    logger.info("Triplet population started ...")
    val candidateRelations = CandidateGenerator.generateTripletCandidates(
      trSpFilter,
      spFilter,
      lmSpFilter,
      isTrain
    )
    triplets.populate(candidateRelations, isTrain)

    xmlReader.setTripletRelationTypes(candidateRelations)

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
  /*def populateTripletGroundTruth() {
    val groundTruthTriplets = if(isTrain) new SpRLXmlReader(trainFile).getTripletsWithArguments() else new SpRLXmlReader(testFile).getTripletsWithArguments()

    val instances = if (isTrain) phrases.getTrainingInstances else phrases.getTestingInstances


    groundTruthTriplets.foreach(t => {
      var fullID = ""
      val gsen = t.getParent.getId
      val tra = instances.filter(i => {
        val sen = phrases(i) <~ sentenceToPhrase
        ((i.getText.trim == t.getArgument(0).getText.trim ||
          partof(i.getText.trim, t.getArgument(0).getText.trim))
          && gsen == sen.head.getId)
      })
      if (tra.nonEmpty) {
        t.setArgumentId(0, tra.head.getId)
        fullID = tra.head.getId
      }
      val sp = instances.filter(i => {
        val sen = phrases(i) <~ sentenceToPhrase
        ((i.getText.trim == t.getArgument(1).getText.trim ||
          partof(i.getText.trim, t.getArgument(1).getText.trim))
          && gsen == sen.head.getId)
      })
      if(sp.nonEmpty) {
        t.setArgumentId(1, sp.head.getId)
        fullID = fullID + "_" + sp.head.getId
      }
      val lm = instances.filter(i => {
        val sen = phrases(i) <~ sentenceToPhrase
        var lmText = t.getArgument(2).getText
        if(lmText!=null)
          ((i.getText.trim == lmText.trim ||
            partof(i.getText.trim, t.getArgument(2).getText.trim))
            && gsen == sen.head.getId)
        else
          false
      })
      if(lm.nonEmpty) {
        t.setArgumentId(2, lm.head.getId)
        fullID = fullID + "_" + lm.head.getId
      }
      else {
        t.setArgumentId(2, dummyPhrase.getId)
        fullID = fullID + "_" + dummyPhrase.getId
      }
      t.setId(fullID)
    })
    triplets.populate(groundTruthTriplets, isTrain)

    xmlReader.setTripletRelationTypes(groundTruthTriplets)

  }*/

  def populateAllTripletsFromPhrases() : Unit = {

    val allTriplets = CandidateGenerator.generateAllTripletCandidate()

    triplets.populate(allTriplets, isTrain)

    xmlReader.setTripletRelationTypes(allTriplets)
  }
}

