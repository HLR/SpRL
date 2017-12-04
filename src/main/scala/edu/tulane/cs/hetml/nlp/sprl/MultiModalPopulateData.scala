package edu.tulane.cs.hetml.nlp.sprl

import java.io.PrintStream

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel.{triplets, _}
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.documentToSentenceGenerating
import edu.tulane.cs.hetml.nlp.sprl.Helpers._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletsDataModel
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
      segmentRelations.populate(imageReader.getImageRelationList, isTrain)
      visualTriplets.populate(imageReader.getVisualTripletList, isTrain)

      setBestAlignment()
    }

    val suffix = if (isTrain) "train" else "gold"
    val writer = new PrintStream(s"$resultsDir/phrases_$suffix.txt")
    val docs = phraseInstances.groupBy(x => x.getDocument)
    docs.foreach {
      case (doc, phraseList) =>
        val imageId = (documents(doc) ~> documentToImage).head.getId
        val imFolder = doc.getId.split(Array('.', '/'))(1)
        phraseList.foreach {
          p =>
            val seg = (phrases(p) ~> -segmentPhrasePairToPhrase ~> -segmentToSegmentPhrasePair).headOption
            var segStr = "-1\t\t-1\t\t-1\t\t-1\t\t-1"
            if (seg.nonEmpty) {
              segStr = s"${seg.get.getSegmentId}\t\t" +
                s"${seg.get.getBoxDimensions.getX}\t\t${seg.get.getBoxDimensions.getY}\t\t" +
                s"${seg.get.getBoxDimensions.getWidth}\t\t${seg.get.getBoxDimensions.getHeight}\t\t"
            }
            writer.println(s"$imFolder\t\t$imageId\t\t${p.getSentence.getId}\t\t${p.getSentence.getText}\t\t" +
              s"${p.getStart}\t\t${p.getEnd}\t\t${p.getText}\t\t$segStr")
        }
    }
    writer.close()
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
    val visualTriplets =
        (triplets() ~> tripletToVisualTriplet).toList.filter(x => x.getSp != "-")
          .sortBy(x => x.getImageId + "_" + x.getFirstSegId + "_" + x.getSecondSegId)

    VisualTripletsDataModel.visualTriplets.populate(visualTriplets, isTrain)

    val rels = xmlReader.getTripletsWithArguments()
    val suffix = if (isTrain) "train" else "gold"
    val writer = new PrintStream(s"$resultsDir/flat_relation_roles_$suffix.txt")

    rels.foreach {
      r =>
        val sent = r.getParent.asInstanceOf[Sentence]
        val tr = r.getArgument(0)
        val sp = r.getArgument(1)
        val lm = if (r.getArgument(2) != null) r.getArgument(2) else dummyPhrase
        val imId = sent.getDocument.getId.split(Array('.', '/'))(2)
        val imFolder = sent.getDocument.getId.split(Array('.', '/'))(1)

        writer.println(s"$imFolder\t\t$imId\t\t${sent.getId}\t\t${sent.getText}" +
          s"\t\t${tr.getStart}\t\t${tr.getEnd}\t\t${tr.getText}" +
          s"\t\t${sp.getStart}\t\t${sp.getEnd}\t\t${sp.getText}" +
          s"\t\t${lm.getStart}\t\t${lm.getEnd}\t\t${lm.getText}")
    }
    writer.close()
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
              if (pair.getProperty("similarity").toDouble > 0.30) {
                p.addPropertyValue("bestAlignment", pair.getArgumentId(1))
                p.addPropertyValue("bestAlignmentScore", pair.getProperty("similarity"))
              }
            }
          })
        })
    }
  }

}

