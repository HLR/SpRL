package edu.tulane.cs.hetml.nlp.sprl.Anaphora

import java.awt.geom.Rectangle2D
import java.io.PrintWriter

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.{documentToSentenceGenerating, getHeadword}
import edu.tulane.cs.hetml.nlp.sprl.Helpers._
import MultiModalSpRLDataModel.{segments, _}
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.TripletSensors.alignmentHelper
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.tripletConfigurator.{isTrain, _}
import edu.tulane.cs.hetml.relations.RelationInformationReader
import edu.tulane.cs.hetml.vision.{ImageTripletReader, Segment, WordSegment}
import me.tongfei.progressbar.ProgressBar

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors.getGoogleSimilarity
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.tripletConfigurator._
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.MultiModalSpRLTripletClassifiers.TripletRelationClassifier
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.TripletSentenceLevelConstraintClassifiers.TripletRelationConstraintClassifier

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

  def getVGCount(r: Relation): Int = {
    val (tr, sp, lm) = getTripletArguments(r)
    val instances = visualgenomeRelationsList.filter(v => v.getPredicate==sp.getText && v.getSubject==getHeadword(tr) &&
      r.getArgument(2).toString.contains(v.getObject))
    instances.size
  }

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
                                              lmFilter: (Phrase) => Boolean,
                                              changeLabels: Boolean
                                            ): Unit = {

    logger.info("Triplet population started ...")
    val candidateRelations = TripletCandidateGenerator.generateAllTripletCandidates(
      trFilter,
      spFilter,
      lmFilter,
      isTrain
    )
    xmlReader.setTripletRelationTypes(candidateRelations)

    if(changeLabels && useModel=="M2" && isTrain) {
        val newRels = candidateRelations ++ tripletAdditionalExamplesPronounLandmark(candidateRelations, lmFilter)
    }
    else {
        tripletLabelsByCoRef(candidateRelations, lmFilter)
    }
    triplets.populate(candidateRelations, isTrain)
    logger.info("Triplet population finished.")
  }

  def tripletAdditionalExamplesPronounLandmark(canRels: List[Relation], lmFilter: (Phrase) => Boolean): List[Relation] = {

    val implicitLMs = List("it", "them", "him", "her")

    println("Adding Additional Examples...")
    //**
    // Landmark Candidates
    val instances = if (isTrain) phrases.getTrainingInstances else phrases.getTestingInstances
    val landmarks = instances.filter(t => t.getId != dummyPhrase.getId && lmFilter(t)).toList
      .sortBy(x => x.getSentence.getStart + x.getStart)

    //**
    // Headwords of triplets
    relationReader.loadClefRelationHeadwords(imageDataPath, isTrain)
    val clefHWRelation = relationReader.clefHeadwordRelations.toList
    val p = new ProgressBar("Processing Relations", canRels.filter(r => implicitLMs.contains(r.getArgument(2).toString) && r.getProperty("Relation")=="true").size)
    p.start()
    val rels = canRels.filter(r => implicitLMs.contains(r.getArgument(2).toString) && r.getProperty("Relation")=="true")

    rels.foreach(r => {
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
        // get all landmark candidates for the sentence
        val rSId = r.getArgumentId(0).split("\\(")(0)
        val sentenceLMs = landmarks.filter(l => {
          l.getSentence.getId == rSId && l.getText!=r.getArgument(0).toString && l.getText!=r.getArgument(2).toString
        })
        if(sentenceLMs.size>0) {
          val totalInstances = uniqueRelsForLM.map(x => x._2).toList.sum
          //**
          // Similarity scroes for each
          val w2vVectorScores = uniqueRelsForLM.toList.map(t => {
            val rSplited = t._1.split("-")
            if(rSplited.length==3) {
              getMaxW2VScore(rSplited(2), sentenceLMs, t._2, totalInstances)
            }
            else {
              val dummyLM = new Phrase()
              dummyLM.setText("None")
              dummyLM.setId("dummy")
              (dummyLM, -1, 0.0)
            }
          })

          val probableLandmark = w2vVectorScores.sortBy(_._3).last
          if(probableLandmark._3>1.0) {
            val rNew = new Relation()
            rNew.setId(r.getId+"~1")
            rNew.setArgument(0, r.getArgument(0))
            rNew.setArgument(1, r.getArgument(1))
            rNew.setArgument(2, probableLandmark._1)

            rNew.setArgumentId(0, r.getArgumentId(0))
            rNew.setArgumentId(1, r.getArgumentId(1))
            rNew.setArgumentId(2, r.getArgumentId(1))

            rNew.setParent(r.getParent)

            rNew.setProperty("RCC8", r.getProperty("RCC8"))
            rNew.setProperty("Relation", r.getProperty("Relation"))
            rNew.setProperty("FoR", r.getProperty("FoR"))
            rNew.setProperty("SpecificType", r.getProperty("SpecificType"))
            rNew.setProperty("ActualId", r.getProperty("ActualId"))
            rNew.setProperty("GeneralType", r.getProperty("GeneralType"))

            coReferenceTriplets += rNew
          }
        }
      }
    })
    p.stop()

    coReferenceTriplets.toList
  }

  def tripletLabelsByCoRef(candidateRelations: List[Relation], lmFilter: (Phrase) => Boolean) = {
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

    candidateRelations.filter(r => r.getProperty("ImplicitLandmark")=="true").foreach(r => {
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
        // get all landmark candidates for the sentence
        val rSId = r.getArgumentId(0).split("\\(")(0)
        val sentenceLMs = landmarks.filter(l => {
          l.getSentence.getId == rSId && l.getText!=r.getArgument(0).toString && l.getText!=r.getArgument(2).toString
        })
        if(sentenceLMs.size>0){
          //**
          // Similarity scroes for each
          val totalInstances = uniqueRelsForLM.size
          val w2vVectorScores = uniqueRelsForLM.toList.map(t => {
            val rSplited = t._1.split("-")
            if(rSplited.length==3) {
              val v = getMaxW2VScore(rSplited(2), sentenceLMs, t._2, totalInstances)
              v
            }
            else {
              val dummyLM = new Phrase()
              dummyLM.setText("None")
              dummyLM.setId("dummy")
              (dummyLM, -1, 0.0)
            }
          })

          val ProbableLandmark = w2vVectorScores.sortBy(_._3)
          if(useModel=="M1"){
            if (ProbableLandmark.last._3 > 1.0 && ProbableLandmark.last._1.getText != "None") {
              val pLM = headWordFrom(ProbableLandmark.last._1)
              r.setProperty("ProbableLandmark", pLM)
            }
            else {
              r.setProperty("ProbableLandmark", "None")
            }
          }
        }
      }
      else {
        r.setProperty("ProbableLandmark", "None")
      }
    })
    if(!isTrain && useModel=="M2")
      tripletsCoReference.populate(coReferenceTriplets, isTrain)
  }

  def getMaxW2VScore(replacementLM: String, sentenceLMs: List[Phrase], tripletCount: Double, totalTriplets : Double): (Phrase, Int, Double) = {
    val scores = sentenceLMs.map(w => {
      val phraseText = w.getText.replaceAll("[^A-Za-z0-9]", " ")
      phraseText.replaceAll("\\s+", " ");
      w.setText(phraseText)
      val g = getGoogleSimilarity(replacementLM, headWordFrom(w))
      val o = tripletCount / totalTriplets
      val total = g/2 + o/2
      total
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

