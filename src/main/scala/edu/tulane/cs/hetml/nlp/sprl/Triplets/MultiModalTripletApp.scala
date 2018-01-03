package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.io.{File, FileOutputStream, PrintStream}

import edu.illinois.cs.cogcomp.saul.classifier.{ConstrainedClassifier, JointTrainSparsePerceptron, Learnable}
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation, Sentence}
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{FeatureSets, ReportHelper}
import MultiModalPopulateData._
import MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.TripletSentenceLevelConstraintClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletClassifiers.VisualTripletClassifier
import tripletConfigurator._
import org.apache.commons.io.FileUtils

object MultiModalTripletApp extends App with Logging {

  val expName = "triplet_" + ((model, useConstraints, usePrepositions) match {
    case (FeatureSets.BaseLine, false, _) => "BM"
    case (FeatureSets.BaseLine, true, _) => "BM+C"
    case (FeatureSets.BaseLineWithImage, false, _) => "BM+I_" + alignmentMethod + "_alignment"
    case (FeatureSets.BaseLineWithImage, true, false) => "BM+C+I_" + alignmentMethod + "_alignment"
    case (FeatureSets.BaseLineWithImage, true, true) => "BM+C+I+Prep_" + alignmentMethod +
      (if (alignmentMethod == "topN") "=" + topAlignmentCount else "") +
      "_alignment"
    case (FeatureSets.WordEmbedding, false, _) => "BM+E"
    case (FeatureSets.WordEmbedding, true, _) => "BM+C+E"
    case (FeatureSets.WordEmbeddingPlusImage, false, _) => "BM+E+I"
    case (FeatureSets.WordEmbeddingPlusImage, true, _) => "BM+C+E+I"
    case _ =>
      logger.error("experiment is not supported")
      System.exit(1)
  })
  MultiModalSpRLTripletClassifiers.featureSet = model

  val constraintRoleClassifiers = List[ConstrainedClassifier[Phrase, Sentence]](
    TRConstraintClassifier,
    LMConstraintClassifier,
    IndicatorConstraintClassifier
  )

  val roleClassifiers = List[Learnable[Phrase]](
    TrajectorRoleClassifier,
    LandmarkRoleClassifier,
    IndicatorRoleClassifier
  )

  val constraintTripletClassifiers = List[ConstrainedClassifier[Relation, Sentence]](
    TripletRelationConstraintClassifier,
    TripletGeneralTypeConstraintClassifier,
    TripletRegionConstraintClassifier,
    TripletDirectionConstraintClassifier
  )

  val tripletClassifiers = List[Learnable[Relation]](
    TripletRelationClassifier,
    TripletGeneralTypeClassifier,
    TripletRegionClassifier,
    TripletDirectionClassifier
  )

  // load all classifiers form all lists
  val classifiers = roleClassifiers ++ tripletClassifiers
  classifiers.foreach(x => {
    x.modelDir = s"models/mSpRL/triplet/$featureSet/"
    x.modelSuffix = suffix
  })
  if (usePrepositions) {
    PrepositionClassifier.modelDir = s"models/mSpRL/triplet/$featureSet/"
    PrepositionClassifier.modelSuffix = suffix
  }

  FileUtils.forceMkdir(new File(resultsDir))

  // train prepositions from external data
  if (isTrain && trainPrepositionClassifier && usePrepositions) {
    populateVisualTripletsFromExternalData()
    PrepositionClassifier.learn(iterations)
    visualTriplets.clear()
  }

  populateRoleDataFromAnnotatedCorpus()

  if (isTrain) {
    println("training started ...")

    roleClassifiers.foreach {
      x =>
        x.learn(iterations)
        x.test(phrases())
        x.save()
    }

    val spCandidatesTrain = TripletCandidateGenerator.getIndicatorCandidates(phrases().toList)
    val trCandidatesTrain = TripletCandidateGenerator.getTrajectorCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))
    val lmCandidatesTrain = TripletCandidateGenerator.getLandmarkCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))


    populateTripletDataFromAnnotatedCorpus(
      x => trCandidatesTrain.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "true",
      x => lmCandidatesTrain.exists(_.getId == x.getId)
    )


    tripletClassifiers.foreach {
      x =>
        x.learn(iterations)
        x.test(triplets())
        x.save()
    }

    if (populateImages) {
      val gtRels = triplets().filter(x => tripletIsRelation(x) == "Relation"
        && x.getArgument(0).containsProperty("goldAlignment") && x.getArgument(2).containsProperty("goldAlignment"))
        .toList
      ImageTripletTypeClassifier.learn(iterations, gtRels)
      ImageTripletTypeClassifier.modelDir = s"models/mSpRL/triplet/$featureSet/"
      ImageTripletTypeClassifier.save()

      ReportHelper.saveImageTripletErrorTypes(gtRels,
        r => triplets(r) ~> tripletToVisualTriplet,
        resultsDir,
        isTrain,
        r => tripletSpecificType(r),
        r => ImageTripletTypeClassifier(r)
      )
    }

    if (usePrepositions && trainPrepositionClassifier) {

      val visualTripletsFiltered = visualTriplets().toList.filter(x => x.getSp != null)
      logger.info("Aligned visual triplets in train:" + visualTripletsFiltered.size)

      PrepositionClassifier.learn(10, visualTripletsFiltered)
      PrepositionClassifier.test(visualTripletsFiltered)
      PrepositionClassifier.save()
    }
    if (jointTrain) {
      logger.info("====================================")
      logger.info("Joint Train started")
      JointTrainSparsePerceptron.train(MultiModalSpRLDataModel.sentences,
        List(TripletRelationConstraintClassifier), 10)
    }

  }

  if (trainTestTogether) {
    documents.clear()
    sentences.clear()
    images.clear()
    segments.clear()
    phrases.clear()
    tokens.clear()
    triplets.clear()
    visualTriplets.clear()
    segmentPhrasePairs.clear()

    isTrain = false
    populateRoleDataFromAnnotatedCorpus()
  }

  if (!isTrain) {

    println("testing started ...")

    if (!trainTestTogether) {
      if (usePrepositions)
        //PrepositionClassifier.load()
        VisualTripletClassifier.load()
      classifiers.foreach(x => x.load())
    }

    val spCandidatesTest = TripletCandidateGenerator.getIndicatorCandidates(phrases.getTestingInstances.toList)
    val trCandidatesTest = TripletCandidateGenerator.getTrajectorCandidates(phrases.getTestingInstances.toList)
      .filterNot(x => spCandidatesTest.contains(x))
    val lmCandidatesTest = TripletCandidateGenerator.getLandmarkCandidates(phrases.getTestingInstances.toList)
      .filterNot(x => spCandidatesTest.contains(x))

    populateTripletDataFromAnnotatedCorpus(
      x => trCandidatesTest.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "true",
      x => lmCandidatesTest.exists(_.getId == x.getId))

    if (populateImages) {
      val gtRels = triplets().filter(x => tripletIsRelation(x) == "Relation"
        && x.getArgument(0).containsProperty("goldAlignment") && x.getArgument(2).containsProperty("goldAlignment"))
        .toList

      ImageTripletTypeClassifier.modelDir = s"models/mSpRL/triplet/$featureSet/"
      ImageTripletTypeClassifier.load()
      ImageTripletTypeClassifier.test(gtRels)

      ReportHelper.saveImageTripletErrorTypes(gtRels,
        r => triplets(r) ~> tripletToVisualTriplet,
        resultsDir,
        isTrain,
        r => tripletSpecificType(r),
        r => ImageTripletTypeClassifier(r)
      )
    }

    if (!useConstraints) {
      val visualTripletsFiltered = visualTriplets.getTestingInstances.toList.filter(x => x.getSp != null)
      val outStream = new FileOutputStream(s"$resultsDir/$expName$suffix.txt", false)

      roleClassifiers.foreach {
        x =>
          val res = x.test()
          ReportHelper.saveEvalResults(outStream, s"${x.getClassSimpleNameForClassifier}(within data model)", res)
      }

      tripletClassifiers.foreach {
        x =>
          val res = x.test()
          ReportHelper.saveEvalResults(outStream, s"${x.getClassSimpleNameForClassifier}(within data model)", res)
      }
      if (usePrepositions && visualTripletsFiltered.nonEmpty) {
        val prepResult = PrepositionClassifier.test(visualTripletsFiltered)
        ReportHelper.saveEvalResults(outStream, s"Preposition(within data model)", prepResult)
      }
      reportForErrorAnalysis(x => TripletRelationClassifier(x),
        x => TrajectorRoleClassifier(x),
        x => LandmarkRoleClassifier(x),
        x => IndicatorRoleClassifier(x),
        x => TripletGeneralTypeClassifier(x),
        x => TripletDirectionClassifier(x),
        x => TripletRegionClassifier(x)
      )
    }
    else {

      val visualTripletsFiltered = visualTriplets.getTestingInstances.toList.filter(x => x.getSp != null)

      val outStream = new FileOutputStream(s"$resultsDir/$expName$suffix.txt", false)

      constraintRoleClassifiers.foreach {
        x =>
          val res = x.test()
          ReportHelper.saveEvalResults(outStream, s"${x.getClassSimpleNameForClassifier}(within data model)", res)
      }

      constraintTripletClassifiers.foreach {
        x =>
          val res = x.test()
          ReportHelper.saveEvalResults(outStream, s"${x.getClassSimpleNameForClassifier}(within data model)", res)
      }

      if (usePrepositions && visualTripletsFiltered.nonEmpty) {
        val prepResult = VisualTripletClassifier.test(visualTripletsFiltered)
        ReportHelper.saveEvalResults(outStream, s"Preposition(within data model)", prepResult)
      }

      reportForErrorAnalysis(x => TripletRelationConstraintClassifier(x),
        x => TRConstraintClassifier(x),
        x => LMConstraintClassifier(x),
        x => IndicatorRoleClassifier(x),
        x => TripletGeneralTypeConstraintClassifier(x),
        x => TripletDirectionConstraintClassifier(x),
        x => TripletRegionConstraintClassifier(x)
      )
    }

  }

  def reportForErrorAnalysis(rel: Relation => String, tr: Phrase => String, lm: Phrase => String, sp: Phrase => String,
                             general: Relation => String, direction: Relation => String, region: Relation => String) = {
    val writer = new PrintStream(s"$resultsDir/error_report_$expName$suffix.txt")

    triplets().toList.sortBy(x => x.getId).foreach { r =>
      val predicted = rel(r)
      val actual = tripletIsRelation(r)
      val gen = tripletGeneralType(r)
      val reg = tripletRegion(r)
      val dir = tripletDirection(r)
      val predictedGen = general(r)
      val predictedReg = region(r)
      val predictedDir = direction(r)
      val t = triplets(r) ~> tripletToTr head
      val s = triplets(r) ~> tripletToSp head
      val l = triplets(r) ~> tripletToLm head
      val tDis = tripletDistanceTrSp(r)
      val lDis = tripletDistanceLmSp(r)

      val tSeg = matchingSegment(t)
      val lSeg = matchingSegment(l)
      val tSegSim = similarityToMatchingSegment(t)
      val lSegSim = similarityToMatchingSegment(l)
      val matchingImageSp = if (usePrepositions) tripletMatchingSegmentRelationLabel(r) else "-"
      val matchingImageSpScores = if (usePrepositions) tripletMatchingSegmentRelationLabelScores(r) else "-"

      val tCorrect = "trajector".equalsIgnoreCase(tr(t))
      val sCorrect = "indicator".equalsIgnoreCase(sp(s))
      val lCorrect = if (l == dummyPhrase) false else "landmark".equalsIgnoreCase(lm(l))

      val sent = triplets(r) ~> -sentenceToTriplets head

      val segments = (sentences(sent) ~> -documentToSentence ~> documentToImage ~> imageToSegment)
        .toList.sortBy(_.getSegmentId)
        .map(x => x.getSegmentId + ":" + x.getSegmentCode + ":" + x.getSegmentConcept).mkString(",")

      val matchings = (sentences(sent) ~> sentenceToPhrase).toList.sortBy(_.getStart)
        .map(p => p.getText + "-> " + matchingSegment(p) + ": " + similarityToMatchingSegment(p))
        .mkString(",")

      val imageRels = ""

      val docId = (triplets(r) ~> -sentenceToTriplets ~> -documentToSentence).head.getId

      //docId, sentId, sent, actual rel, predicted rel, tr text, sp text, lm text
      //tr correct, sp correct, lm correct, segments[id, code, text], ...
      val line = s"$docId\t\t${sent.getId}\t\t${sent.getText}\t\t${actual}" +
        s"\t\t${predicted}\t\t${t.getText}\t\t${s.getText}\t\t${l.getText}\t\t${tCorrect}" +
        s"\t\t${sCorrect}\t\t${lCorrect}\t\t${segments}\t\t$tDis\t\t$lDis\t\t$tSeg\t\t$tSegSim\t\t$lSeg\t\t$lSegSim" +
        s"\t\t$matchings\t\t$imageRels\t\t$matchingImageSp\t\t$matchingImageSpScores\t\t$gen\t\t$reg\t\t$dir" +
        s"\t\t$predictedGen\t\t$predictedReg\t\t$predictedDir"
      writer.println(line)
    }
  }

}
