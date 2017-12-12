package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.io.{File, FileOutputStream, PrintStream}

import edu.illinois.cs.cogcomp.saul.classifier.{ConstrainedClassifier, JointTrainSparsePerceptron}
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation}
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{CandidateGenerator, FeatureSets, ReportHelper}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalPopulateData._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.TripletSentenceLevelConstraintClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import org.apache.commons.io.FileUtils

import scala.util.Random

object MultiModalTripletApp extends App with Logging {

  val expName = "triplet_" + ((model, useConstraints) match {
    case (FeatureSets.BaseLine, false) => "BM"
    case (FeatureSets.BaseLineWithImage, false) => "BM+I"
    case (FeatureSets.BaseLineWithImage, true) => "BM+C+I"
    case (FeatureSets.BaseLine, true) => "BM+C"
    case (FeatureSets.WordEmbedding, false) => "BM+E"
    case (FeatureSets.WordEmbedding, true) => "BM+C+E"
    case (FeatureSets.WordEmbeddingPlusImage, false) => "BM+E+I"
    case (FeatureSets.WordEmbeddingPlusImage, true) => "BM+C+E+I"
    case _ =>
      logger.error("experiment is not supported")
      System.exit(1)
  })
  MultiModalSpRLTripletClassifiers.featureSet = model

  val constrainedPrepClassifiers = List(
    PrepositionAboveConstraintClassifier,
    PrepositionInFrontOfConstraintClassifier,
    PrepositionOnConstraintClassifier,
    PrepositionInConstraintClassifier
  )
  val prepClassifiers = Map(
    //PrepositionClassifier,
    "above" -> PrepositionAboveClassifier,
    "in_front_of" -> PrepositionInFrontOfClassifier,
    "on" -> PrepositionOnClassifier,
    "in" -> PrepositionInClassifier
    //"behind" -> PrepositionBehindClassifier,
    //"around" -> PrepositionAroundClassifier,
    //"at" -> PrepositionAtClassifier,
    //"between" -> PrepositionBetweenClassifier,
    //"in between" -> PrepositionInBetweenClassifier,
    //"in the middile of" -> PrepositionInTheMiddleOfClassifier,
    //"leaning on" -> PrepositionLeaningOnClassifier,
    //"near" -> PrepositionNearClassifier,
    //"next to" -> PrepositionNextToClassifier,
    //"on each side" -> PrepositionOnEachSideClassifier,
    //"over " -> PrepositionOverClassifier,
    //"sitting around" -> PrepositionSittingAroundClassifier,
    //"with" -> PrepositionWithClassifier,
  )

  val constraintRoleClassifiers = List(
    TRConstraintClassifier,
    LMConstraintClassifier
  )

  val roleClassifiers = List(
    TrajectorRoleClassifier,
    LandmarkRoleClassifier,
    IndicatorRoleClassifier
  )

  val constraintTripletClassifiers = List(
    TripletRelationConstraintClassifier,
    TripletRegionTPPConstraintClassifier,
    TripletRegionECConstraintClassifier,
    TripletRegionEQConstraintClassifier,
    TripletRegionDCConstraintClassifier,
    TripletRegionPOConstraintClassifier,
    TripletDirectionRightConstraintClassifier,
    TripletDirectionLeftConstraintClassifier,
    TripletDirectionAboveConstraintClassifier,
    TripletDirectionBelowConstraintClassifier,
    TripletDirectionBehindConstraintClassifier,
    TripletDirectionFrontConstraintClassifier
  )
  val tripletClassifiers = List(
    TripletRelationClassifier,
    TripletGeneralTypeClassifier,
    TripletGeneralRegionClassifier,
    TripletGeneralDirectionClassifier,
    TripletRegionClassifier,
    TripletDirectionClassifier,
    TripletRegionTPPClassifier,
    TripletRegionEQClassifier,
    TripletRegionDCClassifier,
    TripletRegionECClassifier,
    TripletRegionPOClassifier,
    TripletDirectionAboveClassifier,
    TripletDirectionBehindClassifier,
    TripletDirectionBelowClassifier,
    TripletDirectionFrontClassifier,
    TripletDirectionLeftClassifier,
    TripletDirectionRightClassifier
  )

  val classifiers = prepClassifiers.values ++ roleClassifiers ++ tripletClassifiers
  classifiers.foreach(x => {
    x.modelDir = s"models/mSpRL/triplet/$featureSet/"
    x.modelSuffix = suffix
  })
  FileUtils.forceMkdir(new File(resultsDir))

  if (isTrain && trainPrepositionClassifier) {
    populateVisualTripletsFromExternalData()
    prepClassifiers.foreach {
      x =>
        val positive = visualTriplets().filter(y => x._1.equalsIgnoreCase(y.getSp)).toList
        val negative = visualTriplets().filter(y => y.getSp != null && !x._1.equalsIgnoreCase(y.getSp)).toList
        val examples = Random.shuffle(Random.shuffle(negative).take(positive.size * 2) ++ positive)
        if (x._2 != PrepositionInFrontOfClassifier && x._2 != PrepositionAboveClassifier)
          x._2.learn(iterations, examples)
    }
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

    val spCandidatesTrain = CandidateGenerator.getIndicatorCandidates(phrases().toList)
    val trCandidatesTrain = CandidateGenerator.getTrajectorCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))
    val lmCandidatesTrain = CandidateGenerator.getLandmarkCandidates(phrases().toList)
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

    if (trainPrepositionClassifier) {

      val visualTripletsFiltered = visualTriplets().toList.filter(x => x.getSp != null)

      prepClassifiers.foreach {
        x =>
          val positive = visualTripletsFiltered.filter(y => x._1.equalsIgnoreCase(y.getSp))
          val negative = visualTripletsFiltered.filter(y => !x._1.equalsIgnoreCase(y.getSp))
          val examples = Random.shuffle(Random.shuffle(negative).take(positive.size * 2) ++ positive)

          if (x._2 == PrepositionInFrontOfClassifier || x._2 == PrepositionAboveClassifier) {
            x._2.learn(iterations, examples)
          }
          else {
            x._2.learn(10, examples)
          }
          x._2.test(examples)
          x._2.save()
      }

    }
    if (jointTrain) {
      logger.info("====================================")
      logger.info("Joint Train started")
      JointTrainSparsePerceptron.train(MultiModalSpRLDataModel.sentences,
        constrainedPrepClassifiers ++ List(TripletRelationConstraintClassifier), 10)
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

    if (!trainTestTogether)
      classifiers.foreach(x => x.load())

    val spCandidatesTest = CandidateGenerator.getIndicatorCandidates(phrases.getTestingInstances.toList)
    val trCandidatesTest = CandidateGenerator.getTrajectorCandidates(phrases.getTestingInstances.toList)
      .filterNot(x => spCandidatesTest.contains(x))
    val lmCandidatesTest = CandidateGenerator.getLandmarkCandidates(phrases.getTestingInstances.toList)
      .filterNot(x => spCandidatesTest.contains(x))

    populateTripletDataFromAnnotatedCorpus(
      x => trCandidatesTest.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "true",
      x => lmCandidatesTest.exists(_.getId == x.getId))

    if (!useConstraints) {
      val visualTripletsFiltered = visualTriplets.getTestingInstances.toList.filter(x => x.getSp != null)
      val trajectors = phrases.getTestingInstances.filter(x => TrajectorRoleClassifier(x) == "true").toList
      val landmarks = phrases.getTestingInstances.filter(x => LandmarkRoleClassifier(x) == "true").toList
      val indicators = phrases.getTestingInstances.filter(x => IndicatorRoleClassifier(x) == "true").toList

      val tripletList = triplets.getTestingInstances
        .filter(x => TripletRelationClassifier(x) == "true").toList

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
      prepClassifiers.foreach {
        x =>
          val res = x._2.test(visualTripletsFiltered)
          ReportHelper.saveEvalResults(outStream, s"${x._1}(within data model)", res)
      }
    }
    else {

      val visualTripletsFiltered = visualTriplets.getTestingInstances.toList.filter(x => x.getSp != null)
      val trajectors = phrases.getTestingInstances.filter(x => TRConstraintClassifier(x) == "true").toList
      val landmarks = phrases.getTestingInstances.filter(x => LMConstraintClassifier(x) == "true").toList
      val indicators = phrases.getTestingInstances.filter(x => IndicatorConstraintClassifier(x) == "true").toList

      val tripletList = triplets.getTestingInstances
        .filter(x => TripletRelationConstraintClassifier(x) == "true").toList


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
      constrainedPrepClassifiers.foreach {
        x =>
          val res = x.test(visualTripletsFiltered)
          ReportHelper.saveEvalResults(outStream, s"${x.getClassSimpleNameForClassifier}(within data model)", res)
      }

      //      report(x => TripletRelationConstraintClassifier(x),
      //        x => TRConstraintClassifier(x),
      //        x => LMConstraintClassifier(x),
      //        x => IndicatorRoleClassifier(x),
      //        x => TripletGeneralTypeConstraintClassifier(x),
      //        x => TripletDirectionConstraintClassifier(x),
      //        x => TripletRegionConstraintClassifier(x)
      //      )
    }

  }

  def report(rel: Relation => String, tr: Phrase => String, lm: Phrase => String, sp: Phrase => String,
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
      val matchingImageSp = tripletMatchingSegmentRelationLabel(r)
      val matchingImageSpScores = tripletMatchingSegmentRelationLabelScores(r)

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

  //  val all = triplets().map(x => (x.getArgument(1).getText, tripletMatchingSegmentRelationLabelScores(x), tripletIsRelation(x), x))
  //    .filter(x => !x._2.equals("-")).toList
  //  val rels = all.filter(_._3.equalsIgnoreCase("Relation"))
  //  val noRels = all.filter(_._3.equalsIgnoreCase("None"))
  //  val tp = rels.count(x => x._2.split(":").head == x._1)
  //  val fn = rels.size - tp
  //  val fp = noRels.count(x => x._2.split(":").head == x._1)
  //  val tn = noRels.size - fp
  //  val writer = new PrintStream(s"$resultsDir/preposition-prediction_${isTrain}.txt")
  //  writer.println("Aligned ground truth: " + rels.size)
  //  writer.println("tp: " + tp)
  //  writer.println("tn: " + tn)
  //  writer.println("fp: " + fp)
  //  writer.println("fn: " + fn)
  //  all.sortBy(_._3).foreach(x => writer.println(x._3 + "[" + x._4.getProperty("ActualId") + "](" + x._4.getArgument(0).getText + ", " + x._1 + ", " +
  //    x._4.getArgument(2).getText + ") :: " + x._2 + " :: "))
  //  writer.close()

}
