package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.io.{File, FileOutputStream, PrintStream}

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation}
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{CandidateGenerator, FeatureSets, ReportHelper}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalPopulateData._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.TripletSentenceLevelConstraintClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import org.apache.commons.io.FileUtils

object MultiModalTripletApp extends App with Logging {

  val expName = "triplet_" + ((model, useConstraints) match {
    case (FeatureSets.BaseLine, false) => "BM"
    case (FeatureSets.BaseLineWithImage, false) => "BM+I"
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

  val classifiers = List(
    TrajectorRoleClassifier,
    LandmarkRoleClassifier,
    IndicatorRoleClassifier,
    TripletRelationClassifier,
    TripletGeneralTypeClassifier,
    TripletSpecificTypeClassifier,
    TripletRegionClassifier,
    TripletDirectionClassifier
  )
  classifiers.foreach(x => {
    x.modelDir = s"models/mSpRL/triplet/$featureSet/"
    x.modelSuffix = suffix
  })
  FileUtils.forceMkdir(new File(resultsDir))

  populateRoleDataFromAnnotatedCorpus()

  if (isTrain) {
    println("training started ...")

    TrajectorRoleClassifier.learn(iterations)
    IndicatorRoleClassifier.learn(iterations)
    LandmarkRoleClassifier.learn(iterations)

    TrajectorRoleClassifier.save()
    IndicatorRoleClassifier.save()
    LandmarkRoleClassifier.save()

    val trCandidatesTrain = CandidateGenerator.getTrajectorCandidates(phrases().toList)
    val lmCandidatesTrain = CandidateGenerator.getLandmarkCandidates(phrases().toList)

    populateTripletDataFromAnnotatedCorpus(
      x => trCandidatesTrain.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "Indicator",
      x => lmCandidatesTrain.exists(_.getId == x.getId)
    )

    TripletRelationClassifier.learn(iterations)
    TripletGeneralTypeClassifier.learn(iterations)
    TripletSpecificTypeClassifier.learn(iterations)
    TripletRegionClassifier.learn(iterations)
    TripletDirectionClassifier.learn(iterations)

    TripletRelationClassifier.save()
    TripletGeneralTypeClassifier.save()
    TripletSpecificTypeClassifier.save()
    TripletRegionClassifier.save()
    TripletDirectionClassifier.save()

  }

  if (!isTrain) {

    println("testing started ...")

    TrajectorRoleClassifier.load()
    LandmarkRoleClassifier.load()
    IndicatorRoleClassifier.load()
    TripletRelationClassifier.load()
    TripletGeneralTypeClassifier.load()
    TripletSpecificTypeClassifier.load()
    TripletRegionClassifier.load()
    TripletDirectionClassifier.load()

    val trCandidatesTest = CandidateGenerator.getTrajectorCandidates(phrases().toList)
    val lmCandidatesTest = CandidateGenerator.getLandmarkCandidates(phrases().toList)

    populateTripletDataFromAnnotatedCorpus(
      x => trCandidatesTest.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "Indicator",
      x => lmCandidatesTest.exists(_.getId == x.getId))


    if (!useConstraints) {
      //      val trajectors = phrases.getTestingInstances.filter(x => TrajectorRoleClassifier(x) == "Trajector").toList
      //      val landmarks = phrases.getTestingInstances.filter(x => LandmarkRoleClassifier(x) == "Landmark").toList
      //      val indicators = phrases.getTestingInstances.filter(x => IndicatorRoleClassifier(x) == "Indicator").toList
      //
      //      val tripletList = triplets.getTestingInstances
      //        .filter(x => TripletRelationClassifier(x) == "Relation").toList
      //
      //
      //      ReportHelper.saveAsXml(tripletList, trajectors, indicators, landmarks,
      //        x => TripletGeneralTypeClassifier(x),
      //        x => TripletSpecificTypeClassifier(x),
      //        x => TripletRCC8Classifier(x),
      //        x => TripletDirectionClassifier(x),
      //        s"$resultsDir/${expName}${suffix}.xml")
      //
      //      ReportHelper.saveEvalResultsFromXmlFile(testFile, s"$resultsDir/${expName}${suffix}.xml", s"$resultsDir/$expName$suffix.txt")

      val outStream = new FileOutputStream(s"$resultsDir/$expName$suffix.txt", false)

      val tr = TrajectorRoleClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Trajector(within data model)", tr)

      val sp = IndicatorRoleClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Spatial Indicator(within data model)", sp)

      val lm = LandmarkRoleClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Landmark(within data model)", lm)

      val relation = TripletRelationClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Relation(within data model)", relation)

      val general = TripletGeneralTypeClassifier.test()
      ReportHelper.saveEvalResults(outStream, "General(within data model)", general)

      val direction = TripletDirectionClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Direction(within data model)", direction)

      val region = TripletRegionClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Region(within data model)", region)

      report(x => TripletRelationClassifier(x),
        x => TrajectorRoleClassifier(x),
        x => LandmarkRoleClassifier(x),
        x => IndicatorRoleClassifier(x)
      )
    }
    else {

      //      val trajectors = phrases.getTestingInstances.filter(x => TRConstraintClassifier(x) == "Trajector").toList
      //      val landmarks = phrases.getTestingInstances.filter(x => LMConstraintClassifier(x) == "Landmark").toList
      //      val indicators = phrases.getTestingInstances.filter(x => IndicatorConstraintClassifier(x) == "Indicator").toList
      //
      //      val tripletList = triplets.getTestingInstances
      //        .filter(x => TripletRelationConstraintClassifier(x) == "Relation").toList
      //
      //
      //      ReportHelper.saveAsXml(tripletList, trajectors, indicators, landmarks,
      //        x => TripletGeneralTypeConstraintClassifier(x),
      //        x => TripletSpecificTypeClassifier(x),
      //        x => TripletRegionConstraintClassifier(x),
      //        x => TripletDirectionConstraintClassifier(x),
      //        s"$resultsDir/${expName}${suffix}.xml")
      //
      //      ReportHelper.saveEvalResultsFromXmlFile(testFile, s"$resultsDir/${expName}${suffix}.xml", s"$resultsDir/$expName$suffix.txt")

      val outStream = new FileOutputStream(s"$resultsDir/$expName$suffix.txt", false)

      val tr = TRConstraintClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Trajector(within data model)", tr)

      val sp = IndicatorRoleClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Spatial Indicator(within data model)", sp)

      val lm = LMConstraintClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Landmark(within data model)", lm)

      val relation = TripletRelationConstraintClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Relation(within data model)", relation)

      val general = TripletGeneralTypeConstraintClassifier.test()
      ReportHelper.saveEvalResults(outStream, "General(within data model)", general)

      val direction = TripletDirectionConstraintClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Direction(within data model)", direction)

      val region = TripletRegionConstraintClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Region(within data model)", region)

      report(x => TripletRelationConstraintClassifier(x),
        x => TRConstraintClassifier(x),
        x => LMConstraintClassifier(x),
        x => IndicatorRoleClassifier(x)
      )
    }

  }

  def report(rel: Relation => String, tr: Phrase => String, lm: Phrase => String, sp: Phrase => String) = {
    val writer = new PrintStream(s"$resultsDir/error_report_$expName$suffix.txt")

    triplets().foreach { r =>
      val predicted = rel(r)
      val actual = tripletIsRelation(r)
      val t = triplets(r) ~> tripletToFirstArg head
      val s = triplets(r) ~> tripletToSecondArg head
      val l = triplets(r) ~> tripletToThirdArg head

      val tCorrect = "trajector".equalsIgnoreCase(tr(t))
      val sCorrect = "indicator".equalsIgnoreCase(sp(s))
      val lCorrect = if(l == dummyPhrase) false else "landmark".equalsIgnoreCase(lm(l))

      val sent = triplets(r) ~> -sentenceToTriplets head
      val segments = (sentences(sent) ~> -documentToSentence ~> documentToImage ~> imageToSegment)
        .map(x => x.getSegmentId + ":" + x.getSegmentCode +":" + x.getSegmentConcept).mkString(",")
      val docId = (triplets(r) ~> -sentenceToTriplets ~> -documentToSentence).head.getId
      //docId, sentId, sent, actual rel, predicted rel, tr text, sp text, lm text
      //tr correct, sp correct, lm correct, segments[id, code, text]
      val line = s"$docId\t\t${sent.getId}\t\t${sent.getText}\t\t${actual}" +
        s"\t\t${predicted}\t\t${t.getText}\t\t${s.getText}\t\t${l.getText}\t\t${tCorrect}" +
        s"\t\t${sCorrect}\t\t${lCorrect}\t\t${segments}"
      writer.println(line)
    }
  }

}

