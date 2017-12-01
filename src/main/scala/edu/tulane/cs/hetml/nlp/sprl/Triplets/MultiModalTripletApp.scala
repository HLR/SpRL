package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.io.{File, FileOutputStream, PrintStream}

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation}
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{CandidateGenerator, FeatureSets, ReportHelper}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalPopulateData._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.TripletSentenceLevelConstraintClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletClassifiers.VisualTripletClassifier
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletsDataModel
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import edu.tulane.cs.hetml.vision.CLEFAlignmentReader
import org.apache.commons.io.FileUtils

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

  val classifiers = List(
    TrajectorRoleClassifier,
    LandmarkRoleClassifier,
    IndicatorRoleClassifier,
    TripletRelationClassifier,
    TripletGeneralTypeClassifier,
    TripletSpecificTypeClassifier,
    TripletRegionClassifier,
    TripletDirectionClassifier,
    TripletImageRegionClassifier
  )
  classifiers.foreach(x => {
    x.modelDir = s"models/mSpRL/triplet/$featureSet/"
    x.modelSuffix = suffix
  })
  FileUtils.forceMkdir(new File(resultsDir))

  populateRoleDataFromAnnotatedCorpus()
  val visualClassifier = new VisualTripletClassifier()

  if (isTrain) {
    println("training started ...")

    TrajectorRoleClassifier.learn(iterations)
    TrajectorRoleClassifier.test(phrases())

    IndicatorRoleClassifier.learn(iterations)
    IndicatorRoleClassifier.test(phrases())

    LandmarkRoleClassifier.learn(iterations)
    LandmarkRoleClassifier.test(phrases())

    TrajectorRoleClassifier.save()
    IndicatorRoleClassifier.save()
    LandmarkRoleClassifier.save()

    val spCandidatesTrain = CandidateGenerator.getIndicatorCandidates(phrases().toList)
    val trCandidatesTrain = CandidateGenerator.getTrajectorCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))
    val lmCandidatesTrain = CandidateGenerator.getLandmarkCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))


    populateTripletDataFromAnnotatedCorpus(
      x => trCandidatesTrain.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "Indicator",
      x => lmCandidatesTrain.exists(_.getId == x.getId)
    )

//    TripletRelationClassifier.learn(iterations)
//    TripletRelationClassifier.test(triplets())
//
//    TripletGeneralTypeClassifier.learn(iterations)
//    TripletGeneralTypeClassifier.test(triplets())
//
//    TripletSpecificTypeClassifier.learn(iterations)
//    //TripletSpecificTypeClassifier.test(triplets())
//
//    TripletRegionClassifier.learn(iterations)
//    TripletRegionClassifier.test(triplets())
//
//    TripletDirectionClassifier.learn(iterations)
//    TripletDirectionClassifier.test(triplets())
//
//    TripletImageRegionClassifier.learn(iterations)
//    TripletImageRegionClassifier.test(triplets())
//
//
//    TripletRelationClassifier.save()
//    TripletGeneralTypeClassifier.save()
//    TripletSpecificTypeClassifier.save()
//    TripletRegionClassifier.save()
//    TripletDirectionClassifier.save()
//    TripletImageRegionClassifier.save()

    if (fineTunePrepositionClassifier) {
      val classifierDirectory = s"models/mSpRL/VisualTriplets/"
      val classifierSuffix = "combined_perceptron"
      visualClassifier.modelSuffix = classifierSuffix
      visualClassifier.modelDir = classifierDirectory
      //visualClassifier.load()
      val visualTriplets = (triplets() ~> tripletToVisualTriplet).toList.filter(x => x.getSp != "-")
      //visualClassifier.test(visualTriplets)
      visualClassifier.learn(50)
      visualClassifier.test(visualTriplets)
      visualClassifier.modelSuffix += "_tuned"
      visualClassifier.save()
    }
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
    TripletImageRegionClassifier.load()

    val classifierDirectory = s"models/mSpRL/VisualTriplets/"
    val classifierSuffix = "combined_perceptron_tuned"
    visualClassifier.modelSuffix = classifierSuffix
    visualClassifier.modelDir = classifierDirectory
    visualClassifier.load()

    val spCandidatesTest = CandidateGenerator.getIndicatorCandidates(phrases().toList)
    val trCandidatesTest = CandidateGenerator.getTrajectorCandidates(phrases().toList)
      .filterNot(x => spCandidatesTest.contains(x))
    val lmCandidatesTest = CandidateGenerator.getLandmarkCandidates(phrases().toList)
      .filterNot(x => spCandidatesTest.contains(x))

    populateTripletDataFromAnnotatedCorpus(
      x => trCandidatesTest.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "Indicator",
      x => lmCandidatesTest.exists(_.getId == x.getId))

    if (!useConstraints) {
      val trajectors = phrases.getTestingInstances.filter(x => TrajectorRoleClassifier(x) == "Trajector").toList
      val landmarks = phrases.getTestingInstances.filter(x => LandmarkRoleClassifier(x) == "Landmark").toList
      val indicators = phrases.getTestingInstances.filter(x => IndicatorRoleClassifier(x) == "Indicator").toList

      val tripletList = triplets.getTestingInstances
        .filter(x => TripletRelationClassifier(x) == "Relation").toList


      ReportHelper.saveAsXml(tripletList, trajectors, indicators, landmarks,
        x => TripletGeneralTypeClassifier(x),
        x => TripletSpecificTypeClassifier(x),
        x => TripletRegionClassifier(x),
        x => TripletDirectionClassifier(x),
        s"$resultsDir/${expName}${suffix}.xml")

      //ReportHelper.saveEvalResultsFromXmlFile(testFile, s"$resultsDir/${expName}${suffix}.xml", s"$resultsDir/$expName$suffix.txt")

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

      val imRegion = TripletImageRegionClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Image Region(within data model)", imRegion)

      val visual = visualClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Visual triplet(within data model)", visual)

      report(x => TripletRelationClassifier(x),
        x => TrajectorRoleClassifier(x),
        x => LandmarkRoleClassifier(x),
        x => IndicatorRoleClassifier(x),
        x => TripletGeneralTypeClassifier(x),
        x => TripletDirectionClassifier(x),
        x => TripletRegionClassifier(x)
      )
    }
    else {

      val trajectors = phrases.getTestingInstances.filter(x => TRConstraintClassifier(x) == "Trajector").toList
      val landmarks = phrases.getTestingInstances.filter(x => LMConstraintClassifier(x) == "Landmark").toList
      val indicators = phrases.getTestingInstances.filter(x => IndicatorConstraintClassifier(x) == "Indicator").toList

      val tripletList = triplets.getTestingInstances
        .filter(x => TripletRelationConstraintClassifier(x) == "Relation").toList


      ReportHelper.saveAsXml(tripletList, trajectors, indicators, landmarks,
        x => TripletGeneralTypeConstraintClassifier(x),
        x => TripletSpecificTypeClassifier(x),
        x => TripletRegionConstraintClassifier(x),
        x => TripletDirectionConstraintClassifier(x),
        s"$resultsDir/${expName}${suffix}.xml")

      //ReportHelper.saveEvalResultsFromXmlFile(testFile, s"$resultsDir/${expName}${suffix}.xml", s"$resultsDir/$expName$suffix.txt")

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

      val visual = visualClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Visual triplet(within data model)", visual)

      report(x => TripletRelationConstraintClassifier(x),
        x => TRConstraintClassifier(x),
        x => LMConstraintClassifier(x),
        x => IndicatorRoleClassifier(x),
        x => TripletGeneralTypeConstraintClassifier(x),
        x => TripletDirectionConstraintClassifier(x),
        x => TripletRegionConstraintClassifier(x)
      )
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
      val t = triplets(r) ~> tripletToFirstArg head
      val s = triplets(r) ~> tripletToSecondArg head
      val l = triplets(r) ~> tripletToThirdArg head
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

      val imageRels = tripletMatchingSegmentRelations(r).mkString(",")

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

  val all = triplets().map(x => (x.getArgument(1).getText, tripletMatchingSegmentRelationLabelScores(x), tripletIsRelation(x), x))
    .filter(x => !x._2.equals("-")).toList
  val rels = all.filter(_._3.equalsIgnoreCase("Relation"))
  val noRels = all.filter(_._3.equalsIgnoreCase("None"))
  val tp = rels.count(x => x._2.split(":").head == x._1)
  val fn = rels.size - tp
  val fp = noRels.count(x => x._2.split(":").head == x._1)
  val tn = noRels.size - fp
  val writer = new PrintStream(s"$resultsDir/preposition-prediction_${isTrain}.txt")
  writer.println("Aligned ground truth: " + rels.size)
  writer.println("tp: " + tp)
  writer.println("tn: " + tn)
  writer.println("fp: " + fp)
  writer.println("fn: " + fn)
  all.sortBy(_._3).foreach(x => writer.println(x._3 + "[" + x._4.getProperty("ActualId") + "](" + x._4.getArgument(0).getText + ", " + x._1 + ", " +
    x._4.getArgument(2).getText + ") :: " + x._2 + " :: "))
  writer.close()
}

