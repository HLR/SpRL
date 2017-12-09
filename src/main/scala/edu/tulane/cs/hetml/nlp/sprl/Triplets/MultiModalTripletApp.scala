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
    TripletGeneralRegionClassifier,
    TripletGeneralDirectionClassifier,
    TripletSpecificTypeClassifier,
    TripletRegionClassifier,
    TripletDirectionClassifier,
    PrepositionClassifier,
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
    TripletDirectionRightClassifier,
    PrepositionAroundClassifier,
    PrepositionAtClassifier,
    PrepositionBehindClassifier,
    PrepositionBetweenClassifier,
    PrepositionInBetweenClassifier,
    PrepositionInClassifier,
    PrepositionInTheMiddleOfClassifier,
    PrepositionLeaningOnClassifier,
    PrepositionNearClassifier,
    PrepositionNextToClassifier,
    PrepositionOnClassifier,
    PrepositionOnEachSideClassifier,
    PrepositionOverClassifier,
    PrepositionSittingAroundClassifier,
    PrepositionWithClassifier
  )
  classifiers.foreach(x => {
    x.modelDir = s"models/mSpRL/triplet/$featureSet/"
    x.modelSuffix = suffix
  })
  FileUtils.forceMkdir(new File(resultsDir))

  if (isTrain && trainPrepositionClassifier) {
    populateVisualTripletsFromExternalData()
    PrepositionClassifier.learn(iterations)
//    PrepositionAroundClassifier.learn(iterations)
//    PrepositionAtClassifier.learn(iterations)
//    PrepositionBehindClassifier.learn(iterations)
//    PrepositionBetweenClassifier.learn(iterations)
//    PrepositionInBetweenClassifier.learn(iterations)
//    PrepositionInClassifier.learn(iterations)
//    PrepositionInTheMiddleOfClassifier.learn(iterations)
//    PrepositionLeaningOnClassifier.learn(iterations)
//    PrepositionNearClassifier.learn(iterations)
//    PrepositionNextToClassifier.learn(iterations)
//    PrepositionOnClassifier.learn(iterations)
//    PrepositionOnEachSideClassifier.learn(iterations)
//    PrepositionOverClassifier.learn(iterations)
//    PrepositionSittingAroundClassifier.learn(iterations)
//    PrepositionWithClassifier.learn(iterations)
    visualTriplets.clear()
  }

  populateRoleDataFromAnnotatedCorpus()

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
      x => IndicatorRoleClassifier(x) == "true",
      x => lmCandidatesTrain.exists(_.getId == x.getId)
    )

    TripletRelationClassifier.learn(iterations)
    TripletRelationClassifier.test(triplets())

    TripletGeneralTypeClassifier.learn(iterations)
    TripletGeneralTypeClassifier.test(triplets())

//    TripletGeneralDirectionClassifier.learn(iterations)
//    TripletGeneralDirectionClassifier.test(triplets())
//
//    TripletGeneralRegionClassifier.learn(iterations)
//    TripletGeneralRegionClassifier.test(triplets())

    TripletSpecificTypeClassifier.learn(iterations)
    //TripletSpecificTypeClassifier.test(triplets())

    TripletRegionClassifier.learn(iterations)
    TripletRegionClassifier.test(triplets())

//    TripletRegionTPPClassifier.learn(iterations)
//    TripletRegionEQClassifier.learn(iterations)
//    TripletRegionDCClassifier.learn(iterations)
//    TripletRegionECClassifier.learn(iterations)
//    TripletRegionPOClassifier.learn(iterations)

    TripletDirectionClassifier.learn(iterations)
    TripletDirectionClassifier.test(triplets())

//    TripletDirectionAboveClassifier.learn(iterations)
//    TripletDirectionBehindClassifier.learn(iterations)
//    TripletDirectionBelowClassifier.learn(iterations)
//    TripletDirectionFrontClassifier.learn(iterations)
//    TripletDirectionLeftClassifier.learn(iterations)
//    TripletDirectionRightClassifier.learn(iterations)

    if (trainPrepositionClassifier) {

      val visualTripletsFiltered = visualTriplets().toList.filter(x => x.getSp != null)

      //fine tune with clef examples
      PrepositionClassifier.learn(10, visualTripletsFiltered)
//      PrepositionAroundClassifier.learn(10, visualTripletsFiltered)
//      PrepositionAtClassifier.learn(10, visualTripletsFiltered)
//      PrepositionBehindClassifier.learn(10, visualTripletsFiltered)
//      PrepositionBetweenClassifier.learn(10, visualTripletsFiltered)
//      PrepositionInBetweenClassifier.learn(10, visualTripletsFiltered)
//      PrepositionInClassifier.learn(10, visualTripletsFiltered)
//      PrepositionInTheMiddleOfClassifier.learn(10, visualTripletsFiltered)
//      PrepositionLeaningOnClassifier.learn(10, visualTripletsFiltered)
//      PrepositionNearClassifier.learn(10, visualTripletsFiltered)
//      PrepositionNextToClassifier.learn(10, visualTripletsFiltered)
//      PrepositionOnClassifier.learn(10, visualTripletsFiltered)
//      PrepositionOnEachSideClassifier.learn(10, visualTripletsFiltered)
//      PrepositionOverClassifier.learn(10, visualTripletsFiltered)
//      PrepositionSittingAroundClassifier.learn(10, visualTripletsFiltered)
//      PrepositionWithClassifier.learn(10, visualTripletsFiltered)
//      // Train on clef only
//      PrepositionInFrontOfClassifier.learn(iterations, visualTripletsFiltered)
//      PrepositionAboveClassifier.learn(iterations, visualTripletsFiltered)

      PrepositionClassifier.test(visualTripletsFiltered)

//      PrepositionAboveClassifier.test(visualTripletsFiltered)
//      PrepositionAroundClassifier.test(visualTripletsFiltered)
//      PrepositionAtClassifier.test(visualTripletsFiltered)
//      PrepositionBehindClassifier.test(visualTripletsFiltered)
//      PrepositionBetweenClassifier.test(visualTripletsFiltered)
//      PrepositionInBetweenClassifier.test(visualTripletsFiltered)
//      PrepositionInClassifier.test(visualTripletsFiltered)
//      PrepositionInFrontOfClassifier.test(visualTripletsFiltered)
//      PrepositionInTheMiddleOfClassifier.test(visualTripletsFiltered)
//      PrepositionLeaningOnClassifier.test(visualTripletsFiltered)
//      PrepositionNearClassifier.test(visualTripletsFiltered)
//      PrepositionNextToClassifier.test(visualTripletsFiltered)
//      PrepositionOnClassifier.test(visualTripletsFiltered)
//      PrepositionOnEachSideClassifier.test(visualTripletsFiltered)
//      PrepositionOverClassifier.test(visualTripletsFiltered)
//      PrepositionSittingAroundClassifier.test(visualTripletsFiltered)
//      PrepositionWithClassifier.test(visualTripletsFiltered)
//
      PrepositionClassifier.save()
      PrepositionAboveClassifier.save()
      PrepositionAroundClassifier.save()
      PrepositionAtClassifier.save()
      PrepositionBehindClassifier.save()
      PrepositionBetweenClassifier.save()
      PrepositionInBetweenClassifier.save()
      PrepositionInClassifier.save()
      PrepositionInFrontOfClassifier.save()
      PrepositionInTheMiddleOfClassifier.save()
      PrepositionLeaningOnClassifier.save()
      PrepositionNearClassifier.save()
      PrepositionNextToClassifier.save()
      PrepositionOnClassifier.save()
      PrepositionOnEachSideClassifier.save()
      PrepositionOverClassifier.save()
      PrepositionSittingAroundClassifier.save()
      PrepositionWithClassifier.save()

    }

    TripletRelationClassifier.save()
    TripletGeneralTypeClassifier.save()
    TripletGeneralDirectionClassifier.save()
    TripletGeneralRegionClassifier.save()
    TripletSpecificTypeClassifier.save()
    TripletRegionClassifier.save()
    TripletDirectionClassifier.save()
    TripletImageRegionClassifier.save()

//    TripletRegionTPPClassifier.save()
//    TripletRegionEQClassifier.save()
//    TripletRegionDCClassifier.save()
//    TripletRegionECClassifier.save()
//    TripletRegionPOClassifier.save()
//
//    TripletDirectionAboveClassifier.save()
//    TripletDirectionBehindClassifier.save()
//    TripletDirectionBelowClassifier.save()
//    TripletDirectionFrontClassifier.save()
//    TripletDirectionLeftClassifier.save()
//    TripletDirectionRightClassifier.save()

  }

  if (!isTrain) {

    println("testing started ...")

    TrajectorRoleClassifier.load()
    LandmarkRoleClassifier.load()
    IndicatorRoleClassifier.load()
    TripletRelationClassifier.load()
    TripletGeneralTypeClassifier.load()
    TripletGeneralDirectionClassifier.load()
    TripletGeneralRegionClassifier.load()
    TripletSpecificTypeClassifier.load()
    TripletRegionClassifier.load()
    TripletDirectionClassifier.load()

    PrepositionClassifier.load()
//    PrepositionAboveClassifier.load()
//    PrepositionAroundClassifier.load()
//    PrepositionAtClassifier.load()
//    PrepositionBehindClassifier.load()
//    PrepositionBetweenClassifier.load()
//    PrepositionInBetweenClassifier.load()
//    PrepositionInClassifier.load()
//    PrepositionInFrontOfClassifier.load()
//    PrepositionInTheMiddleOfClassifier.load()
//    PrepositionLeaningOnClassifier.load()
//    PrepositionNearClassifier.load()
//    PrepositionNextToClassifier.load()
//    PrepositionOnClassifier.load()
//    PrepositionOnEachSideClassifier.load()
//    PrepositionOverClassifier.load()
//    PrepositionSittingAroundClassifier.load()
//    PrepositionWithClassifier.load()

//    TripletRegionTPPClassifier.load()
//    TripletRegionEQClassifier.load()
//    TripletRegionDCClassifier.load()
//    TripletRegionECClassifier.load()
//    TripletRegionPOClassifier.load()
//
//    TripletDirectionAboveClassifier.load()
//    TripletDirectionBehindClassifier.load()
//    TripletDirectionBelowClassifier.load()
//    TripletDirectionFrontClassifier.load()
//    TripletDirectionLeftClassifier.load()
//    TripletDirectionRightClassifier.load()


    val spCandidatesTest = CandidateGenerator.getIndicatorCandidates(phrases().toList)
    val trCandidatesTest = CandidateGenerator.getTrajectorCandidates(phrases().toList)
      .filterNot(x => spCandidatesTest.contains(x))
    val lmCandidatesTest = CandidateGenerator.getLandmarkCandidates(phrases().toList)
      .filterNot(x => spCandidatesTest.contains(x))

    populateTripletDataFromAnnotatedCorpus(
      x => trCandidatesTest.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "true",
      x => lmCandidatesTest.exists(_.getId == x.getId))

    if (!useConstraints) {
      val trajectors = phrases.getTestingInstances.filter(x => TrajectorRoleClassifier(x) == "true").toList
      val landmarks = phrases.getTestingInstances.filter(x => LandmarkRoleClassifier(x) == "true").toList
      val indicators = phrases.getTestingInstances.filter(x => IndicatorRoleClassifier(x) == "true").toList

      val tripletList = triplets.getTestingInstances
        .filter(x => TripletRelationClassifier(x) == "true").toList

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

      val generalDirection = TripletGeneralDirectionClassifier.test()
      ReportHelper.saveEvalResults(outStream, "General Direction(within data model)", generalDirection)

      val generalRegion = TripletGeneralRegionClassifier.test()
      ReportHelper.saveEvalResults(outStream, "General Region(within data model)", generalRegion)

      val direction = TripletDirectionClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Direction(within data model)", direction)

      val behind = TripletDirectionBehindClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Direction behind(within data model)", behind)

      val below = TripletDirectionBelowClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Direction below(within data model)", below)

      val above = TripletDirectionAboveClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Direction above(within data model)", above)

      val left = TripletDirectionLeftClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Direction left(within data model)", left)

      val right = TripletDirectionRightClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Direction right(within data model)", right)

      val front = TripletDirectionFrontClassifier.test()
      ReportHelper.saveEvalResults(outStream, "Direction front(within data model)", front)

//      val region = TripletRegionClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "Region(within data model)", region)
//
//      val TPP = TripletRegionTPPClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "Region TPP(within data model)", TPP)
//
//      val EC = TripletRegionECClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "Region EC(within data model)", EC)
//
//      val EQ = TripletRegionEQClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "Region EQ(within data model)", EQ)
//
//      val DC = TripletRegionDCClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "Region DC(within data model)", DC)
//
//      val PO = TripletRegionPOClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "Region PO(within data model)", PO)
//
//      val visual = PrepositionClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "Visual triplet(within data model)", visual)
//
//      report(x => TripletRelationClassifier(x),
//        x => TrajectorRoleClassifier(x),
//        x => LandmarkRoleClassifier(x),
//        x => IndicatorRoleClassifier(x),
//        x => TripletGeneralTypeClassifier(x),
//        x => TripletDirectionClassifier(x),
//        x => TripletRegionClassifier(x)
//      )
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

//      val general = TripletGeneralTypeConstraintClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "General(within data model)", general)
//
//      val direction = TripletDirectionConstraintClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "Direction(within data model)", direction)
//
//      val region = TripletRegionConstraintClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "Region(within data model)", region)
//
//      val imRegion = TripletImageRegionClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "Image Region(within data model)", imRegion)
//
//      val visual = PrepositionClassifier.test()
//      ReportHelper.saveEvalResults(outStream, "Preposition(within data model)", visual)

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

