package edu.tulane.cs.hetml.nlp.sprl

import java.io.File

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{CandidateGenerator, FeatureSets, ReportHelper}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalPopulateData._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import org.apache.commons.io.FileUtils

object MultiModalTripletApp extends App with Logging{

  val expName = (model, useConstraints) match {
    case (FeatureSets.BaseLine, false) => "BM"
    case (FeatureSets.BaseLineWithImage, false) => "BM+I"
    case (FeatureSets.BaseLine, true) => "BM+C"
    case (FeatureSets.WordEmbedding, false) => "BM+E"
    case (FeatureSets.WordEmbedding, true) => "BM+C+E"
    case (FeatureSets.WordEmbeddingPlusImage, false) => "BM+E+I"
    case (FeatureSets.WordEmbeddingPlusImage, true) => "BM+C+E+I"
    case _ =>
      logger.error("experiment no supported")
      System.exit(1)
  }
  MultiModalSpRLClassifiers.featureSet = model

  val classifiers = List(
    TrajectorRoleClassifier,
    LandmarkRoleClassifier,
    IndicatorRoleClassifier,
    TripletRelationClassifier,
    TripletGeneralTypeClassifier,
    TripletRCC8Classifier,
    TripletRelationClassifierWithImage
  )
  classifiers.foreach(x => {
    x.modelDir = s"models/mSpRL/$featureSet/"
    x.modelSuffix = suffix
  })
  FileUtils.forceMkdir(new File(resultsDir))

  populateRoleDataFromAnnotatedCorpus()

  if (isTrain) {
    println("training started ...")

//    TrajectorRoleClassifier.learn(iterations)
    IndicatorRoleClassifier.learn(iterations)
//    LandmarkRoleClassifier.learn(iterations)

//    TrajectorRoleClassifier.save()
    IndicatorRoleClassifier.save()
//    LandmarkRoleClassifier.save()

    val trCandidates = (CandidateGenerator.getTrajectorCandidates(phrases().toList))
    val lmCandidates = (CandidateGenerator.getLandmarkCandidates(phrases().toList))
    val indicatorCandidates = (CandidateGenerator.getIndicatorCandidates(phrases().toList))


    populateTripletDataFromAnnotatedCorpus(
      x => trCandidates.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "Indicator",
      x => lmCandidates.exists(_.getId == x.getId)
        )

    println("Image LM Confirmed:" + triplets.getTrainingInstances.count(x=> x.getProperty("Relation") == "true"
      && tripletLMIsImageConceptExactMatch(x)=="true"))
    println("Image LM Wrong Confirmed:" + triplets.getTrainingInstances.count(x=> x.getProperty("Relation") != "true"
      && tripletLMIsImageConceptExactMatch(x)=="true"))
    println("Image TR Confirmed:" + triplets.getTrainingInstances.count(x=>x.getProperty("Relation") == "true" &&
      tripletTRIsImageConceptExactMatch(x)=="true"))
    println("Image TR Wrong Confirmed:" + triplets.getTrainingInstances.count(x=>x.getProperty("Relation") != "true" &&
      tripletTRIsImageConceptExactMatch(x)=="true"))
    println("Image TR-LM Confirmed:" + triplets.getTrainingInstances.count(x=> x.getProperty("Relation") == "true"
      && tripletTRLMIsImageConcept(x)=="true"))
    println("Image TR-LM Wrong Confirmed:" + triplets.getTrainingInstances.count(x=> x.getProperty("Relation") != "true"
      && tripletTRLMIsImageConcept(x)=="true"))

/*      println("Candidate Triplets Training Size -> " + triplets.getTrainingInstances.size)
      println("Relation:" + triplets.getTrainingInstances.count(x=>x.getProperty("Relation") == "true"))
      println("Relations Image Confirmed:" + triplets.getTrainingInstances.count(x=>x.getProperty("Relation") == "true"
        && tripletImageConfirms(x)=="true"))
      println("Relations Image Wrongly Confirmed:" + triplets.getTrainingInstances.count(x=>x.getProperty("Relation") != "true"
      && tripletImageConfirms(x)=="true")) */
/*    triplets().foreach(t => {
      println("Triplet->" + t.getArgument(0) + "-" + t.getArgument(1) + "-" + t.getArgument(2))
      println("Triplet parent->" + t.getParent.getId)
      println("JF2_1->" + JF2_1(t))
      println("JF2_2->" + JF2_2(t))
      println("JF2_3->" + JF2_3(t))
      println("JF2_4->" + JF2_4(t))
      println("JF2_5->" + JF2_5(t))
      println("JF2_6->" + JF2_6(t))
      println("JF2_7->" + JF2_7(t))
      println("JF2_8->" + JF2_8(t))
      println("JF2_9->" + JF2_9(t))
      println("JF2_10->" + JF2_10(t))
      println("JF2_11->" + JF2_11(t))
      println("JF2_13->" + JF2_13(t))
      println("JF2_14->" + JF2_14(t))
      println("JF2_15->" + JF2_15(t))
    })*/

    TripletRelationClassifier.learn(iterations)
//    TripletRelationClassifierWithImage.learn(iterations)
//    TripletGeneralTypeClassifier.learn(iterations)
//    TripletRCC8Classifier.learn(iterations)
//
    TripletRelationClassifier.save()
//    TripletRelationClassifierWithImage.save()
//    TripletGeneralTypeClassifier.save()
//    TripletRCC8Classifier.save()
  }

  if (!isTrain) {

    println("testing started ...")
//    TrajectorRoleClassifier.load()
//    LandmarkRoleClassifier.load()
    IndicatorRoleClassifier.load()
    TripletRelationClassifier.load()
//    TripletGeneralTypeClassifier.load()
//    TripletRCC8Classifier.load()
//    TripletRelationClassifierWithImage.load()

      val trCandidates = (CandidateGenerator.getTrajectorCandidates(phrases().toList))
      val lmCandidates = (CandidateGenerator.getLandmarkCandidates(phrases().toList))
      val indicatorCandidates = (CandidateGenerator.getIndicatorCandidates(phrases().toList))

      populateTripletDataFromAnnotatedCorpus(
          x => trCandidates.exists(_.getId == x.getId),
          x => IndicatorRoleClassifier(x) == "Indicator",
          x => lmCandidates.exists(_.getId == x.getId))

    println("Image LM Confirmed:" + triplets.getTestingInstances.count(x=> x.getProperty("Relation") == "true"
      && tripletLMIsImageConceptExactMatch(x)=="true"))
    println("Image LM Wrong Confirmed:" + triplets.getTestingInstances.count(x=> x.getProperty("Relation") != "true"
      && tripletLMIsImageConceptExactMatch(x)=="true"))
    println("Image TR Confirmed:" + triplets.getTestingInstances.count(x=>x.getProperty("Relation") == "true" &&
      tripletTRIsImageConceptExactMatch(x)=="true"))
    println("Image TR Wrong Confirmed:" + triplets.getTestingInstances.count(x=>x.getProperty("Relation") != "true" &&
      tripletTRIsImageConceptExactMatch(x)=="true"))
    println("Image TR-LM Confirmed:" + triplets.getTestingInstances.count(x=> x.getProperty("Relation") == "true"
      && tripletTRLMIsImageConcept(x)=="true"))
    println("Image TR-LM Wrong Confirmed:" + triplets.getTestingInstances.count(x=> x.getProperty("Relation") != "true"
      && tripletTRLMIsImageConcept(x)=="true"))

//      SentenceLevelConstraintClassifiers.TripletRelationTypeConstraintClassifier.test()
      TripletRelationClassifier.test()
//    TripletRelationClassifierWithImage.test()
/*    TripletGeneralTypeClassifier.test()
    TripletRCC8Classifier.test()

    val trajectors = phrases.getTestingInstances.filter(x => TrajectorRoleClassifier(x) == "Trajector").toList
    val landmarks = phrases.getTestingInstances.filter(x => LandmarkRoleClassifier(x) == "Landmark").toList
    val indicators = phrases.getTestingInstances.filter(x => IndicatorRoleClassifier(x) == "Indicator").toList

    val tripletList = triplets.getTestingInstances
      .filter(x=> TripletRelationClassifier(x) == "Relation").toList

    ReportHelper.saveAsXml(tripletList, trajectors, indicators, landmarks,
      x => TripletGeneralTypeClassifier(x),
      x => "",
      x => TripletRCC8Classifier(x),
      x => "",
      s"$resultsDir/${expName}${suffix}.xml")

    ReportHelper.saveEvalResultsFromXmlFile(testFile, s"$resultsDir/${expName}${suffix}.xml", s"$resultsDir/$expName$suffix.txt")*/
  }
}

