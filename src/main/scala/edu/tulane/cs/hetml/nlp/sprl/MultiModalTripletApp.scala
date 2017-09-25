package edu.tulane.cs.hetml.nlp.sprl

import java.io.File

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{CandidateGenerator, FeatureSets, ReportHelper}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalPopulateData._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import edu.tulane.cs.hetml.nlp.sprl.SentenceLevelConstraintClassifiers._
import org.apache.commons.io.FileUtils
import edu.illinois.cs.cogcomp.saul.classifier.{JointTrainSparseNetwork, JointTrainSparsePerceptron}

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
    TripletSpecificTypeClassifier,
    TripletRCC8Classifier,
    TripletFoRClassifier
  )
  classifiers.foreach(x => {
    x.modelDir = s"models/mSpRL/$featureSet/"
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
    TripletRCC8Classifier.learn(iterations)
    TripletFoRClassifier.learn(iterations)

//    println("Jointly Training...")

//    val cls = List(TripletRelationTypeConstraintClassifier, TripletGeneralTypeConstraintClassifier)
//    JointTrainSparseNetwork.train(sentences, cls, iterations, true)

//    println("Jointly Training Done. Saving...")

    TripletRelationClassifier.save()
    TripletGeneralTypeClassifier.save()
    TripletSpecificTypeClassifier.save()
    TripletRCC8Classifier.save()
    TripletFoRClassifier.save()

  }

  if (!isTrain) {

/*    documents.clear()
    sentences.clear()
    tokens.clear()
    phrases.clear()
    pairs.clear()
    triplets.clear()

    mSpRLConfigurator.isTrain = false

    populateRoleDataFromAnnotatedCorpus()
*/
    println("testing started ...")

    TrajectorRoleClassifier.load()
    LandmarkRoleClassifier.load()
    IndicatorRoleClassifier.load()
    TripletRelationClassifier.load()
    TripletGeneralTypeClassifier.load()
    TripletSpecificTypeClassifier.load()
    TripletRCC8Classifier.load()
    TripletFoRClassifier.load()

      val trCandidatesTest = (CandidateGenerator.getTrajectorCandidates(phrases().toList))
      val lmCandidatesTest = (CandidateGenerator.getLandmarkCandidates(phrases().toList))

      populateTripletDataFromAnnotatedCorpus(
          x => trCandidatesTest.exists(_.getId == x.getId),
          x => IndicatorRoleClassifier(x) == "Indicator",
          x => lmCandidatesTest.exists(_.getId == x.getId))

    val trajectors = phrases.getTestingInstances.filter(x => TrajectorRoleClassifier(x) == "Trajector").toList
    val landmarks = phrases.getTestingInstances.filter(x => LandmarkRoleClassifier(x) == "Landmark").toList
    val indicators = phrases.getTestingInstances.filter(x => IndicatorRoleClassifier(x) == "Indicator").toList

    val tripletList = triplets.getTestingInstances
      .filter(x=> TripletRelationClassifier(x) == "Relation").toList


    TripletRelationTypeConstraintClassifier.test()

//    TripletGeneralTypeConstraintClassifier.test()

    TripletRelationClassifier.test()
//    TripletGeneralTypeClassifier.test()
//    TripletSpecificTypeClassifier.test()
//    TripletRCC8Classifier.test()
//    TripletFoRClassifier.test()
/*
    ReportHelper.saveAsXml(tripletList, trajectors, indicators, landmarks,
      x => TripletGeneralTypeClassifier(x),
      x => TripletSpecificTypeClassifier(x),
      x => TripletRCC8Classifier(x),
      x => TripletFoRClassifier(x),
      s"$resultsDir/${expName}${suffix}.xml")

    ReportHelper.saveEvalResultsFromXmlFile(testFile, s"$resultsDir/${expName}${suffix}.xml", s"$resultsDir/$expName$suffix.txt")
*/
  }
}

