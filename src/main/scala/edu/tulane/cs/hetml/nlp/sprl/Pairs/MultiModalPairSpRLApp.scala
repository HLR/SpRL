package edu.tulane.cs.hetml.nlp.sprl.Pairs

import java.io.File

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{FeatureSets, ReportHelper}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalPopulateData._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Pairs.MultiModalSpRLPairClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator.{isTrain, _}
import org.apache.commons.io.FileUtils

object MultiModalPairSpRLApp extends App with Logging{

  val expName = "pairs_" + ((model, useConstraints) match {
    case (FeatureSets.BaseLine, false) => "BM"
    case (FeatureSets.BaseLine, true) => "BM+C"
    case (FeatureSets.WordEmbedding, false) => "BM+E"
    case (FeatureSets.WordEmbedding, true) => "BM+C+E"
    case (FeatureSets.WordEmbeddingPlusImage, false) => "BM+E+I"
    case (FeatureSets.WordEmbeddingPlusImage, true) => "BM+C+E+I"
    case _ =>
      logger.error("experiment is not supported")
      System.exit(1)
  })
  MultiModalSpRLPairClassifiers.featureSet = model

  val classifiers = List(
    TrajectorRoleClassifier,
    LandmarkRoleClassifier,
    IndicatorRoleClassifier,
    TrajectorPairClassifier,
    LandmarkPairClassifier,
    TripletGeneralTypeClassifier,
    TripletSpecificTypeClassifier,
    TripletRCC8Classifier,
    TripletDirectionClassifier
  )
  classifiers.foreach(x => {
    x.modelDir = s"models/mSpRL/pairs/$featureSet/"
    x.modelSuffix = suffix
  })
  FileUtils.forceMkdir(new File(resultsDir))

  populateRoleDataFromAnnotatedCorpus()

  if (isTrain) {
    println("training started ...")

    if(skipIndividualClassifiersTraining){
      TrajectorRoleClassifier.load()
      IndicatorRoleClassifier.load()
      LandmarkRoleClassifier.load()
    }
    else {
      TrajectorRoleClassifier.learn(iterations)
      IndicatorRoleClassifier.learn(iterations)
      LandmarkRoleClassifier.learn(iterations)
    }
    populatePairDataFromAnnotatedCorpus(x => IndicatorRoleClassifier(x) == "Indicator")
    ReportHelper.saveCandidateList(isTrain = true, pairs.getTrainingInstances.toList)

    if(skipIndividualClassifiersTraining) {
      TrajectorPairClassifier.load()
      LandmarkPairClassifier.load()
    }
    else{
      TrajectorPairClassifier.learn(iterations)
      LandmarkPairClassifier.learn(iterations)
    }

    if (jointTrain) {
      //JoinTraining using constraints
      //To make the trianing faster use the pre-trained models
      // then apply 10 joint training iterations
      /*      JointTrainSparseNetwork(sentences, TRConstraintClassifier :: LMConstraintClassifier ::
              IndicatorConstraintClassifier :: TRPairConstraintClassifier ::
              LMPairConstraintClassifier :: Nil, init = false, it = 10)*/
    }

    TrajectorRoleClassifier.save()
    IndicatorRoleClassifier.save()
    LandmarkRoleClassifier.save()
    TrajectorPairClassifier.save()
    LandmarkPairClassifier.save()

    populateTripletDataFromAnnotatedCorpusFromPairs(
      x => TrajectorPairClassifier(x) == "TR-SP",
      x => IndicatorRoleClassifier(x) == "Indicator",
      x => LandmarkPairClassifier(x) == "LM-SP"
    )

    val goldTriplets = triplets.getTrainingInstances.filter(_.containsProperty("ActualId"))
    TripletGeneralTypeClassifier.learn(iterations, goldTriplets)
    TripletGeneralTypeClassifier.save()

    TripletSpecificTypeClassifier.learn(iterations, goldTriplets)
    TripletSpecificTypeClassifier.save()

    TripletRCC8Classifier.learn(iterations, goldTriplets)
    TripletRCC8Classifier.save()

    TripletDirectionClassifier.learn(iterations, goldTriplets)
    TripletDirectionClassifier.save()

  }

  if (!isTrain) {

    println("testing started ...")

    TrajectorRoleClassifier.load()
    LandmarkRoleClassifier.load()
    IndicatorRoleClassifier.load()
    TrajectorPairClassifier.load()
    LandmarkPairClassifier.load()
    TripletGeneralTypeClassifier.load()
    TripletSpecificTypeClassifier.load()
    TripletRCC8Classifier.load()
    TripletDirectionClassifier.load()
    populatePairDataFromAnnotatedCorpus(x => IndicatorRoleClassifier(x) == "Indicator")
    ReportHelper.saveCandidateList(isTrain = false, pairs.getTestingInstances.toList)

    if (!useConstraints) {

      populateTripletDataFromAnnotatedCorpusFromPairs(
        x => TrajectorPairClassifier(x) == "TR-SP",
        x => IndicatorRoleClassifier(x) == "Indicator",
        x => LandmarkPairClassifier(x) == "LM-SP"
      )

      val trajectors = phrases.getTestingInstances.filter(x => TrajectorRoleClassifier(x) == "Trajector").toList
      val landmarks = phrases.getTestingInstances.filter(x => LandmarkRoleClassifier(x) == "Landmark").toList
      val indicators = phrases.getTestingInstances.filter(x => IndicatorRoleClassifier(x) == "Indicator").toList
      val tripletList = triplets.getTestingInstances.toList

      ReportHelper.saveAsXml(tripletList, trajectors, indicators, landmarks,
        x => TripletGeneralTypeClassifier(x),
        x => TripletSpecificTypeClassifier(x),
        x => TripletRCC8Classifier(x),
        x => TripletDirectionClassifier(x),
        s"$resultsDir/${expName}${suffix}.xml")

    }
    else {

      populateTripletDataFromAnnotatedCorpusFromPairs(
        x => PairsSentenceLevelConstraintClassifiers.TRPairConstraintClassifier(x) == "TR-SP",
        x => PairsSentenceLevelConstraintClassifiers.IndicatorConstraintClassifier(x) == "Indicator",
        x => PairsSentenceLevelConstraintClassifiers.LMPairConstraintClassifier(x) == "LM-SP"
      )

      val trajectors = phrases.getTestingInstances.filter(x => PairsSentenceLevelConstraintClassifiers.TRConstraintClassifier(x) == "Trajector").toList
      val landmarks = phrases.getTestingInstances.filter(x => PairsSentenceLevelConstraintClassifiers.LMConstraintClassifier(x) == "Landmark").toList
      val indicators = phrases.getTestingInstances.filter(x => PairsSentenceLevelConstraintClassifiers.IndicatorConstraintClassifier(x) == "Indicator").toList
      val tripletList = triplets.getTestingInstances.toList

      ReportHelper.reportTripletResults(testFile, resultsDir, s"${expName}${suffix}_triplet", tripletList, globalSpans)

      ReportHelper.saveAsXml(tripletList, trajectors, indicators, landmarks,
        x => TripletGeneralTypeClassifier(x),
        x => TripletSpecificTypeClassifier(x),
        x => TripletRCC8Classifier(x),
        x => TripletDirectionClassifier(x),
        s"$resultsDir/${expName}${suffix}.xml")
    }

    ReportHelper.saveEvalResultsFromXmlFile(testFile, s"$resultsDir/${expName}${suffix}.xml", s"$resultsDir/$expName$suffix.txt")
  }

}

