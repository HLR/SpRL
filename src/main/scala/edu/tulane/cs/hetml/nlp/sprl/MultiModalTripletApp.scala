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
import edu.tulane.cs.hetml.nlp.BaseTypes.Relation

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

/*  TrajectorRoleClassifier.learn(iterations)
    IndicatorRoleClassifier.learn(iterations)
    LandmarkRoleClassifier.learn(iterations)

    TrajectorRoleClassifier.save()
    IndicatorRoleClassifier.save()
    LandmarkRoleClassifier.save()
*/

    TrajectorRoleClassifier.load()
    IndicatorRoleClassifier.load()
    LandmarkRoleClassifier.load()

    val trCandidates = (CandidateGenerator.getTrajectorCandidates(phrases().toList))
    val lmCandidates = (CandidateGenerator.getLandmarkCandidates(phrases().toList))
    val indicatorCandidates = (CandidateGenerator.getIndicatorCandidates(phrases().toList))


    populateTripletDataFromAnnotatedCorpus(
      x => trCandidates.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "Indicator",
      x => lmCandidates.exists(_.getId == x.getId)
        )

/*    TripletRelationClassifier.learn(iterations)
    TripletGeneralTypeClassifier.learn(iterations)
    TripletSpecificTypeClassifier.learn(iterations)
    TripletRCC8Classifier.learn(iterations)
    TripletFoRClassifier.learn(iterations)
*/
    TripletRelationClassifier.load()
    TripletGeneralTypeClassifier.load()
    TripletSpecificTypeClassifier.load()
    TripletRCC8Classifier.load()
    TripletFoRClassifier.load()

    JointTrainSparsePerceptron.train(MultiModalSpRLDataModel.sentences, List(TripletRelationTypeConstraintClassifier,
      TripletGeneralTypeConstraintClassifier), 10)// TripletSpecificTypeConstraintClassifier,
//      TripletRCC8TypeConstraintClassifier, TripletForTypeConstraintClassifier) , 10)


/*    TripletRelationClassifier.save()
    TripletGeneralTypeClassifier.save()
    TripletSpecificTypeClassifier.save()
    TripletRCC8Classifier.save()
    TripletFoRClassifier.save()
*/
  }

  if (!isTrain) {

    println("testing started ...")
    TrajectorRoleClassifier.load()
    LandmarkRoleClassifier.load()
    IndicatorRoleClassifier.load()
    TripletRelationClassifier.load()
    TripletGeneralTypeClassifier.load()
    TripletSpecificTypeClassifier.load()
    TripletRCC8Classifier.load()
    TripletFoRClassifier.load()

      val trCandidates = (CandidateGenerator.getTrajectorCandidates(phrases().toList))
      val lmCandidates = (CandidateGenerator.getLandmarkCandidates(phrases().toList))
      val indicatorCandidates = (CandidateGenerator.getIndicatorCandidates(phrases().toList))

      populateTripletDataFromAnnotatedCorpus(
          x => trCandidates.exists(_.getId == x.getId),
          x => IndicatorRoleClassifier(x) == "Indicator",
          x => lmCandidates.exists(_.getId == x.getId))

    val trajectors = phrases.getTestingInstances.filter(x => TrajectorRoleClassifier(x) == "Trajector").toList
    val landmarks = phrases.getTestingInstances.filter(x => LandmarkRoleClassifier(x) == "Landmark").toList
    val indicators = phrases.getTestingInstances.filter(x => IndicatorRoleClassifier(x) == "Indicator").toList

    val tripletList = triplets.getTestingInstances
      .filter(x=> TripletRelationClassifier(x) == "Relation").toList

    TripletRelationClassifier.test()

    ReportHelper.saveAsXml(tripletList, trajectors, indicators, landmarks,
      x => TripletGeneralTypeClassifier(x),
      x => TripletSpecificTypeClassifier(x),
      x => TripletRCC8Classifier(x),
      x => TripletFoRClassifier(x),
      s"$resultsDir/${expName}${suffix}.xml")

    ReportHelper.saveEvalResultsFromXmlFile(testFile, s"$resultsDir/${expName}${suffix}.xml", s"$resultsDir/$expName$suffix.txt")
  }
}

