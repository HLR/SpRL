package edu.tulane.cs.hetml.nlp.sprl

import java.io.File

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{CandidateGenerator, FeatureSets, ReportHelper}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalPopulateData._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._

object MultiModalTripletApp extends App with Logging{

  val expName = (model, useConstraints) match {
    case (FeatureSets.BaseLine, false) => "BM"
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
    TripletRelationClassifier
  )

  populateRoleDataFromAnnotatedCorpus()

  if (isTrain) {
    println("training started ...")

    val trCandidates = (CandidateGenerator.getTrajectorCandidates(phrases().toList))
    val lmCandidates = (CandidateGenerator.getLandmarkCandidates(phrases().toList))
    val indicatorCandidates = (CandidateGenerator.getIndicatorCandidates(phrases().toList))

    populateTripletDataFromAnnotatedCorpus(
      x => trCandidates.exists(_.getId == x.getId),
      x => indicatorCandidates.exists(p=> p.getId == x.getId),
      x => lmCandidates.exists(_.getId == x.getId))

      println("Candidate Triplets Size -> " + triplets.getTrainingInstances.size)
      println("None:" + triplets.getTrainingInstances.count(x=>x.getProperty("Relation") != "true"))
      println("Relation:" + triplets.getTrainingInstances.count(x=>x.getProperty("Relation") == "true"))

    TripletRelationClassifier.learn(50)
    TripletRelationClassifier.save()

  }

  if (!isTrain) {

    println("testing started ...")

    TripletRelationClassifier.load()

    if (!useConstraints) {

      val trCandidates = (CandidateGenerator.getTrajectorCandidates(phrases().toList))
      val lmCandidates = (CandidateGenerator.getLandmarkCandidates(phrases().toList))
      val indicatorCandidates = (CandidateGenerator.getIndicatorCandidates(phrases().toList))

      populateTripletDataFromAnnotatedCorpus(
          x => trCandidates.exists(_.getId == x.getId),
          x => indicatorCandidates.exists(p=> p.getId == x.getId),
          //x => IndicatorRoleClassifier(x) == "Indicator",
          x => lmCandidates.exists(_.getId == x.getId))

      println("Candidate Triplets Size -> " + triplets.getTestingInstances.size)
      println("None:" + triplets.getTestingInstances.count(x=>x.getProperty("Relation") != "true"))
      println("Relation:" + triplets.getTestingInstances.count(x=>x.getProperty("Relation") == "true"))

      TripletRelationClassifier.test()
    }
  }

}

