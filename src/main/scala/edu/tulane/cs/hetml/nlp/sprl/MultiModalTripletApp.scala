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
      println("Relation:" + triplets.getTrainingInstances.count(x=>x.getProperty("Relation") == "true"))
      println("Relations Image Confirmed:" + triplets.getTrainingInstances.count(x=>x.getProperty("Relation") == "true"
        && tripletImageConfirms(x)=="true"))
      println("Relations Image Wrongly Confirmed:" + triplets.getTrainingInstances.count(x=>x.getProperty("Relation") != "true"
      && tripletImageConfirms(x)=="true"))
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

