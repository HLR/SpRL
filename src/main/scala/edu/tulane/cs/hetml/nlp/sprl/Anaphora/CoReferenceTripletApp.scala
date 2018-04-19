package edu.tulane.cs.hetml.nlp.sprl.Anaphora

import java.io.PrintWriter

import edu.illinois.cs.cogcomp.saul.classifier.{ConstrainedClassifier, Learnable}
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation, Sentence}
import edu.tulane.cs.hetml.nlp.sprl.Eval.SpRLEvaluation
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors.getGoogleSimilarity
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.MultiModalPopulateData._
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.TripletSentenceLevelConstraintClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.tripletConfigurator._
import edu.tulane.cs.hetml.relations.RelationInformationReader
import me.tongfei.progressbar.ProgressBar

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

object CoReferenceTripletApp extends App with Logging {

  val coReferenceTriplets = new ListBuffer[Relation]()
  val roleClassifiers = List[Learnable[Phrase]](
    TrajectorRoleClassifier,
    LandmarkRoleClassifier,
    IndicatorRoleClassifier
  )

  val constraintRoleClassifiers = List[ConstrainedClassifier[Phrase, Sentence]](
    TRConstraintClassifier,
    LMConstraintClassifier,
    IndicatorConstraintClassifier
  )

  val tripletClassifiers = List[Learnable[Relation]](
    TripletRelationClassifier,
    TripletGeneralTypeClassifier,
    TripletRegionClassifier,
    TripletDirectionClassifier
  )
  val constraintTripletClassifiers = List[ConstrainedClassifier[Relation, Sentence]](
    TripletRelationConstraintClassifier,
    TripletGeneralTypeConstraintClassifier,
    TripletRegionConstraintClassifier,
    TripletDirectionConstraintClassifier
  )

  populateRoleDataFromAnnotatedCorpus()

  if (isTrain) {
    println("training started ...")

    roleClassifiers.foreach {
      x =>
        x.learn(iterations)
        x.test(phrases())
        x.save()
    }

    val spCandidatesTrain = TripletCandidateGenerator.getIndicatorCandidates(phrases().toList)
    val trCandidatesTrain = TripletCandidateGenerator.getTrajectorCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))
    val lmCandidatesTrain = TripletCandidateGenerator.getLandmarkCandidates(phrases().toList)
      .filterNot(x => spCandidatesTrain.contains(x))


    populateTripletDataFromAnnotatedCorpus(
      x => trCandidatesTrain.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "true",
      x => lmCandidatesTrain.exists(_.getId == x.getId),
      false
    )

    tripletClassifiers.foreach {
      x =>
        x.learn(iterations)
        x.test(triplets())
        x.save()
    }
    if(useModel=="M2") {
        populateTripletDataFromAnnotatedCorpus(
          x => trCandidatesTrain.exists(_.getId == x.getId),
          x => IndicatorRoleClassifier(x) == "true",
          x => lmCandidatesTrain.exists(_.getId == x.getId),
          true
        )

        TripletCoReferenceRelationClassifier.learn(iterations)
        TripletCoReferenceRelationClassifier.save()
    }
  }

  if (!isTrain) {

    roleClassifiers.foreach {
      x => x.load()
    }
    tripletClassifiers.foreach {
      x => x.load()
    }

    TripletCoReferenceRelationClassifier.load()

    println("testing started ...")

    val spCandidatesTest = TripletCandidateGenerator.getIndicatorCandidates(phrases.getTestingInstances.toList)
    val trCandidatesTest = TripletCandidateGenerator.getTrajectorCandidates(phrases.getTestingInstances.toList)
      .filterNot(x => spCandidatesTest.contains(x))
    val lmCandidatesTest = TripletCandidateGenerator.getLandmarkCandidates(phrases.getTestingInstances.toList)
      .filterNot(x => spCandidatesTest.contains(x))

    populateTripletDataFromAnnotatedCorpus(
      x => trCandidatesTest.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "true",
      x => lmCandidatesTest.exists(_.getId == x.getId), false)

    if(useModel=="M1") {
      roleClassifiers.foreach {
        x => x.test()
      }
      tripletClassifiers.foreach {
        x =>
          x.test()
      }
    }
    else if(useModel=="M2") {
      constraintRoleClassifiers.foreach {
        x => x.test()
      }
      constraintTripletClassifiers.foreach {
        x =>
          x.test()
      }
    }
  }
}
