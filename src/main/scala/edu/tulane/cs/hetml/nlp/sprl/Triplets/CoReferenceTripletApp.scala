package edu.tulane.cs.hetml.nlp.sprl.Triplets

import java.io.PrintWriter

import edu.illinois.cs.cogcomp.saul.classifier.{ConstrainedClassifier, Learnable}
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.BaseTypes.{Phrase, Relation, Sentence}
import edu.tulane.cs.hetml.nlp.sprl.Eval.SpRLEvaluation
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors.getGoogleSimilarity
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalPopulateData._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.TripletSentenceLevelConstraintClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.tripletConfigurator._
import edu.tulane.cs.hetml.relations.RelationInformationReader
import me.tongfei.progressbar.ProgressBar

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

object CoReferenceTripletApp extends App with Logging {

//  val relationReader = new RelationInformationReader();
//  relationReader.loadRelations(imageDataPath);
//  val visualgenomeRelationsList = relationReader.visualgenomeRelations.toList

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
      x => lmCandidatesTrain.exists(_.getId == x.getId)
    )

    tripletClassifiers.foreach {
      x =>
        x.learn(iterations)
        x.test(triplets())
        x.save()
    }
  }

  if (!isTrain) {

    roleClassifiers.foreach {
      x => x.load()
    }
    tripletClassifiers.foreach {
      x => x.load()
    }

    println("testing started ...")

    val spCandidatesTest = TripletCandidateGenerator.getIndicatorCandidates(phrases.getTestingInstances.toList)
    val trCandidatesTest = TripletCandidateGenerator.getTrajectorCandidates(phrases.getTestingInstances.toList)
      .filterNot(x => spCandidatesTest.contains(x))
    val lmCandidatesTest = TripletCandidateGenerator.getLandmarkCandidates(phrases.getTestingInstances.toList)
      .filterNot(x => spCandidatesTest.contains(x))

    populateTripletDataFromAnnotatedCorpus(
      x => trCandidatesTest.exists(_.getId == x.getId),
      x => IndicatorRoleClassifier(x) == "true",
      x => lmCandidatesTest.exists(_.getId == x.getId))

//    tripletClassifiers.foreach {
//      x =>
//        x.test(triplets())
//    }

    constraintRoleClassifiers.foreach {
      x => x.test()
    }

    constraintTripletClassifiers.foreach {
      x =>
        x.test()
    }
  }

//  def getCoReferenceProcesing(lmFilter: (Phrase) => Boolean) = {
//    if(useCoReference) {
////      val writer =
////        if(isTrain)
////          new PrintWriter(resultsDir + "/ImplicitLandmarksTrainReplacement.txt")
////        else
////          new PrintWriter(resultsDir + "/ImplicitLandmarksTestReplacement.txt")
//      println("Processing for Co-Reference...")
//      //**
//      // Landmark Candidates
//      val instances = if (isTrain) phrases.getTrainingInstances else phrases.getTestingInstances
//      val landmarks = instances.filter(t => t.getId != dummyPhrase.getId && lmFilter(t)).toList
//        .sortBy(x => x.getSentence.getStart + x.getStart)
//
//      //**
//      // Headwords of triplets
//      relationReader.loadClefRelationHeadwords(imageDataPath, isTrain)
//      val clefHWRelation = relationReader.clefHeadwordRelations.toList
//
//      val p = new ProgressBar("Processing Relations", triplets().filter(r => r.getArgument(2).toString=="it").size)
//      p.start()
//      triplets().filter(r => r.getArgument(2).toString=="it" && r.getProperty("Relation")=="true").foreach(r => {
//        p.step()
//        val headWordsTriplet = clefHWRelation.filter(c => {
//          c.getId==r.getId
//        })
//        //**
//        // get possible Landmarks from Visual Genome
//        val possibleLMs = visualgenomeRelationsList.filter(v => v.getPredicate==headWordsTriplet.head.getSp
//          && v.getSubject==headWordsTriplet.head.getTr)
//
//        if(possibleLMs.size>0) {
//          //Count Unique Instances
//          var uniqueRelsForLM = scala.collection.mutable.Map[String, Int]()
//          possibleLMs.foreach(t => {
//            val key = t.getSubject + "-" + t.getPredicate + "-" + t.getObject
//            if(!(uniqueRelsForLM.keySet.exists(_ == key)))
//              uniqueRelsForLM += (key -> 1)
//            else {
//              var count = uniqueRelsForLM.get(key).get
//              count = count + 1
//              uniqueRelsForLM.update(key, count)
//            }
//          })
//
//          val rSId = r.getArgumentId(0).split("\\(")(0)
//          val sentenceLMs =
//            if(useCrossSentence) {
//              val docId = sentences().filter(s => s.getId==rSId).head.getDocument.getId
//              val sens = sentences().filter(s => s.getDocument.getId==docId)
//              landmarks.filter(l => {
//                sens.exists(s => {
//                  s.getId==l.getSentence.getId
//                }) && l.getText!=r.getArgument(0).toString && l.getText!=r.getArgument(2).toString
//              })
//            }
//            else {
//              //**
//              // get all landmark candidates for the sentence
//              landmarks.filter(l => {
//                l.getSentence.getId == rSId && l.getText!=r.getArgument(0).toString && l.getText!=r.getArgument(2).toString
//              })
//            }
//          //**
//          // Similarity scroes for each
//          val w2vVectorScores = uniqueRelsForLM.toList.map(t => {
//            val rSplited = t._1.split("-")
//            if(rSplited.length==3) {
//              val v = getMaxW2VScore(rSplited(2), sentenceLMs)
//              v
//            }
//            else {
//              val dummyLM = new Phrase()
//              dummyLM.setText("None")
//              dummyLM.setId("dummy")
//              (dummyLM, -1, 0.0)
//            }
//          })
//          val ProbableLandmark = w2vVectorScores.sortBy(_._3)
//          val gRel = r.getProperty("Relation")
//          val pRel = TripletRelationClassifier(r)
//          println("R ->" + gRel + "P ->" + pRel)
//          println("Features ->" + JF2_1(r) + " " + JF2_2(r) + " " + JF2_3(r) + " " + JF2_4(r) + " " + JF2_5(r) + " " + JF2_6(r) + " " +
//            JF2_8(r) + " " + JF2_9(r) + " " + JF2_10(r) + " " + JF2_11(r) + " " + JF2_13(r) + " " + JF2_14(r) + " " + JF2_15(r) + " " +
//            tripletSpWithoutLandmark(r) + " " + tripletPhrasePos(r) + " " + tripletDependencyRelation(r) + " " + tripletHeadWordPos(r) + " " +
//            tripletLmBeforeSp(r) + " " + tripletTrBeforeLm(r) + " " + tripletTrBeforeSp(r) + " " +
//            tripletDistanceTrSp(r) + " " + tripletDistanceLmSp(r))
//          var count = 0
//          ProbableLandmark.foreach(p => {
//            if(p._3>0) {
//              val rNew = new Relation()
//              rNew.setId(r.getId+"~"+count)
//              rNew.setArgument(0, r.getArgument(0))
//              rNew.setArgument(1, r.getArgument(1))
//              rNew.setArgument(2, p._1)
//
//              rNew.setArgumentId(0, r.getArgumentId(0))
//              rNew.setArgumentId(1, r.getArgumentId(1))
//              rNew.setArgumentId(2, p._1.getId)
//
//              rNew.setParent(r.getParent)
//
//              rNew.setProperty("RCC8", r.getProperty("RCC8"))
//              rNew.setProperty("Relation", r.getProperty("Relation"))
//              rNew.setProperty("FoR", r.getProperty("FoR"))
//              rNew.setProperty("SpecificType", r.getProperty("SpecificType"))
//              rNew.setProperty("ActualId", r.getProperty("ActualId"))
//              rNew.setProperty("GeneralType", r.getProperty("GeneralType"))
//
//              coReferenceTriplets += rNew
//              count = count + 1
//            }
//          })
//
////          if(ProbableLandmark._3>0.25 && ProbableLandmark._1.getText!="None") {
////            val pLM = headWordFrom(ProbableLandmark._1)
////            r.getArgument(2).setText(pLM)
////            //r.setProperty("ProbableLandmark", pLM)
////          }
////          else {
////            //r.setProperty("ProbableLandmark", "None")
////          }
//        }
//        else {
//          //r.setProperty("ProbableLandmark", "None")
//        }
//      })
//      val gt = triplets().filter(t=> t.getArgument(2).toString=="it")
//      gt.foreach(g => {
//        coReferenceTriplets += g
//      })
//      p.stop()
//      triplets.clear()
//      triplets.populate(coReferenceTriplets, isTrain)
//      TripletRelationClassifier.test()
//      //          triplets().foreach(t => {
//      //            println("Features ->" + JF2_1(t) + " " + JF2_2(t) + " " + JF2_3(t) + " " + JF2_4(t) + " " + JF2_5(t) + " " + JF2_6(t) + " " +
//      //              JF2_8(t) + " " + JF2_9(t) + " " + JF2_10(t) + " " + JF2_11(t) + " " + JF2_13(t) + " " + JF2_14(t) + " " + JF2_15(t) + " " +
//      //              tripletSpWithoutLandmark(t) + " " + tripletPhrasePos(t) + " " + tripletDependencyRelation(t) + " " + tripletHeadWordPos(t) + " " +
//      //              tripletLmBeforeSp(t) + " " + tripletTrBeforeLm(t) + " " + tripletTrBeforeSp(t) + " " +
//      //              tripletDistanceTrSp(t) + " " + tripletDistanceLmSp(t))
//      //          })
//      var count = 0
//      triplets().groupBy(t=> t.getId.split("~")(0)).foreach(t=> {
//        var found = false
//        t._2.foreach(c => {
//          val res = TripletRelationClassifier(c.asInstanceOf[Relation])
//          if(res==true && c.getProperty("Relation")=="true")
//            found = true
//        })
//        if(found==true)
//          count = count + 1
//      })
//      println(count)
//
//
//      //writer.close()
//    }
//  }
//
//  def getMaxW2VScore(replacementLM: String, sentenceLMs: List[Phrase]): (Phrase, Int, Double) = {
//    val scores = sentenceLMs.map(w => {
//      val phraseText = w.getText.replaceAll("[^A-Za-z0-9]", " ")
//      phraseText.replaceAll("\\s+", " ");
//      w.setText(phraseText)
//      getGoogleSimilarity(replacementLM, headWordFrom(w))
//    })
//    val maxScore = scores.max
//    val maxIndex = scores.indexOf(scores.max)
//    val possibleLM = sentenceLMs(maxIndex)
//    (possibleLM, maxIndex, maxScore)
//  }

}
