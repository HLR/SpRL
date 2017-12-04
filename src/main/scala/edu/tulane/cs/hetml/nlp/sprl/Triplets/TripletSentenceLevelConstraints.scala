package edu.tulane.cs.hetml.nlp.sprl.Triplets

import edu.illinois.cs.cogcomp.lbjava.infer.{FirstOrderConstant, FirstOrderConstraint}
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.constraint.ConstraintTypeConversion._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator

import scala.collection.JavaConversions._

object TripletSentenceLevelConstraints {

  val sim = new SegmentPhraseSimilarityClassifier()
  val imageSupportsSp = new ImageSupportsSpClassifier()

  val roleShouldHaveRel = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      val sentTriplets = (sentences(s) ~> sentenceToTriplets).toList
      sentTriplets.foreach {
        x =>
          val trRel = sentTriplets.filter(r => r.getArgumentId(0) == x.getArgumentId(0))
            ._exists(x => (TripletRelationClassifier on x) is "Relation")

          val lmRel = sentTriplets.filter(r => r.getArgumentId(2) == x.getArgumentId(2))
            ._exists(x => (TripletRelationClassifier on x) is "Relation")

          val tr = TrajectorRoleClassifier on (triplets(x) ~> tripletToFirstArg).head is "Trajector"
          val lm = LandmarkRoleClassifier on (triplets(x) ~> tripletToThirdArg).head is "Landmark"

          a = a and (tr ==> trRel) and (lm ==> lmRel)
      }
      a
  }

  val roleIntegrity = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (
            (
              (TripletRelationClassifier on x) is "Relation") ==>
              (
                (TrajectorRoleClassifier on (triplets(x) ~> tripletToFirstArg).head is "Trajector") and
                  (IndicatorRoleClassifier on (triplets(x) ~> tripletToSecondArg).head is "Indicator") and
                  (LandmarkRoleClassifier on (triplets(x) ~> tripletToThirdArg).head is "Landmark")
                )
            )
      }
      a
  }

  val relationsShouldNotHaveCommonRoles = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      val spGroups = (sentences(s) ~> sentenceToTriplets).groupBy(x => x.getArgumentId(1)).filter(_._2.size > 1)
      spGroups.foreach {
        sp =>
          a = a and sp._2.toList._atmost(1)(t => TripletRelationClassifier on t is "Relation")
      }
      a
  }

  val boostTrajector = ConstrainedClassifier.constraint[Sentence] {
    s: Sentence =>
      (
        (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => IndicatorRoleClassifier on x is "Indicator" }
          ==>
          (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => TrajectorRoleClassifier on x is "Trajector" }
        )
  }
  //  val boostTrajectorByImage = ConstrainedClassifier.constraint[Sentence] {
  //    var a: FirstOrderConstraint = null
  //    s: Sentence =>
  //      a = new FirstOrderConstant(true)
  //      (sentences(s) ~> sentenceToPhrase).foreach {
  //        p =>
  //          val pairs = (phrases(p) ~> -segmentPhrasePairToPhrase).toList
  //
  //          a = a and
  //            (
  //              pairs._exists(pair => sim on pair is "true") ==>
  //                (TrajectorRoleClassifier on p is "Trajector") or (LandmarkRoleClassifier on p is "Landmark"))
  //      }
  //      a
  //  }

  val boostLandmark = ConstrainedClassifier.constraint[Sentence] {
    s: Sentence =>
      (
        (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => IndicatorRoleClassifier on x is "Indicator" }
          ==>
          (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => LandmarkRoleClassifier on x is "Landmark" }
        )
  }

  val boostTripletByGeneralType = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (
            (
              (TripletGeneralTypeClassifier on x) isNot "None"
              ) <==>
              (TripletRelationClassifier on x is "Relation")
            )
      }
      a
  }

  val boostTripletByImage = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          val tr = (triplets(x) ~> tripletToFirstArg ~> -segmentPhrasePairToPhrase).toList
          val lm = (triplets(x) ~> tripletToThirdArg ~> -segmentPhrasePairToPhrase).toList

          a = a and (
            tr._exists(t => (sim on t) is "true") and lm._exists(t => (sim on t) is "true") ==>
              (TripletRelationClassifier on x is "Relation")
            )
      }
      a
  }

  val boostGeneralByDirection = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (
            (TripletDirectionClassifier on x isNot "None") <==> (TripletGeneralTypeClassifier on x is "direction")
            )
      }
      a
  }

  val visualTripletClassifier = new VisualTripletClassifier()
  visualTripletClassifier.modelDir = s"models/mSpRL/VisualTriplets/"
  visualTripletClassifier.modelSuffix = "combined_perceptron_tuned"
  visualTripletClassifier.load()
  val boostTripletByImageTriplet = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (
            (visualTripletClassifier on x isNot "None") <==> (TripletGeneralTypeClassifier on x is "region")
            )
      }
      a
  }

  val boostGeneralByRegion = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (
            (TripletRegionClassifier on x isNot "None") <==> (TripletGeneralTypeClassifier on x is "region")
            )
      }
      a
  }

  val regionShouldHaveLandmark = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          val lmIsNull = (triplets(x) ~> tripletToThirdArg).head == dummyPhrase
          if (lmIsNull) {
            a = a and (TripletRegionClassifier on x is "None")
          }
      }
      a
  }


  val approveRelationByImage = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and ((imageSupportsSp on x is "true") ==>
            (TripletRelationClassifier on x is "Relation"))
      }
      a
  }

  val discardRelationByImage = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and ((imageSupportsSp on x is "false") ==>
            (TripletRelationClassifier on x is "None"))
      }
      a
  }


  val approveRelationByPreposition = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets ).foreach {
        x =>
          val vT = (triplets(x) ~> tripletToVisualTriplet).headOption
          if(vT.nonEmpty) {
            a = a and ((visualTripletClassifier on vT.get is "in_front_of") ==>
              (TripletRelationClassifier on x is "Relation"))
          }
      }
      a
  }

  val prepositionsConsistency = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets ~> tripletToVisualTriplet).foreach {
        x =>
          a = a and (
            (VisualTripletInFrontOfClassifier on x is "true") ==>
              (VisualTripletOnClassifier on x isNot "true") and
              (VisualTripletInClassifier on x isNot "true") and
              (VisualTripletAboveClassifier on x isNot "true")
            ) and
            ((VisualTripletOnClassifier on x is "true") ==>
            (VisualTripletInFrontOfClassifier on x isNot "true") and
            (VisualTripletInClassifier on x isNot "true") and
            (VisualTripletAboveClassifier on x isNot "true")
            ) and
            ((VisualTripletInClassifier on x is "true") ==>
            (VisualTripletOnClassifier on x isNot "true") and
            (VisualTripletInFrontOfClassifier on x isNot "true") and
            (VisualTripletAboveClassifier on x isNot "true")
            ) and
            ((VisualTripletAboveClassifier on x is "true") ==>
              (VisualTripletOnClassifier on x isNot "true") and
              (VisualTripletInClassifier on x isNot "true") and
              (VisualTripletInFrontOfClassifier on x isNot "true")
            )
      }
      a
  }

  //  val uniqueSegmentAssignment = ConstrainedClassifier.constraint[Sentence] {
  //    var a: FirstOrderConstraint = null
  //    s: Sentence =>
  //      a = new FirstOrderConstant(true)
  //      val segPhrases = (sentences(s) ~> sentenceToPhrase ~> -segmentPhrasePairToPhrase).toList
  //
  //      // The segments assigned to a phrase in a sentence should be at most 1
  //      segPhrases.groupBy(_.getArgumentId(0)).foreach {
  //        phraseSegments =>
  //          if (phraseSegments._2.size > 1)
  //            a = a and phraseSegments._2._atmost(1)(x => sim on x is "true")
  //      }
  //
  //      // The phrases assigned to a segment in a sentence should be at most 1
  //      segPhrases.groupBy(_.getArgumentId(1)).foreach {
  //        segmentPhrases =>
  //          if (segmentPhrases._2.size > 1)
  //            a = a and segmentPhrases._2._atmost(1)(x => sim on x is "true")
  //      }
  //
  //      a
  //  }

  val tripletConstraints = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence =>
      var a =
      //roleIntegrity(x) and
        roleShouldHaveRel(x) and
          boostTrajector(x) and
          boostLandmark(x) and
          boostTripletByGeneralType(x) and
          boostGeneralByDirection(x) and
          boostGeneralByRegion(x) and
          regionShouldHaveLandmark(x) and
          //discardRelationByImage(x) and
          //approveRelationByImage(x) //and
          //prepositionsConsistency(x) and
          approveRelationByPreposition(x)
      //relationsShouldNotHaveCommonRoles(x)
      //noDuplicates(x)

      //      if (mSpRLConfigurator.imageConstraints)
      //        a = a and
      //          boostTripletByImage(x) and
      //          //boostTrajectorByImage(x) and
      //          uniqueSegmentAssignment(x)
      a
  }

}
