package edu.tulane.cs.hetml.nlp.sprl.Triplets

import edu.illinois.cs.cogcomp.lbjava.infer.{FirstOrderConstant, FirstOrderConstraint}
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.constraint.ConstraintTypeConversion._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._

object TripletSentenceLevelConstraints {

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

  val boostTrajector = ConstrainedClassifier.constraint[Sentence] {
    s: Sentence =>
      (
        (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => IndicatorRoleClassifier on x is "Indicator" }
          ==>
          (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => TrajectorRoleClassifier on x is "Trajector" }
        )
  }
  val sim = new SegmentPhraseSimilarityClassifier()
  val boostTrajectorByImage = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToPhrase).foreach {
        p =>
          val pairs = (phrases(p) ~> -segmentPhrasePairToPhrase).toList

          a = a and
            (
              (TrajectorRoleClassifier on p is "Trajector") ==> pairs._exists(pair => sim on pair is "true")
            )
      }
      a
  }

  val boostLandmark = ConstrainedClassifier.constraint[Sentence] {
    s: Sentence =>
      (
        (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => IndicatorRoleClassifier on x is "Indicator" }
          ==>
          (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => LandmarkRoleClassifier on x is "Landmark" }
        )
  }

  val boostTripletByRoles = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      (
        (sentences(s) ~> sentenceToPhrase).toList._exists { p: Phrase => IndicatorRoleClassifier on p is "Indicator" } or
          (sentences(s) ~> sentenceToPhrase).toList._exists { p: Phrase => LandmarkRoleClassifier on p is "Landmark" } or
          (sentences(s) ~> sentenceToPhrase).toList._exists { p: Phrase => TrajectorRoleClassifier on p is "Trajector" }

        ) ==>
        (sentences(s) ~> sentenceToTriplets).toList._exists { r: Relation => TripletRelationClassifier on r is "Relation" }
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
              ) ==>
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
            tr._exists(t => (sim on t) is "true")
              ==>
              (TripletRelationClassifier on x is "Relation")
            )
      }
      a
  }

  val boostGeneralType = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (
            (
              (TripletRelationClassifier on x) is "Relation") ==>
              (TripletGeneralTypeClassifier on x isNot "None")
            )
      }
      a
  }

  val boostDirection = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (
            (
              (TripletGeneralTypeClassifier on x) is "direction") <==>
              (TripletDirectionClassifier on x isNot "None")
            )
      }
      a
  }

  val boostRegion = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (
            (
              (TripletGeneralTypeClassifier on x) is "region") <==>
              (TripletRegionClassifier on x isNot "None")
            )
      }
      a
  }

  val roleConstraints = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence => boostTrajector(x) and boostLandmark(x) and roleIntegrity(x) //and boostTrajectorByImage(x)
  }

  val tripletConstraints = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence => boostTripletByRoles(x) and boostTripletByGeneralType(x) //and boostTripletByImage(x)
  }

  val generalConstraints = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence => boostTripletByRoles(x) and boostGeneralType(x)
  }

  val directionConstraints = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence => generalConstraints(x) and boostDirection(x) and boostRegion(x)
  }

  val regionConstraints = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence => generalConstraints(x) and boostRegion(x)
  }

}
