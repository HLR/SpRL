package edu.tulane.cs.hetml.nlp.sprl.Triplets

import edu.illinois.cs.cogcomp.lbjava.infer.{FirstOrderConstant, FirstOrderConstraint}
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.constraint.ConstraintTypeConversion._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._

import scala.collection.JavaConversions._

object TripletSentenceLevelConstraints {
  val imageSupportsSp = new ImageSupportsSpClassifier()

  val roleShouldHaveRel = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      val sentTriplets = (sentences(s) ~> sentenceToTriplets).toList
      sentTriplets.foreach {
        x =>
          val trRel = sentTriplets.filter(r => r.getArgumentId(0) == x.getArgumentId(0))
            ._exists(x => (TripletRelationClassifier on x) is "true")

          val lmRel = sentTriplets.filter(r => r.getArgumentId(2) == x.getArgumentId(2))
            ._exists(x => (TripletRelationClassifier on x) is "true")

          val tr = TrajectorRoleClassifier on (triplets(x) ~> tripletToTr).head is "true"
          val lm = LandmarkRoleClassifier on (triplets(x) ~> tripletToLm).head is "true"

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
              (TripletRelationClassifier on x) is "true") ==>
              (
                (TrajectorRoleClassifier on (triplets(x) ~> tripletToTr).head is "true") and
                  (IndicatorRoleClassifier on (triplets(x) ~> tripletToSp).head is "true") and
                  (LandmarkRoleClassifier on (triplets(x) ~> tripletToLm).head is "true")
                )
            )
      }
      a
  }

  val relationsShouldNotHaveCommonRoles = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      val sentTriplets = (sentences(s) ~> sentenceToTriplets).toList
      sentTriplets.foreach {
        r =>
          val tr = (triplets(r) ~> tripletToTr).head
          val sp = (triplets(r) ~> tripletToSp).head
          val lm = (triplets(r) ~> tripletToLm).head
          var othersFalse: FirstOrderConstraint = new FirstOrderConstant(true)
          val others = sentTriplets.filter(x => x != r && (triplets(x) ~> tripletToSp).head == sp)
          if (others.nonEmpty) {
            others.foreach(x => othersFalse = othersFalse and ((TripletRelationClassifier on x) is "false"))
            a = a and (((TripletRelationClassifier on r) is "true") ==> othersFalse)
          }
      }
      a
  }

  val boostTrajector = ConstrainedClassifier.constraint[Sentence] {
    s: Sentence =>
      (
        (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => IndicatorRoleClassifier on x is "true" }
          ==>
          (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => TrajectorRoleClassifier on x is "true" }
        )
  }

  val boostLandmark = ConstrainedClassifier.constraint[Sentence] {
    s: Sentence =>
      (
        (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => IndicatorRoleClassifier on x is "true" }
          ==>
          (sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => LandmarkRoleClassifier on x is "true" }
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
              (TripletRelationClassifier on x is "true")
            )
      }
      a
  }


  val boostGeneralByDirectionMulti = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and
            ((TripletDirectionClassifier on x isNot "None") <==> (TripletGeneralTypeClassifier on x is "direction"))
      }
      a
  }


  val boostGeneralByRegionMulti = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and
            ((TripletRegionClassifier on x isNot "None") <==> (TripletGeneralTypeClassifier on x is "region"))
      }
      a
  }

  val regionShouldHaveLandmark = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          val lmIsNull = (triplets(x) ~> tripletToLm).head == dummyPhrase
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
            (TripletRelationClassifier on x is "true"))
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
            (TripletRelationClassifier on x is "false"))
      }
      a
  }

  val tripletConstraints = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence =>
      var a: FirstOrderConstraint = null
      a =
        roleShouldHaveRel(x) and
          boostTrajector(x) and
          boostLandmark(x) and
          boostTripletByGeneralType(x) and
          boostGeneralByDirectionMulti(x) and
          boostGeneralByRegionMulti(x)

      if (tripletConfigurator.usePrepositions) {
        a = a and
          discardRelationByImage(x) and
          approveRelationByImage(x) //and
      }
      a
  }

}
