package edu.tulane.cs.hetml.nlp.sprl.Triplets

import edu.illinois.cs.cogcomp.lbjava.infer.{FirstOrderConstant, FirstOrderConstraint}
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.constraint.ConstraintTypeConversion._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.MultiModalSpRLTripletClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.Triplets.TripletSentenceLevelConstraints.imageSupportsSp
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator

import scala.collection.JavaConversions._

object TripletSentenceLevelConstraints {
  val imageSupportsSp = new ImageSupportsSpClassifier()
  val prepClassifiers = Map(
    //"in front of" -> PrepositionInFrontOfClassifier,
    "in" -> PrepositionInClassifier
    //"on" -> PrepositionOnClassifier,
    //"above" -> PrepositionAboveClassifier
    //"at" -> PrepositionAtClassifier,
    //"around" -> PrepositionAroundClassifier,
    //    "behind" -> PrepositionBehindClassifier,
    //    "between" -> PrepositionBetweenClassifier,
    //    "in between" -> PrepositionInBetweenClassifier,
    //    "leaning on" -> PrepositionLeaningOnClassifier,
    //    "next to" -> PrepositionNextToClassifier,
    //    "on each side" -> PrepositionOnEachSideClassifier,
    //    "over" -> PrepositionOverClassifier,
    //    "sitting around" -> PrepositionSittingAroundClassifier
  )

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

  val boostGeneralByDirection = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and
            ((TripletDirectionAboveClassifier on x is "true") ==> (TripletGeneralTypeClassifier on x is "direction")) and
            ((TripletDirectionBelowClassifier on x is "true") ==> (TripletGeneralTypeClassifier on x is "direction")) and
            ((TripletDirectionBehindClassifier on x is "true") ==> (TripletGeneralTypeClassifier on x is "direction")) and
            ((TripletDirectionFrontClassifier on x is "true") ==> (TripletGeneralTypeClassifier on x is "direction")) and
            ((TripletDirectionLeftClassifier on x is "true") ==> (TripletGeneralTypeClassifier on x is "direction")) and
            ((TripletDirectionRightClassifier on x is "true") ==> (TripletGeneralTypeClassifier on x is "direction"))
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
            ((TripletDirectionClassifier on x isNot "None") ==> (TripletGeneralTypeClassifier on x is "direction"))
      }
      a
  }

  val boostGeneralByRegion = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and
            ((TripletRegionTPPClassifier on x is "true") ==> (TripletGeneralTypeClassifier on x is "region")) and
            ((TripletRegionECClassifier on x is "true") ==> (TripletGeneralTypeClassifier on x is "region")) and
            ((TripletRegionEQClassifier on x is "true") ==> (TripletGeneralTypeClassifier on x is "region")) and
            ((TripletRegionPOClassifier on x is "true") ==> (TripletGeneralTypeClassifier on x is "region")) and
            ((TripletRegionDCClassifier on x is "true") ==> (TripletGeneralTypeClassifier on x is "region"))
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

  val matchVisualAndTextRels = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          val vT = (triplets(x) ~> tripletToVisualTriplet).headOption
          if (vT.nonEmpty) {
            val sp = (triplets(x) ~> tripletToSp).head.getText.toLowerCase()
            if (prepClassifiers.contains(sp)) {
              a = a and ((prepClassifiers(sp) on vT.get is "true") ==> (TripletRelationClassifier on x is "true"))
            }
          }
      }
      a
  }

  val prepositionConsistency = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)

      val preps = (sentences(s) ~> sentenceToTriplets).toList
        .map(x => (triplets(x) ~> tripletToSp).head.getText.toLowerCase().trim)
        .filter(x => prepClassifiers.contains(x))


      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          val vt = (triplets(x) ~> tripletToVisualTriplet).headOption
          if (vt.nonEmpty) {
            var othersFalse: FirstOrderConstraint = new FirstOrderConstant(true)
            val sp = (triplets(x) ~> tripletToSp).head.getText.toLowerCase().trim
            if (preps.contains(sp)) {
              val others = preps.filter(p => p != sp)
              if (others.nonEmpty) {
                others.foreach(prep => othersFalse = othersFalse and ((prepClassifiers(prep) on vt.get) is "false"))

                a = a and (((prepClassifiers(sp) on vt.get) is "true") ==> othersFalse)
              }
            }
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
      var a =
      //roleIntegrity(x) and
        roleShouldHaveRel(x) and
          boostTrajector(x) and
          boostLandmark(x) and
          boostTripletByGeneralType(x) and
          boostGeneralByDirectionMulti(x) and
          boostGeneralByRegionMulti(x) and
          //prepositionConsistency(x) and
          //matchVisualAndTextRels(x)
          discardRelationByImage(x) and
          approveRelationByImage(x) //and
      //prepositionsConsistency(x) and
      //          approveRelationByMultiPreposition(x) and
      //          agreePrepositionClassifer(x)
      //relationsShouldNotHaveCommonRoles(x)
      //noDuplicates(x)
      //boostTripletByImageTriplet(x)

      //      if (mSpRLConfigurator.imageConstraints)
      //        a = a and

      //          //boostTrajectorByImage(x) and
      //          uniqueSegmentAssignment(x)
      a
  }

}
