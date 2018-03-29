package edu.tulane.cs.hetml.nlp.sprl.Anaphora

import edu.illinois.cs.cogcomp.lbjava.infer.{FirstOrderConstant, FirstOrderConstraint}
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.constraint.ConstraintTypeConversion._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.tripletConfigurator._
import edu.tulane.cs.hetml.nlp.sprl.Anaphora.MultiModalSpRLTripletClassifiers._

object TripletSentenceLevelConstraints {
  val wordAsClassifierHelper = TripletSensors.alignmentHelper

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

  lazy val sentWordSegs = wordSegments().groupBy(_.getPhrase.getSentence.getId)
  val alignmentConsistency = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      if(sentWordSegs.contains(s.getId)) {
        val pairs = sentWordSegs(s.getId)
        //wordSegments().filter(x => x.getPhrase.getSentence.getId == s.getId).toList
        val perSeg = pairs.groupBy(_.getSegment)
        val perPhrase = pairs.groupBy(_.getPhrase)

        perPhrase.foreach {
          z =>
            val w = z._2.head.getWord
            val c = wordAsClassifierHelper.trainedWordClassifier(w)
            z._2.foreach {
              x =>
                var b: FirstOrderConstraint = new FirstOrderConstant(true)
                z._2.filter(y => y != x).foreach {
                  y =>
                    b = b and (c on y is "false")
                }
                a = a and ((c on x is "true") ==> b)
            }
        }
        perSeg.foreach {
          z =>
            z._2.foreach {
              x =>
                val w = x.getWord
                val c = wordAsClassifierHelper.trainedWordClassifier(w)
                var b: FirstOrderConstraint = new FirstOrderConstant(true)
                z._2.filter(y => y != x).foreach {
                  y =>
                    b = b and (c on y is "false")
                }
                a = a and ((c on x is "true") ==> b)
            }
        }
      }
      a
  }

  val approveRelationByCoReference = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          if(x.getProperty("ImplicitLandmark")=="true") {
            val coRefRels = triplets(x) ~> tripletsToCoRefTriplet
            coRefRels.foreach(r => {
              val x_2 = r.getArgument(2)
              x.setArgument(2, r.getArgument(2))
              val Rc = TripletCoReferenceRelationClassifier on x is "true"
              x.setArgument(2, x_2)
              val Rr = TripletRelationClassifier on x is "true"
              a = a and (Rc ==> Rr)
            })
          }
      }
      a
  }

  val tripletConstraints = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence =>
      var a: FirstOrderConstraint = null
      a =
          roleShouldHaveRel(x) and
          boostTrajector(x) and
          boostTripletByGeneralType(x) and
          boostGeneralByDirectionMulti(x) and
          boostGeneralByRegionMulti(x)
      if(useCoReference) {
        a = a and approveRelationByCoReference(x)
      }
      a
  }

}
