package edu.tulane.cs.hetml.nlp.sprl

import edu.illinois.cs.cogcomp.lbjava.infer.{FirstOrderConstant, FirstOrderConstraint}
import edu.illinois.cs.cogcomp.saul.classifier.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.constraint.ConstraintTypeConversion._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLClassifiers._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel.{sentenceToPhrase, _}
import edu.tulane.cs.hetml.nlp.BaseTypes._

/** Created by parisakordjamshidi on 2/9/17.
  */
object SentenceLevelConstraints {

  val integrityTR = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToPairs).foreach {
        x =>
          a = a and (((TrajectorPairClassifier on x) is "TR-SP") ==>
            ((TrajectorRoleClassifier on (pairs(x) ~> pairToFirstArg).head is "Trajector") and
            (IndicatorRoleClassifier on (pairs(x) ~> pairToSecondArg).head is "Indicator")))
      }
      a
  }

  val integrityLM = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToPairs).foreach {
        x =>
          a = a and (((LandmarkPairClassifier on x) is "LM-SP") ==>
            ((LandmarkRoleClassifier on (pairs(x) ~> pairToFirstArg).head is "Landmark") and
            (IndicatorRoleClassifier on (pairs(x) ~> pairToSecondArg).head is "Indicator")))
      }
      a
  }

  val multiLabelPair = ConstrainedClassifier.constraint[Sentence] {
    //a pair w1-w2 can be only tr-sp or lm-sp not both at the same time.
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToPairs).foreach {
        x: Relation =>
          a = a and ((((LandmarkPairClassifier on x) is "LM-SP") ==> ((TrajectorPairClassifier on x) isNot "TR-SP")) and
            (((TrajectorPairClassifier on x) is "TR-SP") ==> ((LandmarkPairClassifier on x) isNot "LM-SP")))
      }
      a
  }

  val boostIndicator = ConstrainedClassifier.constraint[Sentence] {
    //if there exists a trajector or a landmark in the sentence then there should exist an indicator in the sentence too.
    s: Sentence =>
      (((sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => TrajectorRoleClassifier on x is "Trajector" }) or
        ((sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => LandmarkRoleClassifier on x is "Landmark" })) ==>
        ((sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => IndicatorRoleClassifier on x is "Indicator" })
  }

  val boostPairs = ConstrainedClassifier.constraint[Sentence] {
    //if there is an indicator in the sentence then there should be a relation in the sentence, though the roles can be null.
    s: Sentence =>
      ((sentences(s) ~> sentenceToPhrase).toList._exists { x: Phrase => IndicatorRoleClassifier on x is "Indicator" }) ==>
        (((sentences(s) ~> sentenceToPairs).toList._exists { x: Relation => TrajectorPairClassifier on x is "TR-SP" }) and
        ((sentences(s) ~> sentenceToPairs).toList._exists { x: Relation => LandmarkPairClassifier on x is "LM-SP" }))
  }

  val boostTripletsTrSpLm = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (((TripletRelationClassifier on x) is "Relation") ==>
            (TrajectorRoleClassifier on (triplets(x) ~> tripletToFirstArg).head is "Trajector") and
            (IndicatorRoleClassifier on (triplets(x) ~> tripletToSecondArg).head is "Indicator") and
            (LandmarkRoleClassifier on (triplets(x) ~> tripletToThirdArg).head is "Landmark")
            )
      }
      a
  }

  val boostTripletTrSp = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (((TripletRelationClassifier on x) is "Relation") ==>
            ((TrajectorRoleClassifier on (triplets(x) ~> tripletToFirstArg).head is "Trajector") and
              (IndicatorRoleClassifier on (triplets(x) ~> tripletToSecondArg).head is "Indicator")))
      }
      a
  }

  val boostTripletLmSp = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (((TripletRelationClassifier on x) is "Relation") ==>
            ((TrajectorRoleClassifier on (triplets(x) ~> tripletToThirdArg).head is "Landmark") and
              (IndicatorRoleClassifier on (triplets(x) ~> tripletToSecondArg).head is "Indicator")))
      }
      a
  }

  val boostTripletRelation = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (((TripletRelationClassifier on x) is "Relation") ==>
            (TripletGeneralTypeClassifier on x isNot "None") or
            (TripletSpecificTypeClassifier on x isNot "None") or
            (TripletRCC8Classifier on x isNot "None") or
            (TripletFoRClassifier on x isNot "None")
            )
      }
      a
  }

  val boostTripletGeneral = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (((TripletGeneralTypeClassifier on x) isNot "None") ==>
            (TripletRelationClassifier on x is "Relation") or
            (TripletSpecificTypeClassifier on x isNot "None") or
            (TripletRCC8Classifier on x isNot "None") or
            (TripletFoRClassifier on x isNot "None")
            )
      }
      a
  }

  val boostTripletSpecific = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (((TripletSpecificTypeClassifier on x) isNot "None") ==>
            (TripletRelationClassifier on x is "Relation") or
            (TripletGeneralTypeClassifier on x isNot "None") or
            (TripletRCC8Classifier on x isNot "None") or
            (TripletFoRClassifier on x isNot "None")
            )
      }
      a
  }

  val boostTripletRcc = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (((TripletRCC8Classifier on x) isNot "None") ==>
            (TripletRelationClassifier on x is "Relation") or
            (TripletGeneralTypeClassifier on x isNot "None") or
            (TripletSpecificTypeClassifier on x isNot "None") or
            (TripletFoRClassifier on x isNot "None")
            )
      }
      a
  }

  val boostTripletFor = ConstrainedClassifier.constraint[Sentence] {
    var a: FirstOrderConstraint = null
    s: Sentence =>
      a = new FirstOrderConstant(true)
      (sentences(s) ~> sentenceToTriplets).foreach {
        x =>
          a = a and (((TripletFoRClassifier on x) isNot "None") ==>
            (TripletRelationClassifier on x is "Relation") or
            (TripletGeneralTypeClassifier on x isNot "None") or
            (TripletSpecificTypeClassifier on x isNot "None") or
            (TripletRCC8Classifier on x isNot "None")
            )
      }
      a
  }

  val allConstraints = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence => integrityLM(x) and integrityTR(x) and multiLabelPair(x) and boostIndicator(x) and boostPairs(x)
  }

  val relationConstraintTriplets = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence => boostTripletRelation(x) and boostTripletLmSp(x)
  }

  val generalConstraintTriplets = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence => boostTripletGeneral(x) and boostTripletLmSp(x)
  }

  val specificConstraintTriplets = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence => boostTripletSpecific(x) and boostTripletLmSp(x)
  }

  val rccConstraintTriplets = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence => boostTripletRcc(x) and boostTripletLmSp(x)
  }

  val forConstraintTriplets = ConstrainedClassifier.constraint[Sentence] {

    x: Sentence => boostTripletFor(x) and boostTripletLmSp(x)
  }
}
