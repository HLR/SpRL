package edu.tulane.cs.hetml.nlp.sprl

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.illinois.cs.cogcomp.saulexamples.nlp.SpatialRoleLabeling.Dictionaries
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors._
import edu.tulane.cs.hetml.vision._
import java.io._
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConversions._

/** Created by Taher on 2017-01-11.
  */
object MultiModalSpRLDataModel extends DataModel {

  val dummyPhrase = new Phrase()
  dummyPhrase.setText("[[None]]")
  dummyPhrase.setId("[[dummy]]")
  dummyPhrase.addPropertyValue("TRAJECTOR_id", dummyPhrase.getId)
  dummyPhrase.addPropertyValue("LANDMARK_id", dummyPhrase.getId)
  val undefined = "[undefined]"

  var useVectorAverages = false

  /*
  Nodes
   */
  val documents = node[Document]((d: Document) => d.getId)
  val sentences = node[Sentence]((s: Sentence) => s.getId)
  val phrases = node[Phrase]((p: Phrase) => p.getId)
  val tokens = node[Token]((t: Token) => t.getId)
  val pairs = node[Relation]((r: Relation) => r.getId)
  val triplets = node[Relation]((r: Relation) => r.getId)

  val images = node[Image]
  val segments = node[Segment]
  val segmentRelations = node[SegmentRelation]
  val segmentPhrasePairs = node[Relation]((r: Relation) => r.getId)

  /*
  Edges
   */
  val documentToSentence = edge(documents, sentences)
  documentToSentence.addSensor(documentToSentenceMatching _)

  val sentenceToPairs = edge(sentences, pairs)
  sentenceToPairs.addSensor(sentenceToRelationMatching _)

  val sentenceToPhrase = edge(sentences, phrases)
  sentenceToPhrase.addSensor(refinedSentenceToPhraseGenerating _)

  val phraseToToken = edge(phrases, tokens)
  phraseToToken.addSensor(phraseToTokenGenerating _)

  val pairToFirstArg = edge(pairs, phrases)
  pairToFirstArg.addSensor(relationToFirstArgumentMatching _)

  val pairToSecondArg = edge(pairs, phrases)
  pairToSecondArg.addSensor(relationToSecondArgumentMatching _)

  var sentenceToTriplets = edge(sentences, triplets)
  sentenceToTriplets.addSensor(sentenceToRelationMatching _)

  val tripletToFirstArg = edge(triplets, phrases)
  tripletToFirstArg.addSensor(relationToFirstArgumentMatching _)

  val tripletToSecondArg = edge(triplets, phrases)
  tripletToSecondArg.addSensor(relationToSecondArgumentMatching _)

  val tripletToThirdArg = edge(triplets, phrases)
  tripletToThirdArg.addSensor(relationToThirdArgumentMatching _)

  val documentToImage = edge(documents, images)
  documentToImage.addSensor(documentToImageMatching _)

  val imageToSegment = edge(images, segments)
  imageToSegment.addSensor(imageToSegmentMatching _)

  val segmentRelationsToSegments = edge(segmentRelations, segments)
  //segmentRelationsToSegments.addSensor(segmentRelationToSegmentMatching _)

  val segmentToSegmentPhrasePair = edge(segments, segmentPhrasePairs)
  //segmentToSegmentPhrasePair.addSensor(segmentToSegmentPhrasePairs _)

  val segmentPhrasePairToPhrase = edge(segmentPhrasePairs, phrases)
  //segmentPhrasePairToPhrase.addSensor(SegmentPhrasePairToPhraseMatching _)

  /*
  Properties
   */
  val trajectorRole = property(phrases) {
    x: Phrase =>
      if (x.containsProperty("TRAJECTOR_id") && x != dummyPhrase)
        "Trajector"
      else
        "None"
  }

  val landmarkRole = property(phrases) {
    x: Phrase =>
      if (x.containsProperty("LANDMARK_id") && x != dummyPhrase)
        "Landmark"
      else
        "None"
  }

  val indicatorRole = property(phrases) {
    x: Phrase =>
      if (x.containsProperty("SPATIALINDICATOR_id"))
        "Indicator"
      else
        "None"
  }

  val spatialRole = property(phrases) {
    x: Phrase =>
      if (x.containsProperty("TRAJECTOR_id") && x != dummyPhrase)
        "Trajector"
      else if (x.containsProperty("LANDMARK_id") && x != dummyPhrase)
        "Landmark"
      else if (x.containsProperty("SPATIALINDICATOR_id"))
        "Indicator"
      else
        "None"
  }

  val wordForm = property(phrases, cache = true) {
    x: Phrase =>
      if (x != dummyPhrase) (phrases(x) ~> phraseToToken).toList.sortBy(_.getStart)
        .map(t => t.getText.toLowerCase).mkString("|") else "None"
  }

  val lemma = property(phrases, cache = true) {
    x: Phrase =>
      if (x != dummyPhrase) (phrases(x) ~> phraseToToken).toList.sortBy(_.getStart)
        .map(t => getLemma(t).mkString).mkString("|") else "None"
  }

  val pos = property(phrases, cache = true) {
    x: Phrase =>
      if (x != dummyPhrase) (phrases(x) ~> phraseToToken).toList.sortBy(_.getStart)
        .map(t => getPos(t).mkString).mkString("|") else "None"
  }

  val headWordFrom = property(phrases, cache = true) {
    x: Phrase => if (x != dummyPhrase) getHeadword(x).getText.toLowerCase else "None"
  }

  val headWordPos = property(phrases, cache = true) {
    x: Phrase => if (x != dummyPhrase) getPos(getHeadword(x)).mkString else "None"
  }

  val headWordLemma = property(phrases, cache = true) {
    x: Phrase => if (x != dummyPhrase) getLemma(getHeadword(x)).mkString else "None"
  }

  val phrasePos = property(phrases, cache = true) {
    x: Phrase => if (x != dummyPhrase) getPhrasePos(x) else "None"
  }

  val semanticRole = property(phrases) {
    x: Phrase => "" //getSemanticRole(x)
  }

  val dependencyRelation = property(phrases, cache = true) {
    x: Phrase =>
      if (x != dummyPhrase) (phrases(x) ~> phraseToToken).toList.sortBy(_.getStart)
        .map(t => getDependencyRelation(t)).mkString("|") else "None"
  }

  val headDependencyRelation = property(phrases, cache = true) {
    x: Phrase => if (x != dummyPhrase) getDependencyRelation(getHeadword(x)) else "None"
  }

  val subCategorization = property(phrases, cache = true) {
    x: Phrase =>
      if (x != dummyPhrase) (phrases(x) ~> phraseToToken).toList.sortBy(_.getStart)
        .map(t => getSubCategorization(t)).mkString("|") else "None"
  }

  val headSubCategorization = property(phrases, cache = true) {
    x: Phrase => if (x != dummyPhrase) getSubCategorization(getHeadword(x)) else "None"
  }

  val headSpatialContext = property(phrases, cache = true) {
    x: Phrase =>
      val head = if (x == dummyPhrase) null else getHeadword(x)
      if (x == dummyPhrase)
        "None"
      else if (!Dictionaries.isPreposition(head.getText))
        "0"
      else if (getWindow(head, 0, 5).count(w => Dictionaries.isPreposition(w)) > 1)
        "1"
      else
        "2"
  }

  val spatialContext = property(phrases, cache = true) {
    x: Phrase =>
      val tokens = if (x == dummyPhrase) null else phrases(x) ~> phraseToToken
      if (x == dummyPhrase)
        "None"
      else if (tokens.forall(t => !Dictionaries.isPreposition(t.getText)))
        "0"
      else if (tokens.exists(t => getWindow(t, 0, 5).count(w => Dictionaries.isPreposition(w)) > 1))
        "1"
      else
        "2"
  }

  val headVector = property(phrases, cache = true, ordered = true) {
    x: Phrase => if (x != dummyPhrase) getVector(getHeadword(x).getText.toLowerCase) else getVector(null)
  }

  val isImageConceptExactMatch = property(phrases, cache = true) {
    p: Phrase =>
      if (p != dummyPhrase) {
        getSegmentConcepts(p)
          .exists(x => p.getText.toLowerCase.contains(x.toLowerCase.trim)).toString
      } else {
        ""
      }
  }

  val nearestSegmentConceptToHeadVector = property(phrases, ordered = true, cache = true) {
    p: Phrase =>
      if (p != dummyPhrase) {
        val head = getHeadword(p)
        val concepts = getSegmentConcepts(p).map(x => (x, getSimilarity(head.getText.toLowerCase, x)))
        val (nearest, _) = if (concepts.isEmpty) ("", 0) else concepts.maxBy(x => x._2)
        getVector(nearest)
      } else {
        getVector(null)
      }
  }

  val nearestSegmentConceptToPhraseVector = property(phrases, ordered = true, cache = true) {
    p: Phrase =>
      if (p != dummyPhrase) {
        //val head = getHeadword(p)
        val tokens = getTokens(p)
        val concepts = getSegmentConcepts(p).flatMap(x => tokens.map(t => (x, getSimilarity(t.getText.toLowerCase, x))))
        val (nearest, _) = if (concepts.isEmpty) ("", 0) else concepts.maxBy(x => x._2)
        getVector(nearest)
      } else {
        getVector(null)
      }
  }

  val isTrajectorRelation = property(pairs, cache = true) {
    x: Relation =>
      x.getProperty("RelationType") match {
        case "TR-SP" => "TR-SP"
        case _ => "None"
      }
  }

  val isLandmarkRelation = property(pairs, cache = true) {
    x: Relation =>
      x.getProperty("RelationType") match {
        case "LM-SP" => "LM-SP"
        case _ => "None"
      }
  }

  val isTrajectorCandidate = property(pairs) {
    r: Relation => getPairArguments(r)._1.containsProperty("TR-Candidate")
  }

  val isLandmarkCandidate = property(pairs) {
    r: Relation => getPairArguments(r)._1.containsProperty("LM-Candidate")
  }

  val isIndicatorCandidate = property(pairs) {
    r: Relation => getPairArguments(r)._1.containsProperty("SP-Candidate")
  }

  val pairWordForm = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      wordForm(first) + "::" + wordForm(second)
  }

  val pairHeadWordForm = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      headWordFrom(first) + "::" + headWordFrom(second)
  }

  val relationLemma = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      lemma(first) + "::" + lemma(second)
  }

  val relationHeadWordLemma = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      headWordLemma(first) + "::" + headWordLemma(second)
  }

  val pairPos = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      pos(first) + "::" + pos(second)
  }

  val pairHeadWordPos = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      headWordPos(first) + "::" + headWordPos(second)
  }

  val pairPhrasePos = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      phrasePos(first) + "::" + phrasePos(second)
  }

  val pairSemanticRole = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      semanticRole(first) + "::" + semanticRole(second)
  }

  val pairDependencyRelation = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      dependencyRelation(first) + "::" + dependencyRelation(second)
  }

  val relationHeadDependencyRelation = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      headDependencyRelation(first) + "::" + headDependencyRelation(second)
  }

  val pairSubCategorization = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      subCategorization(first) + "::" + subCategorization(second)
  }

  val relationHeadSubCategorization = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      headSubCategorization(first) + "::" + headSubCategorization(second)
  }

  val relationSpatialContext = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      spatialContext(first) + "::" + spatialContext(second)
  }

  val pairHeadSpatialContext = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      headSpatialContext(first) + "::" + headSpatialContext(second)
  }

  val pairTokensVector = property(pairs, ordered = true, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      headVector(first) ++ headVector(second)
  }

  val pairNearestSegmentConceptToHeadVector = property(pairs, ordered = true, cache = true) {
    r: Relation =>
      val (first, _) = getPairArguments(r)
      nearestSegmentConceptToHeadVector(first)
  }

  val pairNearestSegmentConceptToPhraseVector = property(pairs, ordered = true, cache = true) {
    r: Relation =>
      val (first, _) = getPairArguments(r)
      nearestSegmentConceptToPhraseVector(first)
  }

  val pairIsImageConcept = property(pairs, cache = true) {
    r: Relation =>
      val (first, _) = getPairArguments(r)
      isImageConceptExactMatch(first)
  }

  val before = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      if (first == dummyPhrase)
        ""
      else
        isBefore(first, second).toString
  }

  val distance = property(pairs, cache = true) {
    r: Relation =>
      val (first, second) = getPairArguments(r)
      if (first == dummyPhrase)
        -1
      else
        getTokenDistance(first, second)
  }

  val tripletIsRelation = property(triplets, cache = true) {
    x: Relation =>
      x.getProperty("Relation") match {
        case "true" => "Relation"
        case _ => "None"
      }
  }

  val tripletIsTr = property(triplets, cache = true) {
    x: Relation =>
      x.getProperty("Relation") match {
        case "true" =>
          val (first, second, third) = getTripletArguments(x)
          trajectorRole(first)
        case _ => "None"
      }
  }

  val tripletIsLm = property(triplets, cache = true) {
    x: Relation =>
      x.getProperty("Relation") match {
        case "true" =>
          val (first, second, third) = getTripletArguments(x)
          landmarkRole(third)
        case _ => "None"
      }
  }

  val tripletGeneralType = property(triplets) {
    r: Relation => if (r.containsProperty("GeneralType")) r.getProperty("GeneralType") else "None"
  }

  val tripletSpecificType = property(triplets) {
    r: Relation => if (r.containsProperty("SpecificType")) r.getProperty("SpecificType") else "None"
  }
  val rcc8Values = List("PO", "TPP", "EC", "DC", "EQ")
  val tripletRegion = property(triplets) {
    r: Relation =>
      if (r.containsProperty("RCC8") && rcc8Values.exists(x => r.getProperty("RCC8").toUpperCase().contains(x)))
        rcc8Values.find(x => r.getProperty("RCC8").toUpperCase().contains(x)).get
      else "None"
  }

  val directionValues = List("above", "behind", "below", "front", "left", "right")
  val tripletDirection = property(triplets) {
    r: Relation =>
      if (r.containsProperty("RCC8") && directionValues.exists(x => r.getProperty("RCC8").toLowerCase().contains(x)))
        directionValues.find(x => r.getProperty("RCC8").toLowerCase().contains(x)).get
      else "None"
  }

  val JF2_1 = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      first.getText + "::" + second.getText + "::" + third.getText
  }

  val JF2_2 = property(triplets, cache = true) {
    r: Relation =>
      val (_, _, third) = getTripletArguments(r)
      if (third != dummyPhrase) {
        (phrases(third) ~> phraseToToken)
          .exists(p => Dictionaries.spLexicon.exists(sp => p.getText.toLowerCase.contains(sp)))
      }
      else {
        false
      }
  }

  val JF2_3 = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      val (start, end) = getStartAndEndArgs(r)

      val toks = (triplets(r) ~> -sentenceToTriplets ~> sentenceToPhrase ~> phraseToToken)
        .filter(x => x.getStart >= start.getEnd && x.getEnd <= end.getStart)
        .toList.sortBy(_.getStart)

      val template = toks.foldLeft("")((str, token) => {

        if (first.contains(token) || second.contains(token) || third.contains(token))
          str
        else
          str + "::" + token.getText
      })
      template
  }

  val JF2_4 = property(triplets, cache = true) {
    r: Relation =>
      val (_, second, third) = getTripletArguments(r)
      second.getText + "::" + roleToSpDependencyPath(second, third)
  }

  val JF2_5 = property(triplets, cache = true) {
    r: Relation =>
      val (first, _, _) = getTripletArguments(r)
      first.getText
  }

  val JF2_6 = property(triplets, cache = true) {
    r: Relation =>
      val (_, second, third) = getTripletArguments(r)
      roleToSpDependencyPath(second, third)
  }

  val JF2_7 = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, _) = getTripletArguments(r)
      roleToSpDependencyPath(first, second) + "::" + second.getText
  }

  val JF2_8 = property(triplets, cache = true) {
    r: Relation =>
      val (_, _, third) = getTripletArguments(r)
      if (third == dummyPhrase)
        undefined
      else {
        getWordnetHypernyms(getHeadword(third))
      }
  }

  val JF2_9 = property(triplets, cache = true) {
    r: Relation =>
      val (first, _, _) = getTripletArguments(r)
      getWordnetHypernyms(getHeadword(first))
  }

  val JF2_10 = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      val (start, end) = getStartAndEndArgs(r)

      val toks = (triplets(r) ~> -sentenceToTriplets ~> sentenceToPhrase ~> phraseToToken)
        .filter(x => x.getStart >= start.getStart && x.getEnd <= end.getEnd)
        .toList.sortBy(_.getStart)

      val template = toks.foldLeft("")((str, token) => {

        if (first.contains(token)) {
          if (str.endsWith("[TR]"))
            str
          else
            str + "::[TR]"
        }
        else if (second.contains(token)) {
          if (str.endsWith("[SP]"))
            str
          else
            str + "::[SP]"
        }
        else if (third.contains(token)) {
          if (str.endsWith("[LM]"))
            str
          else
            str + "::[LM]"
        }
        else {
          str + "::" + token.getText
        }
      })
      template
  }

  val JF2_11 = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, _) = getTripletArguments(r)
      val tr = getHeadword(first)
      val sp = getHeadword(second)
      val preps = getDependencyRelationWith(tr, "PREP")
      val otherPrep = preps.filterNot(p => p._1 == sp.getStart && p._2 == sp.getEnd).headOption
      if (otherPrep.nonEmpty)
        otherPrep.head._3
      else
        ""
  }

  val JF2_13 = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      if (third == dummyPhrase) {
        val phrases = (triplets(r) ~> -sentenceToTriplets ~> sentenceToPhrase).filterNot(_ == first)
        phrases.exists(p => getPos(p).exists(x => x.startsWith("NN")))
      }
      else {
        false
      }
  }

  val JF2_14 = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      headWordLemma(first) + "::" + second.getText + "::" + headWordLemma(third)
  }

  val JF2_15 = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, _) = getTripletArguments(r)
      val tr = getHeadword(first)
      val sp = getHeadword(second)
      val preps = getDependencyRelationWith(tr, "POBJ")
      if (preps.nonEmpty) {
        preps.exists(p => p._1 != sp.getStart)
      }
      else
        false
  }

  val tripletHeadWordForm = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      headWordFrom(first) + "::" + headWordFrom(second) + "::" + headWordFrom(third)
  }

  val tripletHeadWordPos = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      headWordPos(first) + "::" + headWordPos(second) + "::" + headWordPos(third)
  }

  val tripletPhrasePos = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      phrasePos(first) + "::" + phrasePos(second) + "::" + phrasePos(third)
  }

  val tripletDependencyRelation = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      dependencyRelation(first) + "::" + dependencyRelation(second) + "::" + dependencyRelation(third)
  }

  val tripletTrVector = property(triplets, cache = true, ordered = true) {
    r: Relation =>
      val (first, _, _) = getTripletArguments(r)
      headVector(first)
  }

  val tripletSpVector = property(triplets, cache = true, ordered = true) {
    r: Relation =>
      val (_, second, _) = getTripletArguments(r)
      headVector(second)
  }

  val tripletLmVector = property(triplets, cache = true, ordered = true) {
    r: Relation =>
      val (_, _, third) = getTripletArguments(r)
      headVector(third)
  }

  val tripletLemma = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      lemma(first) + "::" + lemma(second) + "::" + lemma(third)
  }

  val tripletHeadWordLemma = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      headWordLemma(first) + "::" + headWordLemma(second) + "::" + headWordLemma(third)
  }

  val tripletHeadDependencyRelation = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      headDependencyRelation(first) + "::" + headDependencyRelation(second) + "::" + headDependencyRelation(third)
  }

  val tripletPos = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      pos(first) + "::" + pos(second) + "::" + pos(third)
  }

  val tripletSubCategorization = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      subCategorization(first) + "::" + subCategorization(second) + "::" + subCategorization(third)
  }

  val tripletHeadSubCategorization = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      headSubCategorization(first) + "::" + headSubCategorization(second) + "::" + headSubCategorization(third)
  }

  val tripletSpatialContext = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      spatialContext(first) + "::" + spatialContext(second) + "::" + spatialContext(third)
  }

  val tripletHeadSpatialContext = property(triplets, cache = true) {
    r: Relation =>
      val (first, second, third) = getTripletArguments(r)
      headSpatialContext(first) + "::" + headSpatialContext(second)
  }

  val imageLabel = property(images, cache = true) {
    x: Image => x.getLabel
  }

  val imageId = property(images, cache = true) {
    x: Image => x.getId
  }

  val segmentLabel = property(segments, cache = true) {
    x: Segment => x.getSegmentConcept
  }

  val segmentId = property(segments, cache = true) {
    x: Segment => x.getSegmentCode
  }

  val segmentFeatures = property(segments, cache = true) {
    x: Segment => x.getSegmentFeatures.split(" ").toList.map(_.toDouble)
  }

  ////////////////////////////////////////////////////////////////////
  /// Helper methods
  ////////////////////////////////////////////////////////////////////

  private def getStartAndEndArgs(r: Relation): (Phrase, Phrase) = {
    val (first, second, third) = getTripletArguments(r)

    val start = if (isBefore(first, second))
      if (third == dummyPhrase || isBefore(first, third)) first else third
    else if (third == dummyPhrase || isBefore(second, third)) second else third


    val end = if (isBefore(first, second))
      if (third != dummyPhrase && isBefore(second, third)) third else second
    else if (third != dummyPhrase && isBefore(first, third)) third else first


    (start, end)
  }

  private def getVector(w: String): List[Double] = {
    if (useVectorAverages) {
      getAverage(getGoogleWordVector(w), getClefWordVector(w))
    } else {
      getGoogleWordVector(w)
    }
  }

  def getSimilarity(w1: String, w2: String): Double = {
    if (useVectorAverages) {
      (getGoogleSimilarity(w1, w2) + getClefSimilarity(w1, w2)) / 2.0
    } else {
      getGoogleSimilarity(w1, w2)
    }
  }

  private def getTripletArguments(r: Relation): (Phrase, Phrase, Phrase) = {
    ((triplets(r) ~> tripletToFirstArg).head, (triplets(r) ~> tripletToSecondArg).head, (triplets(r) ~> tripletToThirdArg).head)
  }

  private def getPairArguments(r: Relation): (Phrase, Phrase) = {
    ((pairs(r) ~> pairToFirstArg).head, (pairs(r) ~> pairToSecondArg).head)
  }

  private def getSegmentConcepts(p: Phrase) = {
    ((phrases(p) ~> -sentenceToPhrase ~> -documentToSentence) ~> documentToImage ~> imageToSegment)
      .map(x =>
        if (!phraseConceptToWord.contains(x.getSegmentConcept))
          x.getSegmentConcept
        else
          phraseConceptToWord(x.getSegmentConcept))
  }

  private def roleToSpDependencyPath(first: Phrase, second: Phrase) = {
    if (first != dummyPhrase && second != dummyPhrase) {
      val f = getHeadword(first)
      val s = getHeadword(second)
      getDependencyPath(f, s)
    }
    else
      undefined
  }

}
