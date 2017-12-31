package edu.tulane.cs.hetml.nlp.sprl.Triplets

import edu.tulane.cs.hetml.nlp.BaseTypes.{Relation, Sentence}
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.getHeadword
import edu.tulane.cs.hetml.nlp.sprl.Helpers.WordClassifierHelper
import MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors.matchingCandidates
import edu.tulane.cs.hetml.vision.{ImageTriplet, Segment}

object TripletSensors {
  lazy val alignmentHelper = new WordClassifierHelper()
  if (tripletConfigurator.alignmentMethod == "classifier")
    alignmentHelper.loadAllTrainedClassifiers(true)

  def TripletToVisualTripletGenerating(r: Relation): List[ImageTriplet] = {
    val (first, second, third) = getTripletArguments(r)
    val trSegId = first.getPropertyFirstValue("bestAlignment")
    val lmSegId = third.getPropertyFirstValue("bestAlignment")
    if (trSegId != null && lmSegId != null) {

      val trSeg = (phrases(first) ~> -segmentPhrasePairToPhrase ~> -segmentToSegmentPhrasePair)
        .find(_.getSegmentId.toString.equalsIgnoreCase(trSegId))
      val lmSeg = (phrases(third) ~> -segmentPhrasePairToPhrase ~> -segmentToSegmentPhrasePair)
        .find(_.getSegmentId.toString.equalsIgnoreCase(lmSegId))

      if (trSeg.nonEmpty && lmSeg.nonEmpty) {

        val imId = trSeg.get.getAssociatedImageID
        val imageRel = imageSegmentsDic(imId).filter(x =>
          x.getFirstSegId.toString == trSegId && x.getSecondSegId.toString == lmSegId)

        imageRel.foreach {
          x =>
            x.setTrajector(first.getText.toLowerCase())
            x.setLandmark(third.getText.toLowerCase())
            if (tripletIsRelation(r) == "Relation")
              x.setSp(second.getText.toLowerCase.trim.replaceAll(" ", "_"))

        }
        imageRel.toList
      }
      else
        List()
    }
    else {
      List()
    }
  }

  def segmentToSegmentPhrasePairs(s: Segment): List[Relation] = {
    val image = images().filter(i => i.getId == s.getAssociatedImageID)
    val phrases = (images(image) ~> -documentToImage ~> documentToSentence ~> sentenceToPhrase)
      .filter(p => p != dummyPhrase && matchingCandidates.exists(x => headWordPos(p).toUpperCase().contains(x)))
      .toList

    phrases.map {
      p =>
        val r = new Relation()
        r.setId(p.getId + "__" + s.getSegmentId)
        r.setArgumentId(0, p.getId)
        r.setArgumentId(1, s.getSegmentId.toString)
        tripletConfigurator.alignmentMethod match {
          case "gold" =>
            if (p.getPropertyValues("goldAlignment").contains(s.getSegmentId.toString)) {
              r.setProperty("similarity", "1")
            }
            else {
              r.setProperty("similarity", "0")
            }

          case "w2v" =>
            val head = getHeadword(p)
            val sim = s.getSegmentConcept.split('-').map(x => getSimilarity(x, head.getText)).max
            r.setProperty("similarity", sim.toString)

          case "classifier" =>
            val lemma = headWordLemma(p)
            val sim = alignmentHelper.getPhraseHeadwordSegmentScore(lemma, s)
            r.setProperty("similarity", sim.toString)
        }
        r
    }
  }

  def SentenceToVisualTripletMatching(s: Sentence, vt: ImageTriplet): Boolean = {
    val triplet = (visualTriplets(vt) ~> -tripletToVisualTriplet).headOption
    if (triplet.nonEmpty)
      triplet.get.getParent.getId == s.getId
    else
      false
  }

}
