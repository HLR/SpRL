package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierDataModel._

object WordasClassifierSensors {

  def documentToSentenceMatching(d: Document, s: Sentence): Boolean = {
    d.getId == s.getDocument.getId
  }

  def imageToSegmentMatching(i: Image, s: Segment): Boolean = {
    i.getId == s.getAssociatedImageID
  }

  def documentToImageMatching(d: Document, i: Image): Boolean = {
    d.getId.split("_")(0)==i.getId
  }

  def relationToFirstArgumentMatching(r: Relation, s: Sentence): Boolean = {
    r.getArgumentId(0) == s.getId
  }

  def relationToSecondArgumentMatching(r: Relation, s: Segment): Boolean = {
    r.getArgumentId(1)==s.getUniqueId
  }

  def sentenceToSegmentSentencePairsMatching(sen: Sentence): List[Relation] = {

    val imgSegId = sen.getId.split("_")
    val segs = segments().filter(s=> s.getAssociatedImageID==imgSegId(0)).toList
    segs.map{
      seg=>
        val r = new Relation()

        val isRel = if (imgSegId(0)==seg.getAssociatedImageID && imgSegId(1)==seg.getSegmentId.toString) "1" else "0"
        r.setId(sen.getId + "__" + isRel + "__" + seg.getSegmentId)
        r.setArgumentId(0, sen.getId)
        r.setArgumentId(1, imgSegId(0) + "_" + imgSegId(1))
        r.setArgumentId(2, isRel)
        r
    }
  }
}