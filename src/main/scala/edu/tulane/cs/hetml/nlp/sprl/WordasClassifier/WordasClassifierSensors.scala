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

  def sentenceToSegmentSentencePairsMatching(sen: Sentence): List[Relation] = {
    val ID = sen.getId.split("_")
    val segs = segments().filter(s=> s.getAssociatedImageID==ID(0)).toList
      //(sentences(sen) ~> -documentToSentence ~> documentToImage ~> imageToSegment).toList
    val len = if (segs.size > 5) 6 else segs.size
    segs.take(len).map{
      seg=>
        val r = new Relation()

        val isRel = if (ID(0)==seg.getAssociatedImageID && ID(1)==seg.getSegmentId.toString) "1" else "0"
        r.setId(sen.getId + "__" + isRel)
        r.setArgumentId(0, sen.getId)
        r.setArgumentId(1, isRel)
        r.setProperty("isRelation", isRel)
        r
    }
  }
}