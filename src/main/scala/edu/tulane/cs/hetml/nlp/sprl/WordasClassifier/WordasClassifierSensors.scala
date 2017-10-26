package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierDataModel._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._

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
    if(isTrain) {
      val posSeg = segments().filter(s=> s.getUniqueId==sen.getId).toList
      val negSegs = segments().filter(s=> s.getAssociatedImageID==sen.getId.split("_")(0) &&
        s.getSegmentId.toString!=sen.getId.split("_")(1)).toList
      val len = if(negSegs.size > 5) 5 else negSegs.size
      val scoreVector = getExpressionSegmentScore(sen, posSeg.head, negSegs.take(len)) // Take only defined negative examples
      val posExps = posSeg.map {
        seg=>
          val r = new Relation()

          r.setId(sen.getId + "__1__" + seg.getSegmentId)
          r.setArgumentId(0, sen.getId)
          r.setArgumentId(1, seg.getUniqueId)
          r.setArgumentId(2, "1")
          r.setArgumentId(3, scoreVector(0).toString) // Computed & Normalized Score
          r
      }
      var index = 1;
      val negExps = negSegs.take(len).map {
        seg =>
          val r = new Relation()

          r.setId(sen.getId + "__0__" + seg.getSegmentId)
          r.setArgumentId(0, sen.getId)
          r.setArgumentId(1, seg.getUniqueId)
          r.setArgumentId(2, "0")
          r.setArgumentId(3, scoreVector(index).toString) // Computed & Normalized Score
          index += 1
          r
      }
      posExps ++ negExps
    }
    else { // if Testing

      val segs = segments().filter(s=> s.getUniqueId==sen.getId).toList
      segs.map{
        seg=>
          val r = new Relation()

          val isRel = if (seg.getUniqueId==sen.getId) "1" else "0"
          r.setId(sen.getId + "__" + isRel + "__" + seg.getSegmentId)
          r.setArgumentId(0, sen.getId)
          r.setArgumentId(1, seg.getUniqueId)
          r.setArgumentId(2, isRel)
          r
      }
    }
  }

  def getExpressionSegmentScore(sen: Sentence, seg: Segment, negSegs: List[Segment]) : List[Double] = {

    val allsegs = List(seg) ++ negSegs
    val instances = sen.getText.split(" ").map(w => {
      allsegs.map(s => new WordSegment(w, s, s.getUniqueId==seg.getUniqueId, false, "")).toList
    }).toList

    val scoresMatrix = instances.flatten.groupBy(i => i.getWord).map(w => {
      computeScore(w._1, w._2)
    }).toList
    val norm = normalizeScores(scoresMatrix)
    val vector = combineScores(norm)
    vector
  }
}