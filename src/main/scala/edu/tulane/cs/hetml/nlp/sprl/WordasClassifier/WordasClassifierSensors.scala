package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import edu.tulane.cs.hetml.vision._

object WordasClassifierSensors {

  def expressionToSegmentMatching(e: ExpressionSegment, s: Segment): Boolean = {
    e.getSegment.getAssociatedImageID == s.getAssociatedImageID && e.getSegment.getSegmentId == s.getSegmentId
  }

  def expressionToImageMatching(e: ExpressionSegment, i: Image): Boolean = {
    e.getSegment.getAssociatedImageID == i.getId
  }

  def imageToSegmentMatching(i: Image, s: Segment): Boolean = {
    i.getId == s.getAssociatedImageID
  }
}