package edu.tulane.cs.hetml.nlp.sprl.Helpers

import edu.tulane.cs.hetml.nlp.BaseTypes.Phrase
import scala.collection.JavaConversions._

class AlignmentReader(annotationDir: String, isTrain: Boolean) {

  def setAlignments(phrases: List[Phrase]) = {
    val name = if (isTrain) "train.txt" else "test.txt"
    val annotationLines = scala.io.Source.fromFile(annotationDir + name).getLines()
    annotationLines.filter(_.trim != "").foreach {
      l =>
        val part = l.split("\t\t")
        val imFolder = part(0)
        val imId = part(1)
        val sentId = part(2)
        val sentence = part(3)
        val start = part(4).toInt
        val end = part(5).toInt
        val text = part(6)
        val segId =part(7).toInt
        val segX = part(8).toDouble.toInt
        val segY = part(9).toDouble.toInt
        val segWidth = part(10).toDouble.toInt
        val segHeight = part(11).toDouble.toInt
        val phrase = phrases.find(x=> x.getSentence.getId == sentId && x.getStart == start && x.getEnd == end).get
        phrase.addPropertyValue("goldAlignment", segId.toString)
        phrase.addPropertyValue("imageId", imId.toString)
        phrase.addPropertyValue("segId", segId.toString)
        phrase.addPropertyValue("segX", segX.toString)
        phrase.addPropertyValue("segY", segY.toString)
        phrase.addPropertyValue("segWidth", segWidth.toString)
        phrase.addPropertyValue("segHeight", segHeight.toString)
    }
  }

}
