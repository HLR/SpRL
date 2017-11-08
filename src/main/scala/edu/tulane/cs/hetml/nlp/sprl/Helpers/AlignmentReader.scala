package edu.tulane.cs.hetml.nlp.sprl.Helpers

import edu.tulane.cs.hetml.nlp.BaseTypes.Phrase
import edu.tulane.cs.hetml.vision.CLEFAlignmentReader
import scala.collection.JavaConversions._

class AlignmentReader(annotationDir: String, textDir: String) {
  val reader = new CLEFAlignmentReader(annotationDir, textDir)

  def setAlignments(phrases: List[Phrase]) = {
    val fileAlignments = reader.getAlignments.groupBy(_ (0))
    fileAlignments.foreach {
      case (f, sentences) =>
        val filePhrases = phrases.filter(p => p.getDocument.getId.contains(s"/$f.eng"))
        if (filePhrases.nonEmpty) {

          val sentenceAlignments = sentences.groupBy(_ (1))

          sentenceAlignments.foreach {
            case (s, alignments) =>

              val sentencePhrases = filePhrases.filter(p => p.getSentence.getText.equalsIgnoreCase(s))

              if(sentencePhrases.isEmpty){
                println(s"sentence '$s' from file $f skipped")
              }
              else {

                val phraseList = alignments.groupBy(_ (2))

                phraseList.foreach {
                  case (p, segments) =>

                    var phrase = sentencePhrases.filter(x => x.getText.equalsIgnoreCase(p))
                    if (phrase.isEmpty)
                      phrase = sentencePhrases.filter(x => x.getText.split(",").exists(_.trim.equalsIgnoreCase(p)))

                    if (phrase.length != 1)
                      println(s"No unique phrase matches for '$p': ${phrase.length} in file $f")
                    segments.foreach({
                      x =>
                        phrase.head.addPropertyValue("alignedSegment", x(3))
                    })
                }
              }
          }
        }
    }
  }

}
