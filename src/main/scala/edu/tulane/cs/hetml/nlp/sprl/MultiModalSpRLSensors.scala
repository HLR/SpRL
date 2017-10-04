package edu.tulane.cs.hetml.nlp.sprl

import java.io.File
import java.util.regex.Pattern

import edu.illinois.cs.cogcomp.saulexamples.nlp.SpatialRoleLabeling.Dictionaries
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors._
import edu.tulane.cs.hetml.nlp.sprl.Helpers.LexiconHelper
import edu.tulane.cs.hetml.vision._
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec

import scala.collection.immutable.HashMap
import scala.collection.JavaConversions._
import scala.util.matching.Regex
import MultiModalSpRLDataModel._

object MultiModalSpRLSensors {

  private lazy val googleWord2Vec = WordVectorSerializer.loadGoogleModel(new File("data/GoogleNews-vectors-negative300.bin"), true)
  private lazy val clefWord2Vec = WordVectorSerializer.readWord2VecModel("data/clef.bin")

  def getWord2VectorSimilarity(w2v: Word2Vec, w1: String, w2: String): Double = {
    if (w2v.getVocab.containsWord(w1) && w2v.getVocab.containsWord(w2))
      w2v.similarity(w1, w2)
    else
      0.0
  }

  def getWordVector(w2v: Word2Vec, w: String): List[Double] = {
    if (w == null)
      return List.fill(300)(0.0)
    val v = w2v.getWordVector(w)
    if (v == null) {
      List.fill(300)(0.0)
    } else {
      v.toList
    }
  }

  def getAverageSimilarity(w1: String, w2: String): Double = (googleWord2Vec.similarity(w1, w2) + getClefSimilarity(w1, w2)) / 2

  def getGoogleSimilarity(w1: String, w2: String): Double = getWord2VectorSimilarity(googleWord2Vec, w1, w2)

  def getGoogleWordVector(w: String): List[Double] = getWordVector(googleWord2Vec, w)

  def getClefSimilarity(w1: String, w2: String): Double = getWord2VectorSimilarity(clefWord2Vec, w1, w2)

  def getClefWordVector(w: String): List[Double] = getWordVector(clefWord2Vec, w)

  def getAverage(a: List[Double]*): List[Double] = a.head.zipWithIndex.map { case (_, i) => a.map(_ (i)).sum / a.size }

  def refinedSentenceToPhraseGenerating(s: Sentence): Seq[Phrase] = {
    val phrases = mergeUsingIndicatorLex(s, sentenceToPhraseGenerating(s))
    //mergeUsingTemplates(s, phrases)
    phrases
  }

  def imageToSegmentMatching(i: Image, s: Segment): Boolean = {
    i.getId == s.getAssociatedImageID
  }

  def segmentRelationToSegmentMatching(r: SegmentRelation, s: Segment): Boolean = {
    (r.getFirstSegmentId == s.getSegmentId || r.getSecondSegmentId == s.getSegmentId) && (r.getImageId == s.getAssociatedImageID)
  }

  def relationToFirstArgumentMatching(r: Relation, p: Phrase): Boolean = {
    r.getArgumentId(0) == p.getId
  }

  def relationToSecondArgumentMatching(r: Relation, p: Phrase): Boolean = {
    r.getArgumentId(1) == p.getId
  }

  def relationToThirdArgumentMatching(r: Relation, p: Phrase): Boolean = {
    r.getArgumentId(2) == p.getId
  }

  def documentToImageMatching(d: Document, i: Image): Boolean = {
    d.getPropertyFirstValue("IMAGE").endsWith("/" + i.getLabel)
  }

  def segmentToSegmentPhrasePairs(s: Segment): List[Relation] = {
    val image = images().filter(i=>i.getId == s.getAssociatedImageID)
    val phrases = (images(image) ~> -documentToImage ~> documentToSentence ~> sentenceToPhrase)
      .filter(p => p != dummyPhrase)
      .toList
    phrases.map {
      p =>
        val r = new Relation()
        r.setId(p.getId + "__" + s.getSegmentId)
        r.setArgumentId(0, p.getId)
        r.setArgumentId(1, s.getSegmentId.toString)
        r
    }
  }
  def SegmentPhrasePairToPhraseMatching(pair: Relation, phrase: Phrase): Boolean = {
    pair.getArgumentId(0) == phrase.getId
  }

  val phraseConceptToWord = HashMap(
    "child-boy" -> "child",
    "child-girl" -> "child",
    "construction-other" -> "construction",
    "couple-of-persons" -> "humans",
    "face-of-person" -> "person",
    "group-of-persons" -> "humans",
    "hand-of-person" -> "person",
    "head-of-person" -> "person",
    "person-related-objects" -> "person",
    "floor-other" -> "floor",
    "floor-wood" -> "floor",
    "floor-carpet" -> "floor",
    "floor-tennis-court" -> "floor",
    "branch" -> "trees",
    "non-wooden-furniture" -> "furniture",
    "wooden-furniture" -> "furniture",
    "furniture-other" -> "furniture",
    "public-sign" -> "public-sign",
    "ruin-archeological" -> "edifice",
    "sand-beach" -> "beach",
    "sand-dessert" -> "dessert",
    "sky-blue" -> "sky",
    "sky-light" -> "sky",
    "sky-night" -> "sky",
    "sky-red-sunset-dusk" -> "sky",
    "water-reflection" -> "water",
    "air-vehicles" -> "vehicle",
    "ground-vehicles" -> "vehicle",
    "vehicles-with-tires" -> "vehicle",
    "public-sign" -> "sign",
    "ancent-building" -> "edifice",
    "boat-rafting" -> "vehicle",
    "polar-bear" -> "bear",
    "fowl-hen" -> "bird",
    "flock-of-birds" -> "animal",
    "school-of-fishes" -> "animal",
    "man-made-other" -> "man-made",
    "ocean-animal" -> "animal",
    "herd-of-mammals" -> "animal",
    "mammal-other" -> "mammal"
  )

  val SpToImageSp = HashMap(
    "of his left" -> "left",
    "on" -> "above",
    "on the right side" -> "right",
    "under" -> "below",
    "underneath" -> "below",
    "to the right" -> "right",
    "on the left" -> "left",
    "on top of" -> "above",
    "to the left" -> "left",
    "on the right" -> "right",
    "next to" -> "adjacent",
    "on the right side" -> "right",
    "on top" -> "above",
    "along the left side of" -> "left",
    "on the right tower" -> "right",
    "of his left hand" -> "left",
    "on the right and smaller shops" -> "right",
    "is lying around" -> "",
    "behind" -> "",
    "in between" -> "",
    "are leaning on" -> "",
    "lying around in" -> "",
    "in" -> "",
    "surrounded by" -> "",
    "the top of" -> "",
    "among" -> "",
    "is leaning against" -> "",
    "in front" -> "",
    "away from" -> "",
    "the distant background" -> "",
    "at the back" -> "",
    "below" -> "below",
    "is standing outside" -> "",
    "leading up" -> "",
    "outside" -> "",
    "before" -> "",
    "are lying around in" -> "",
    "in the centre of" -> "",
    "is going up" -> "",
    "are sitting around" -> "",
    "out from" -> "",
    "in the front row" -> "",
    "above" -> "above",
    "opposite" -> "",
    "are leading up to" -> "",
    "in the distance" -> "",
    "out of" -> "",
    "in the centre" -> "",
    "around" -> "",
    "near background" -> "",
    "at" -> "",
    "through" -> "",
    "in distance" -> "",
    "in the front" -> "",
    "over" -> "",
    "on" -> "above",
    "on each side" -> "",
    "in centre of" -> "",
    "lying around in front" -> "",
    "at the front" -> "",
    "lined up" -> "",
    "under" -> "below",
    "attached" -> "",
    "distant background" -> "",
    "are walking over" -> "",
    "is sitting outside" -> "",
    "between" -> "",
    "in middle of" -> "",
    "sitting outside" -> "",
    "leaning against" -> "",
    "is walking towards" -> "",
    "is climbing up" -> "",
    "are just sitting around" -> "",
    "along" -> "",
    "at the back of" -> "",
    "in the middle of" -> "",
    "at each side" -> "",
    "leaning on" -> "",
    "in front of" -> "",
    "leading up" -> "",
    "leading up to" -> "",
    "leaning against" -> "",
    "leaning on" -> "",
    "lined up" -> "",
    "lying around" -> "",
    "lying around in" -> "",
    "near" -> "",
    "of" -> "",
    "on each side" -> "",
    "opposite" -> "",
    "out from" -> "",
    "out of" -> "",
    "outside" -> "",
    "over" -> "",
    "sitting around" -> "",
    "sitting outside" -> "",
    "standing outside" -> "",
    "supporting" -> "",
    "surrounded by" -> "",
    "the distant" -> "",
    "through" -> "",
    "to" -> "",
    "walking over" -> "",
    "walking towards" -> "",
    "with" -> ""
  )


  private def mergeUsingTemplates(s: Sentence, phrases: Seq[Phrase]): Seq[Phrase] = {

    val pattern = Pattern.compile("(^|[^\\w])(in .+? of)([^\\w]|$)")
    val matcher = pattern.matcher(s.getText.toLowerCase)
    while (matcher.find()) {
      val m = matcher.toMatchResult
      val span = new SpanBasedElement
      span.setStart(m.start)
      span.setEnd(m.end)
      val toMerge = phrases.filter(p => span.overlaps(p))
      if (toMerge.length > 1) {
        val tokens = toMerge.flatMap(getTokens)
        if (tokens.length < 6 && tokens.exists(t => Dictionaries.isPreposition(t.getText))) {
          val phrase = toMerge.head
          phrase.setEnd(toMerge.last.getEnd)
          phrase.setText(s.getText.substring(phrase.getStart, phrase.getEnd))
          toMerge.tail.foreach(x => {
            x.setStart(-1)
            x.setEnd(-1)
          })
        }
      }
    }
    phrases.filter(_.getStart != -1)
  }

  private def mergeUsingIndicatorLex(s: Sentence, phrases: Seq[Phrase]) = {
    val lex = LexiconHelper.spatialIndicatorLexicon
      .filter(l => l.contains(" ") && s.getText.toLowerCase.contains(l))
      .sortBy(x => -x.length)

    if (lex.nonEmpty) {
      lex.foreach(l => {
        val regex = new Regex("(^|[^a-z])(" + l + ")([^a-z]|$)")
        val matches = regex.findAllMatchIn(s.getText)
        matches.foreach(m => {

          val span = new SpanBasedElement
          span.setStart(m.start)
          span.setEnd(m.end)
          if (!m.toString().head.isLetter)
            span.setStart(span.getStart + 1)
          if (!m.toString().last.isLetter)
            span.setEnd(span.getEnd - 1)

          val toMerge = phrases.filter(p => span.overlaps(p))
          if (toMerge.size > 1) {
            val phrase = toMerge.head
            phrase.setEnd(toMerge.last.getEnd)
            phrase.setText(s.getText.substring(phrase.getStart, phrase.getEnd))
            toMerge.tail.foreach(x => {
              x.setStart(-1)
              x.setEnd(-1)
            })
          }
        })
      })
    }
    phrases.filter(_.getStart != -1)
  }

}