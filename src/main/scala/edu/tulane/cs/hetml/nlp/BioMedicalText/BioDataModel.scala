package edu.tulane.cs.hetml.nlp.BioMedicalText

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.illinois.cs.cogcomp.saulexamples.nlp.SpatialRoleLabeling.Dictionaries
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors._
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors._
import edu.tulane.cs.hetml.vision._
import scala.collection.JavaConversions._


import scala.collection.JavaConversions._

object BioDataModel extends DataModel {

  //get postag from


 // val documents = node[Document]((d: Document) => d.getId)
  val sentences = node[Sentence]((s: Sentence) => s.getId)
  val mentions = node[Phrase]((p: Phrase) => p.getId)
  val tokens = node[Token]((t: Token) => t.getId)


  //val docTosen = edge(documents, sentences)
  //docTosen.addSensor(documentToSentenceMatching _)

 val sentenceToPhrase = edge(sentences, mentions)
  sentenceToPhrase.addSensor(sentenceToPhraseGenerating _)

  val phraseToToken = edge(mentions, tokens)
  phraseToToken.addSensor(phraseToTokenGenerating _)


  //get feature from TAC files
  val mentiontype = property(mentions) {
    x: Phrase => x.getPropertyValues("type").toList
  }
  val textlength = property(mentions) {
    x: Phrase => x.getPropertyValues("text").toString.split(" ").length
  }

  val wordForm = property(mentions) {
    x: Phrase =>
      (mentions(x) ~> phraseToToken).toList.sortBy(_.getStart)
        .map(t => t.getText.toLowerCase).mkString("|")
  }


  val pos = property(mentions) {
    x: Phrase =>
      (mentions(x) ~> phraseToToken).toList.sortBy(_.getStart)
        .map(t => getPos(t).mkString).mkString("|")
  }

  val lemma = property(mentions) {
    x: Phrase =>
      (mentions(x) ~> phraseToToken).toList.sortBy(_.getStart)
        .map(t => getLemma(t).mkString).mkString("|")
  }

  val headWordLemma = property(mentions) {
    x: Phrase => getLemma(getHeadword(x)).mkString.toLowerCase
  }

  val headWordFrom = property(mentions) {
    x: Phrase => getHeadword(x).getText.toLowerCase
  }
  val headWordPos = property(mentions) {
    x: Phrase => getPos(getHeadword(x)).mkString
  }
  val phrasePos = property(mentions) {
    x: Phrase => getPhrasePos(x)}



}
