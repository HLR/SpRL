package edu.tulane.cs.hetml.nlp.BioMedicalText


import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.tulane.cs.hetml.nlp.BaseTypes._
import bionlpConfigurator._

import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors._
import scala.io.Source
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors.getGoogleWordVector

object BioDataModel extends DataModel {

  var useVectorAverages = false



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


  val mentionType2 = property(mentions) {
    x: Phrase =>
      if (x.containsProperty("Mention_type"))
        x.getPropertyFirstValue("Mention_type")
      else
        "None"
  }

  val isInDrugList = property(mentions) {
    x: Phrase => isDrug(x)

  }




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
  val posmodify = property(mentions) {
    x: Phrase =>
      getTokens(x).toList.sortBy(_.getStart)
        .map(t => getPos(t).mkString).mkString("|")
  }


  val lemma = property(mentions) {
    x: Phrase =>
      (mentions(x) ~> phraseToToken).toList.sortBy(_.getStart)
        .map(t => getLemma(t).mkString).mkString("|")
  }

  val headWordLemma = property(mentions) {
    x: Phrase =>
      if (pos(x).toString.contains("VBN") && pos(x).toString.contains("VB")) {
        getLemma(modifiedHeadWord(x)).mkString.toLowerCase
      }
      else {
        getLemma(getHeadword(x)).mkString.toLowerCase
      }


  }

  val headWordFrom = property(mentions) {
    x: Phrase =>
      if (pos(x).toString.contains("VBN") && pos(x).toString.contains("VB")) {
        modifiedHeadWord(x).getText.toLowerCase
      }
      else {
        getHeadword(x).getText.toLowerCase
      }
  }


  val headWordPos = property(mentions) {

    x: Phrase =>
      if (pos(x).toString.contains("VBN") && pos(x).toString.contains("VB")) {
      getPos(modifiedHeadWord(x)).mkString
    }
    else {
        getPos(getHeadword(x)).mkString
    }


  }
  val phrasePos = property(mentions) {
    x: Phrase => getPhrasePos(x)
  }



  val dependencyRelation = property(mentions) {
    x: Phrase =>
      (mentions(x) ~> phraseToToken).toList.sortBy(_.getStart)
        .map(t => getDependencyRelation(t)).mkString("|")
  }

  val headDependencyRelation = property(mentions) {
    x: Phrase =>
      if (pos(x).toString.contains("VBN") && pos(x).toString.contains("VB")) {
       getDependencyRelation(modifiedHeadWord(x))
      }
      else {
        getDependencyRelation(getHeadword(x))
      }

  }

  val subCategorization = property(mentions) {
    x: Phrase =>
      (mentions(x) ~> phraseToToken).toList.sortBy(_.getStart)
        .map(t => getSubCategorization(t)).mkString("|")
  }
  val headSubCategorization = property(mentions) {
    x: Phrase =>
      if (pos(x).toString.contains("VBN") && pos(x).toString.contains("VB")) {
        getSubCategorization(modifiedHeadWord(x))
      }
      else {
        getSubCategorization(getHeadword(x))
      }

  }


  val headVector = property(mentions, ordered = true) {
    x: Phrase =>
      if (pos(x).toString.contains("VBN") && pos(x).toString.contains("VB")) {
        getVector(modifiedHeadWord(x).getText.toLowerCase)
      }
      else {
        getVector(getHeadword(x).getText.toLowerCase)
      }
  }

  def isDrug(phrase: Phrase): Boolean = {
    val dictionarylist = ListBuffer[String]()

    val source = Source.fromFile(precipitantdictionary, "UTF-8")
    for (i <- source.getLines()) {
      dictionarylist.add(i)
    }
    source.close()
    //    if(dictionarylist.contains(phrase.toString))
    //      true
    //    else
    //      false
    for (m <- dictionarylist.toList) {
      if (phrase.toString.toLowerCase.contains(m)|| phrase.toString.contains(m))
        return true
    }
    false
  }


  private def getVector(w: String): List[Double] = {

    getGoogleWordVector(w)

  }


  def modifiedHeadWord(p: Phrase): Token = {
    val temp=new Token()
    val a = (mentions(p) ~> phraseToToken).toList
    for (i <- a)
      if (getPos(i).mkString == "VBN")
        return  i
    temp


  }









}

