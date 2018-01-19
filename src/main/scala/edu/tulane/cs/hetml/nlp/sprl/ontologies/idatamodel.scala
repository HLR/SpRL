package edu.tulane.cs.hetml.nlp.sprl.ontologies

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.{documentToSentenceMatching, phraseToTokenGenerating, sentenceToPhraseGenerating, sentenceToRelationMatching}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel.edge
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors.{refinedSentenceToPhraseGenerating, relationToFirstArgumentMatching, relationToSecondArgumentMatching, relationToThirdArgumentMatching}

/**
  * Created by parisakordjamshidi on 12/15/17.
  */
object idatamodel extends DataModel {

  val documents = node[Document]((d: Document) => d.getId)
  val sentences = node[Sentence]((s: Sentence) => s.getId)
  val phrases = node[Phrase]((p: Phrase) => p.getId)
  val tokens = node[Token]((t: Token) => t.getId)
  val pairs = node[Relation]((r: Relation) => r.getId)
  val triplets = node[Relation]((r: Relation) => r.getId)


  /*
  Edges
   */
  val documentToSentence = edge(documents, sentences)
  documentToSentence.addSensor(documentToSentenceMatching _)

  val sentenceToPhrase = edge(sentences, phrases)
  sentenceToPhrase.addSensor(sentenceToPhraseGenerating _)

//  val phraseToToken = edge(phrases, tokens)
//  phraseToToken.addSensor(phraseToTokenGenerating _)
//
//  val sentenceToPairs = edge(sentences, pairs)
//  sentenceToPairs.addSensor(sentenceToRelationMatching _)
//
//  val pairToFirstArg = edge(pairs, phrases)
//  pairToFirstArg.addSensor(relationToFirstArgumentMatching _)
//
//  val pairToSecondArg = edge(pairs, phrases)
//  pairToSecondArg.addSensor(relationToSecondArgumentMatching _)
//
//  var sentenceToTriplets = edge(sentences, triplets)
//  sentenceToTriplets.addSensor(sentenceToRelationMatching _)
//
//  val tripletToTr = edge(triplets, phrases)
//  tripletToTr.addSensor(relationToFirstArgumentMatching _)
//
//  val tripletToSp = edge(triplets, phrases)
//  tripletToSp.addSensor(relationToSecondArgumentMatching _)
//
//  val tripletToLm = edge(triplets, phrases)
//  tripletToLm.addSensor(relationToThirdArgumentMatching _)

}
