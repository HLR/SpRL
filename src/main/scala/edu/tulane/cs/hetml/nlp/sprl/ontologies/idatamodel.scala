package edu.tulane.cs.hetml.nlp.sprl.ontologies

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.documentToSentenceMatching
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel.edge
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLSensors.refinedSentenceToPhraseGenerating

/**
  * Created by parisakordjamshidi on 12/15/17.
  */
object idatamodel extends DataModel {

  val documents = node [Document]
  val sentences = node [Sentence]
  val phrases = node [Phrase]

  val documentToSentence = edge(documents, sentences)
  documentToSentence.addSensor(documentToSentenceMatching _)

  val sentenceToPhrase = edge(sentences, phrases)
  sentenceToPhrase.addSensor(refinedSentenceToPhraseGenerating _)

}
