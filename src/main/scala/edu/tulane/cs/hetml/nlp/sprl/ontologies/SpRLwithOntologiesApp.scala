package edu.tulane.cs.hetml.nlp.sprl.ontologies

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader
/**
  * Created by parisakordjamshidi on 12/15/17.
  */
object SpRLwithOntologiesApp extends App with Logging {

  val reader = new NlpXmlReader("data/mSpRL/trips_parse/train-sentences.txt-100.xml.rdf.clean", "SCENE", "SENTENCE", "PHRASE", null)
  val sentenceList = reader.getSentences()
  val phraselist = reader.getPhrases()
  println(phraselist.get(0).getPropertyValues("type"))
  println(sentenceList.size())
  println(phraselist.size())

}
