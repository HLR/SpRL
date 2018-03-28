package edu.tulane.cs.hetml.nlp.sprl.ontologies

import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader
import edu.tulane.cs.hetml.nlp.XmlMatchings
import edu.tulane.cs.hetml.nlp.sprl.Helpers._
import edu.tulane.cs.hetml.nlp.sprl.ontologies.idatamodel._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._

import scala.collection.JavaConversions._
/**
  * Created by parisakordjamshidi on 12/15/17.
  */
object SpRLwithOntologiesApp extends App with Logging {

  val readerTrips = new NlpXmlReader("data/mSpRL/trips_cleaned_parses/train-sentences.txt-100.xml", "SCENE", "SENTENCE", "PHRASE", null)

  val readerSpRL = new SpRLXmlReader("data/mSpRL/trips_cleaned_parses/testTrips.xml", false)

  logger.info("Role population started ...")

  readerSpRL.setRoles(phrases.getTrainingInstances.toList)

  documents.populate(readerSpRL.getDocuments, isTrain)
  sentences.populate(readerSpRL.getSentences, isTrain)

  readerSpRL.reader.addPropertiesFromTag( "TRAJECTOR", phrases.getTrainingInstances.toList, XmlMatchings.elementContainsXmlHeadwordMatching)
  readerTrips.addPropertiesFromTag("PHRASE",phrases.getTrainingInstances.toList, XmlMatchings.elementContainsXmlHeadwordMatching)

  val s= phrases.getTrainingInstances.head.getPropertyFirstValue("PHRASE_type")

  print("p= ", s)

  logger.info("Role population finished.")

}
