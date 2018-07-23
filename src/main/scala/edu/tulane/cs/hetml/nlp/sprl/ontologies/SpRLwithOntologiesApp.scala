package edu.tulane.cs.hetml.nlp.sprl.ontologies

import java.io.{File, PrintWriter}
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader
import edu.tulane.cs.hetml.nlp.sprl.Helpers._
import edu.tulane.cs.hetml.nlp.sprl.ontologies.idatamodel._

import scala.collection.JavaConversions._

/**
  * Created by parisakordjamshidi on 12/15/17.
  */

object SpRLwithOntologiesApp extends App with Logging {

  val readerTrips = new NlpXmlReader("data/mSpRL/trips_cleaned_parses/newSprl2017_gold_output.xml", "SCENE", "SENTENCE", "PHRASE", null)

  val pw1 = new PrintWriter(new File("/Users/parisakordjamshidi/IdeaProjects/SpRL/data/mSpRL/trips_cleaned_parses/tripsStatistics/relationLabel.txt" ))
  val pw2 = new PrintWriter(new File("/Users/parisakordjamshidi/IdeaProjects/SpRL/data/mSpRL/trips_cleaned_parses/tripsStatistics/phraseLabel.txt" ))

  val tripsRelationList = readerTrips.getRelations("RELATION","head", "res")
  val tripsPhraseList = readerTrips.getPhrases()

  val tripsPhraseTypes = tripsPhraseList.map(x=> x.getPropertyValues("type")).distinct
  val tripsRelationTypes = tripsRelationList.map(x => (x.getProperty("label"))).distinct

  tripsPhraseTypes.foreach(x => pw1.write(x.toString+"\n"))
  tripsRelationTypes.foreach(x => pw2.write(x.toString+"\n"))


  val readerSpRL = new SpRLXmlReader("data/mSpRL/saiapr_tc-12/newSprl2017_gold.xml", false)

  //val c = readerSpRL.getTripletsWithArguments()

  readerSpRL.setRoles(tripsPhraseList.toList)
  //readerSpRL.reader.addPropertiesFromTag("Trajector",tripsPhraseList.toList)

  //readerSpRL.setTripletRelationTypes(tripsRelationList.toList)

  //logger.info("Role population started ...")

 // readerSpRL.setRoles(phrases.getTrainingInstances.toList)

  documents.populate(readerSpRL.getDocuments)
  sentences.populate(readerSpRL.getSentences)

  tripsPhrases.populate(tripsPhraseList)
  tripsRelations.populate(tripsRelationList)

  (tripsRelations()~>tripsRelationToPhrase1).foreach { x =>
    if (!x.getPropertyValues("TRAJECTOR_id").isEmpty)
      print("TR="+x.getPropertyValues("TRAJECTOR_text")+"\n")
    if (!x.getPropertyValues("LANDMARK_id").isEmpty)
      print("LM="+x.getPropertyValues("LANDMARK_text")+"\n")
    if (!x.getPropertyValues("SPATIALINDICATOR_id").isEmpty)
      print("SP="+x.getPropertyValues("SPATIALINDICATOR_text")+"\n")
    //print(x.getPropertyValues("TRAJECTOR_id"),x.getPropertyValues("LANDMARK_id"),x.getPropertyValues("SPATIALINDICATOR_id")+"\n")
  }
  //readerSpRL.reader.addPropertiesFromTag( "TRAJECTOR", tripsPhraseList, XmlMatchings.elementContainsXmlHeadwordMatching)

  //readerSpRL.reader.addPropertiesFromTag( "TRAJECTOR", phrases.getTrainingInstances.toList, XmlMatchings.elementContainsXmlHeadwordMatching)
  //readerTrips.addPropertiesFromTag("PHRASE",phrases.getTrainingInstances.toList, XmlMatchings.elementContainsXmlHeadwordMatching)

 // val s= phrases.getTrainingInstances.head.getPropertyFirstValue("PHRASE_type")

  //print("p= ", s)

  //logger.info("Role population finished.")
  pw1.close
  pw2.close

  print("hi")
}
