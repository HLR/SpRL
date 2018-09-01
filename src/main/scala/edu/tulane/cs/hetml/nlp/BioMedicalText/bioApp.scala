package edu.tulane.cs.hetml.nlp.BioMedicalText
import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader
import bionlpConfigurator._
import BioDataModel._
import edu.tulane.cs.hetml.nlp.BioMedicalText.bioclassifier.biomentionclassifier
import edu.tulane.cs.hetml.nlp.XmlMatchings

import scala.collection.JavaConversions._

object  bioApp extends App {

    val trainDataReader= new NlpXmlReader(trainfile,null,"SENTENCE","Mention",null)
   // val biotestData= new NlpXmlReader(testfile,null,"SENTENCE","Mention",null)

    val phraseTrainReader=trainDataReader.getPhrases()
    //Here you get the list of mentions from your XML input.

    val trainSentences = trainDataReader.getSentences()
   // Here when you get the sentences from yout XML reader.

    //val biomentiontest=biotestData.getPhrases()

    sentences.populate(trainSentences)
    //Here you populate sentences node in the data model using the sentences that you have read from the reader.
    // Since you have a generating sensor from sentence to phrase
    //Phrases are generated automatically using a shallow parser.  This will fill up your mentions() node.
    //Next you need to add the tag labels  from xml to the generated chuncks in mentions(). You do it as follows:
    trainDataReader.addPropertiesFromTag("Mention", mentions.getTrainingInstances.toList, XmlMatchings.elementContainsXmlHeadwordMatching)

    //You do not need to populate the mention() node then since it is populated with chuncker automatically. However, please check how much the mention() covers phraseTrainReader.

    //mentions.populate(biomentiontest)
    mentions().foreach(x=>print(headWordPos(x)))

//    if (isTrain) {
//        println("training started ...")
//        biomentionclassifier.learn(iteration=30)
//       // biomentionclassifier.test(biomentiontest)
//
//
//
//
//        }

}
