package edu.tulane.cs.hetml.nlp.BioMedicalText
import java.util

import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader
import bionlpConfigurator._
import BioDataModel._
import edu.tulane.cs.hetml.nlp.BioMedicalText.bioclassifier.{biotriggerclassifier}
import edu.tulane.cs.hetml.nlp.XmlMatchings
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors._

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

object  bioApp extends App {

    val trainDataReader= new NlpXmlReader(trainfile,null,"SENTENCE","Mention",null)
    val testDataReader= new NlpXmlReader(testfile,null,"SENTENCE",null,null)

    val phraseTrainReader=trainDataReader.getPhrases()

    val trainSentences = trainDataReader.getSentences()
    val testSentences= testDataReader.getSentences()
    //val readTrigger=phraseTrainReader.filter(x=> x.getPropertyFirstValue("type").equals("Precipitant"))
    //val newphrase=phraseTrainReader.filter(x=>getPos(x).toString.contains("VB"))


    val initialPhrases = trainSentences.foreach(x => sentenceToPhraseGenerating(x))
    //val newPhrases= initialPhrases.filter(x => getPos(x).toString.contains("VB"))//  The error

    sentences.populate(trainSentences)
    //mentions.populate(newphrase)
    sentences.populate(testSentences,false)
   // mentions().foreach(x=>print(x.toString+"   "+headWordFrom(x)+"\n"))


    val newphrases2=mentions.getTrainingInstances.filter(x=>getPos(x).toString.contains("VB"))
    trainDataReader.addPropertiesFromTag("Mention",newphrases2.toList, XmlMatchings.elementContainsXmlHeadwordMatching) //You do not need to populate the mention() node then since it is populated with chuncker automatically. However, please check how much the mention() covers phraseTrainReader.
    testDataReader.addPropertiesFromTag("Mention",mentions.getTestingInstances.toList,XmlMatchings.elementContainsXmlHeadwordMatching)
    //mentions() prop mentionType2
//    val allprecipitant = mentions().filter(x => mentionType2(x).equals("Trigger"))
//    print(allprecipitant)

    if (isTrain) {
        println("training started ...")
        biotriggerclassifier.learn(iteration=30)
        biotriggerclassifier.test()
        biotriggerclassifier.save()
//        biomentionclassifier.load()
//        biomentionclassifier.test()
        for(i<-mentions.getTestingInstances){
            val prediction=biotriggerclassifier.classifier.classify(i)
            print(i)
            print(prediction)
            print('\n')

        }
        }

}
