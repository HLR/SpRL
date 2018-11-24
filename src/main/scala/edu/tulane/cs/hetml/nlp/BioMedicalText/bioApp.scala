package edu.tulane.cs.hetml.nlp.BioMedicalText

import java.io.FileOutputStream
import java.util

import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader
import bionlpConfigurator._
import BioDataModel._
import edu.tulane.cs.hetml.nlp.BaseTypes.Phrase
import edu.tulane.cs.hetml.nlp.BioMedicalText.bioclassifier.biomentionclassifier
import edu.tulane.cs.hetml.nlp.XmlMatchings
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors._
import edu.tulane.cs.hetml.nlp.sprl.Helpers.ReportHelper

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}



object   bioApp extends App {

  val trainDataReader= new NlpXmlReader(trainfile,null,"SENTENCE","Mention",null)
   val testDataReader= new NlpXmlReader(testfile,null,"SENTENCE","",null)

 val phraseTrainReader=trainDataReader.getPhrases()
  val trainSentences = trainDataReader.getSentences()
 val testSentences= testDataReader.getSentences()
  // val readTrigger=phraseTrainReader.filter(x=> x.getPropertyFirstValue("type").equals("Precipitant"))
  // val newphrase=phraseTrainReader.filter(x=>getPos(x).toString.contains("VB"))


    val initialPhrases = trainSentences.map(x => sentenceToPhraseGenerating(x)).flatten
   val newphrases=initialPhrases.filter(x=>phrasePos(x)=="VP")

  // sentences.populate(trainSentences)
    mentions.populate(newphrases)
    sentences.populate(testSentences,false)
    //mentions().foreach(x=>print(x+"  "+pos(x)+"\n"))
//print(mentions.getTestingInstances.toList)

trainDataReader.addPropertiesFromTag("Mention",mentions.getTrainingInstances.toList, XmlMatchings.elementContainsXmlHeadwordMatching) //You do not need to populate the mention() node then since it is populated with chuncker automatically. However, please check how much the mention() covers phraseTrainReader.
testDataReader.addPropertiesFromTag("Mention",mentions.getTestingInstances.toList,XmlMatchings.elementContainsXmlHeadwordMatching)

//
//  mentions() prop mentionType2
//      val allprecipitant = mentions().filter(x => mentionType2(x).equals("Precipitant"))
//      allprecipitant.foreach(x=>print(x+"\n"))

  if (isTrain) {
    println("training started ...")
    biomentionclassifier.learn(iteration=30)
    biomentionclassifier.save()
    biomentionclassifier.test()
   // biomentionclassifier.load()
    //biomentionclassifier.test()
    for(i<- mentions.getTestingInstances){
      val prediction=biomentionclassifier.classifier.classify(i)

      print(i+"\t"+prediction)
      print("\n")
    }

  }

}
