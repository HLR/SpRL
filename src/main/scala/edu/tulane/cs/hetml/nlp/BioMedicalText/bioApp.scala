package edu.tulane.cs.hetml.nlp.BioMedicalText
import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader
import bionlpConfigurator._
import BioDataModel._
import edu.tulane.cs.hetml.nlp.BioMedicalText.bioclassifier.biomentionclassifier

import scala.collection.JavaConversions._

object  bioApp extends App {

    val biotrainData= new NlpXmlReader(trainfile,null,"SENTENCE","Mention",null)
    val biotestData= new NlpXmlReader(testfile,null,"SENTENCE","Mention",null)

    val biomentiontrain=biotrainData.getPhrases()
    //val biomentiontest=biotestData.getPhrases()

    mentions.populate(biomentiontrain)
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
