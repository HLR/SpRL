package edu.tulane.cs.hetml.nlp.sprl.Ontology

import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader
import MultiModalSpRLDataModel._
import scala.collection.JavaConversions._


object testApp extends App{
  val mybioreader= new NlpXmlReader("data/manuallyCollectedTripsParse_train.xml","Document","Sentence","PHRASE",null)
  val biomention=mybioreader.getPhrases()
  phrases.populate(biomention)

  //mentions().foreach(x=>print(mentiontype(x)))// 获取定义的属性



  phrases().foreach(x=>print(pos(x)))

 phrases().foreach(x=>print(wordForm(x)))

}
