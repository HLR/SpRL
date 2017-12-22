package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import edu.tulane.cs.hetml.nlp.sprl.Helpers.WordClassifierHelper
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierConfigurator._

/** Created by Umar on 2017-10-04.
  */

object WordasClassifierApp extends App {

  val wordClassifierHelper = new WordClassifierHelper()

  if(preprocessReferitExp)
    wordClassifierHelper.preprocessReferIt()

  if(isTrain) {
    println("Training...")
    wordClassifierHelper.trainOnFrequencyWordClassifiers()
    wordClassifierHelper.trainMissingWordsClassifers()
  }

  if(!isTrain) {
    println("Testing...")
    wordClassifierHelper.testWordClassifiers()
  }
}