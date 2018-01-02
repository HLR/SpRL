package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

/** Created by Umar on 12/22/17.
  */
object WordasClassifierConfigurator {
  val resultsDir = "data/mSpRL/results/"
  val useAnntotatedClef = true
  val isTrain = false
  val imageDataPath = "data/mSprl/saiapr_tc-12"
  val alignmentAnnotationPath = imageDataPath + "/alignments/"
  val clefNewSegmentFeatures = imageDataPath + "/SegmentCNNFeatures/"
  val clefNewPhraseSegmentPairs = imageDataPath + "/PhraseSegmentPairs/"
  val trainWordsPath = imageDataPath + "/trainWords/"
  val preprocessReferitExp = true
  val trainFile = "data/mSprl/saiapr_tc-12/newSprl2017_train.xml"
  val testFile = "data/mSprl/saiapr_tc-12/newSprl2017_gold.xml"
  val iterations = 50
  val useReferClefTrained = false
  val useWord2VecClassifier = false
  val classifierPath = "models/mSpRL/wordclassiferClefWordsTrain/"
}
