package edu.tulane.cs.hetml.nlp.sprl

import edu.tulane.cs.hetml.nlp.sprl.Helpers.FeatureSets

/** Created by parisakordjamshidi on 3/23/17.
  */
object mSpRLConfigurator {
  val onTheFlyLexicon = true // build the sp lex on the fly when training or using pre existed lex
  val resultsDir = "data/mSpRL/results/"
  val imageDataPath = "data/mSprl/saiapr_tc-12"
  val alignmentAnnotationPath = imageDataPath + "/alignments/"
  val modelDir = "models/mSpRL/"
  val spatialIndicatorLex = "data/mSprl/spatialIndicator.lex"
  val trainFile = "data/mSprl/saiapr_tc-12/newSprl2017_train.xml"
  val testFile = "data/mSprl/saiapr_tc-12/newSprl2017_gold.xml"//"data/TrainSet.xml"
  val suffix = ""
  val model = FeatureSets.BaseLineWithImage
  var isTrain = false
  val trainTestTogether = false
  val useAnntotatedClef = true
  val jointTrain = false
  val skipIndividualClassifiersTraining = false  /* When using joint train, it will ignore individual classifiers
                                                  * training and loads them from the disk*/
  val iterations = 50
  val useConstraints = true
  val trainPrepositionClassifier = false
  val alignmentMethod = "classifier" // possible values: "classifier" "gold", "w2v"
  val imageConstraints = model == FeatureSets.WordEmbeddingPlusImage || model == FeatureSets.BaseLineWithImage
  val populateImages = true //model == FeatureSets.WordEmbeddingPlusImage || model == FeatureSets.BaseLineWithImage
  val globalSpans = false // set true when dataset has global spans for roles
}
