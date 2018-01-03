package edu.tulane.cs.hetml.nlp.sprl.Pairs

import edu.tulane.cs.hetml.nlp.sprl.Helpers.FeatureSets

/** Created by parisakordjamshidi on 3/23/17.
  */
object pairConfigurator {
  val onTheFlyLexicon = true // build the sp lex on the fly when training or using pre existed lex
  val resultsDir = "data/mSpRL/results/"
  val imageDataPath = "data/mSprl/saiapr_tc-12"
  val modelDir = "models/mSpRL/"
  val spatialIndicatorLex = "data/mSprl/spatialIndicator.lex"
  val trainFile = "data/mSprl/saiapr_tc-12/newSprl2017_train.xml"//"data/TrainSet.xml"//
  val testFile = "data/mSprl/saiapr_tc-12/newSprl2017_gold.xml"//"data/TestSet.xml"//
  val suffix = ""
  val model = FeatureSets.BaseLineWithImage
  var isTrain = false
  val iterations = 50
  val useConstraints = false
  var populateImages = true //model == FeatureSets.WordEmbeddingPlusImage || model == FeatureSets.BaseLineWithImage
}
