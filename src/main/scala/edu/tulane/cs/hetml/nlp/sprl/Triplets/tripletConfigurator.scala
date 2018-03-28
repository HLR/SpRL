package edu.tulane.cs.hetml.nlp.sprl.Triplets

import edu.tulane.cs.hetml.nlp.sprl.Helpers.FeatureSets
import edu.tulane.cs.hetml.nlp.sprl.WordasClassifier.WordasClassifierConfigurator

/** Created by parisakordjamshidi on 3/23/17.
  */
object tripletConfigurator {
  val onTheFlyLexicon = true // build the sp lex on the fly when training or using pre existed lex
  val resultsDir = "data/mSpRL/results/"
  val imageDataPath = "data/mSprl/saiapr_tc-12"
  val alignmentAnnotationPath = imageDataPath + "/alignments/"
  val modelDir = "models/mSpRL/"
  val spatialIndicatorLex = "data/mSprl/spatialIndicator.lex"
  val trainFile = "data/mSprl/saiapr_tc-12/newSprl2017_train.xml"
  val testFile = "data/mSprl/saiapr_tc-12/newSprl2017_gold.xml"
  val suffix = ""
  val model = FeatureSets.BaseLine
  var isTrain = false
  WordasClassifierConfigurator.isTrain = isTrain
  val trainTestTogether = false
  val jointTrain = false
  val iterations = 50
  val useConstraints = false
  val usePrepositions = false
  val trainPrepositionClassifier = false
  val alignmentMethod = "gold" // possible values: "classifier" "gold", "w2v", "topN"
  val topAlignmentCount = 3
  var populateImages = false //model == FeatureSets.WordEmbeddingPlusImage || model == FeatureSets.BaseLineWithImage
  val globalSpans = false // set true when dataset has global spans for roles
  val useCoReference = false
  var useCoReferenceConstraints = false
}
