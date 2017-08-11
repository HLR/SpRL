package edu.tulane.cs.hetml.nlp.sprl

import edu.tulane.cs.hetml.nlp.sprl.Helpers.{CandidateGenerator, ReportHelper, SpRLXmlReader}
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.Eval.{OverlapComparer, SpRLEvaluation}

import scala.collection.JavaConversions._

/** Created by taher on 2017-02-26.
  */
object TripletClassifierUtils {

  def test(
            dataPath: String,
            resultsDir: String,
            resultsFilePrefix: String,
            isTrain: Boolean,
            trClassifier: (Relation) => Boolean,
            spClassifier: (Phrase) => Boolean,
            lmClassifier: (Relation) => Boolean
          ): Seq[SpRLEvaluation] = {

    val predicted: List[Relation] = predict(trClassifier, spClassifier, lmClassifier, isTrain)
    val actual = new SpRLXmlReader(dataPath).getTripletsWithArguments()

    ReportHelper.reportRelationResults(resultsDir, resultsFilePrefix + "_triplet", actual, predicted, new OverlapComparer, 3)
  }

  def predict(
               trClassifier: (Relation) => Boolean,
               spClassifier: (Phrase) => Boolean,
               lmClassifier: (Relation) => Boolean,
               isTrain: Boolean = false
             ): List[Relation] = {
    CandidateGenerator.generateTripletCandidatesFromPairs(trClassifier, spClassifier, lmClassifier, isTrain)
  }

}

