package edu.tulane.cs.hetml.nlp.sprl.Helpers

import edu.tulane.cs.hetml.nlp.BaseTypes.Relation
import edu.tulane.cs.hetml.nlp.sprl.Eval.{OverlapComparer, SpRLEvaluation}

import scala.collection.JavaConversions._

/** Created by taher on 2017-02-26.
  */
object PairClassifierUtils {

  def evaluate(
                predicted: List[Relation],
                dataPath: String,
                resultsDir: String,
                resultsFilePrefix: String,
                isTrain: Boolean,
                isTrajector: Boolean
              ): Seq[SpRLEvaluation] = {

    val reader = new SpRLXmlReader(dataPath)
    val actual = if(isTrajector) reader.getTrSpPairsWithArguments() else reader.getLmSpPairsWithArguments()

    val name = if (isTrajector) "TrSp" else "LmSp"
    ReportHelper.reportRelationResults(resultsDir, resultsFilePrefix + s"_$name", actual, predicted, new OverlapComparer, 2)
  }

}

