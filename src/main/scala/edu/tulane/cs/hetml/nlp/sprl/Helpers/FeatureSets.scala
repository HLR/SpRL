package edu.tulane.cs.hetml.nlp.sprl.Helpers

/** Created by taher on 2017-02-28.
  */
object FeatureSets extends Enumeration {
  type FeatureSets = Value
  val BaseLine, WordEmbedding, BaseLineWithImage, WordEmbeddingPlusImage = Value
}
