package edu.tulane.cs.hetml.nlp.sprl.VisualTriplets

import edu.tulane.cs.hetml.vision._
import edu.tulane.cs.hetml.nlp.sprl.VisualTriplets.VisualTripletsDataModel._
import scala.collection.JavaConversions._
/** Created by Umar on 2017-11-09.
  */
object VisualTripletsApp extends App {

  val visualTripletReader = new ImageTripletReader("data/mSprl/saiapr_tc-12/imageTriplets")

  val trainTriplets = visualTripletReader.trainImageTriplets
  val testTriplets = visualTripletReader.testImageTriplets

  visualTriplets.populate(trainTriplets)


}

