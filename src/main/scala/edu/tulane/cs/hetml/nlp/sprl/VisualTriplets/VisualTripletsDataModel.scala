package edu.tulane.cs.hetml.nlp.sprl.VisualTriplets

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.tulane.cs.hetml.vision._

/** Created by Umar on 2017-11-09.
  */
object VisualTripletsDataModel extends DataModel {

  val visualTriplets = node[ImageTriplet]

  val wordSegFeatures = property(visualTriplets) {
    t: ImageTriplet =>
      ""
  }

}
