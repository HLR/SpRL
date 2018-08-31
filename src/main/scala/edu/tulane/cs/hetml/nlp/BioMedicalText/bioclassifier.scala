package edu.tulane.cs.hetml.nlp.BioMedicalText

import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import BioDataModel._
import edu.illinois.cs.cogcomp.lbjava.learn.{SparseAveragedPerceptron, SparseNetworkLearner, SupportVectorMachine}
import edu.tulane.cs.hetml.nlp.BaseTypes.Phrase
import edu.illinois.cs.cogcomp.saul.datamodel.property.Property


object bioclassifier {
  def phraseFeatures(): List[Property[Phrase]] =
    List(textlength,lemma,wordForm,pos,headWordPos,phrasePos)

  object biomentionclassifier extends Learnable(mentions){
    def label=mentiontype
    override lazy val classifier = new SparseNetworkLearner {
      val p = new SparseAveragedPerceptron.Parameters()
      p.learningRate = .1
      p.thickness = 2
      baseLTU = new SparseAveragedPerceptron(p)
    }

    override def feature = phraseFeatures()





  }


}
