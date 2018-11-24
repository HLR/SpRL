package edu.tulane.cs.hetml.nlp.BioMedicalText

import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import BioDataModel._
import edu.illinois.cs.cogcomp.lbjava.learn.{SparseAveragedPerceptron, SparseNetworkLearner, SupportVectorMachine}
import edu.tulane.cs.hetml.nlp.BaseTypes.Phrase
import edu.illinois.cs.cogcomp.saul.datamodel.property.Property


object bioclassifier {
  def phraseFeatures(): List[Property[Phrase]] =
    //List(pos,wordForm,isInDrugList,headWordFrom,phrasePos,headWordLemma,lemma,dependencyRelation,headDependencyRelation,subCategorization,headSubCategorization)
     //List(phrasePos,lemma,wordForm,dependencyRelation)
//     def phraseFeatures(): List[Property[Phrase]]
       List(headVector,pos,wordForm,isInDrugList,headWordFrom,headWordPos,phrasePos,headWordLemma,lemma,dependencyRelation,headDependencyRelation,subCategorization,headSubCategorization)


  object biomentionclassifier extends Learnable(mentions){// trigger classifier
    def label=mentionType2

    override lazy val classifier = new SupportVectorMachine()
     {
      val p=new SupportVectorMachine.Parameters()



    }
//    override lazy val classifier = new SparseNetworkLearner {
//      val p = new SparseAveragedPerceptron.Parameters()
//      p.learningRate = .1
//      p.thickness = 1
//      baseLTU = new SparseAveragedPerceptron(p)
//    }
    override def feature = phraseFeatures
  }


//  object bioprecipitantclassifier extends Learnable(mentions){//precipitant classifier
//    def label=mentionType2
//
//    override lazy val classifier = new SupportVectorMachine()
//    {
//      val p=new SupportVectorMachine.Parameters()
//      p.C=2
//
//
//    }
//    //    override lazy val classifier = new SparseNetworkLearner {
//    //      val p = new SparseAveragedPerceptron.Parameters()
//    //      p.learningRate = .1
//    //      p.thickness = 1
//    //      baseLTU = new SparseAveragedPerceptron(p)
//    //    }
//    override def feature = phraseFeatures
//  }
//
//
//  object biospecificclassifier extends Learnable(mentions){//specific classifier
//  def label=mentionType2
//
//    override lazy val classifier = new SupportVectorMachine()
//    {
//      val p=new SupportVectorMachine.Parameters()
//      p.C=2
//
//
//    }
//    //    override lazy val classifier = new SparseNetworkLearner {
//    //      val p = new SparseAveragedPerceptron.Parameters()
//    //      p.learningRate = .1
//    //      p.thickness = 1
//    //      baseLTU = new SparseAveragedPerceptron(p)
//    //    }
//    override def feature = phraseFeatures
//  }













}
