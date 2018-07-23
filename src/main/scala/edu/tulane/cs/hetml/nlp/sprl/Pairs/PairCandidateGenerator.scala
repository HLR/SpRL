package edu.tulane.cs.hetml.nlp.sprl.Pairs

import java.io.{File, IOException, PrintWriter}

import edu.illinois.cs.cogcomp.saulexamples.nlp.SpatialRoleLabeling.Dictionaries
import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors._
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{LexiconHelper, ReportHelper}
import edu.tulane.cs.hetml.nlp.sprl.Ontology.MultiModalSpRLDataModel.{phrases, _}

import scala.collection.JavaConversions._
import scala.io.Source

/** Created by taher on 2017-02-28.
  */
object PairCandidateGenerator {

  def generateTripletCandidatesFromPairs(
                                          trSpFilter: (Relation) => Boolean,
                                          spFilter: (Phrase) => Boolean,
                                          lmSpFilter: (Relation) => Boolean,
                                          isTrain: Boolean
                                        ): List[Relation] = {
    val instances = if (isTrain) phrases.getTrainingInstances else phrases.getTestingInstances
    val indicators = instances.filter(t => t.getId != dummyPhrase.getId && spFilter(t)).toList
      .sortBy(x => x.getSentence.getStart + x.getStart)

    indicators.flatMap(sp => {
      val pairs = phrases(sp) ~> -pairToSecondArg
      val trajectorPairs = (pairs.filter(r => trSpFilter(r)) ~> pairToFirstArg).groupBy(x => x).keys
      if (trajectorPairs.nonEmpty) {
        val landmarkPairs = (pairs.filter(r => lmSpFilter(r)) ~> pairToFirstArg).groupBy(x => x).keys
        if (landmarkPairs.nonEmpty) {
          trajectorPairs.flatMap(tr => landmarkPairs.map(lm => createRelation(Some(tr), Some(sp), Some(lm))))
            .filter(r => r.getArgumentIds.toList.distinct.size == 3) // distinct arguments
            .toList
        } else {
          List()
        }
      } else {
        List()
      }
    })
  }

  def generatePairCandidates(
                              phraseInstances: List[Phrase],
                              populateNullPairs: Boolean,
                              indicatorClassifier: Phrase => Boolean
                            ): List[Relation] = {

    val trCandidates = getTrajectorCandidates(phraseInstances)
    trCandidates.foreach(_.addPropertyValue("TR-Candidate", "true"))

    val lmCandidates = getLandmarkCandidates(phraseInstances)
    lmCandidates.foreach(_.addPropertyValue("LM-Candidate", "true"))

    val spCandidates = phraseInstances.filter(indicatorClassifier) // getIndicatorCandidates(phraseInstances)
    spCandidates.foreach(_.addPropertyValue("SP-Candidate", "true"))

    val firstArgCandidates = (if (populateNullPairs) List(null) else List()) ++
      phraseInstances.filter(x => x.containsProperty("TR-Candidate") || x.containsProperty("LM-Candidate"))

    val candidateRelations = getCandidateRelations(firstArgCandidates, spCandidates)
    candidateRelations.foreach(x => x.setParent(x.getArgument(1).asInstanceOf[Phrase].getSentence))

    if (populateNullPairs) {
      // replace null arguments with dummy token
      candidateRelations.filter(_.getArgumentId(0) == null).foreach(x => {
        x.setArgumentId(0, dummyPhrase.getId)
        x.setArgument(0, dummyPhrase)
      })
    }
    candidateRelations
  }

  def getIndicatorCandidates(phrases: List[Phrase]): List[Phrase] = {

    val spLex = LexiconHelper.spatialIndicatorLexicon
    val spPosTagLex = List("IN", "TO")
    val spCandidates = phrases
      .filter(x =>
        spLex.exists(s => x.getText.toLowerCase.matches("(^|.*[^\\w])" + s + "([^\\w].*|$)")) ||
          spPosTagLex.exists(p => getPos(x).contains(p)) ||
          Dictionaries.isPreposition(x.getText))
    ReportHelper.reportRoleStats(phrases, spCandidates, "SPATIALINDICATOR")
    spCandidates
  }

  def getLandmarkCandidates(phrases: List[Phrase]): List[Phrase] = {

    val lmPosTagLex = List("PRP", "NN", "PRP$", "JJ", "NNS", "NNP", "CD")
    val lmCandidates = phrases.filter(x => lmPosTagLex.exists(p => getPos(x).contains(p)))
    ReportHelper.reportRoleStats(phrases, lmCandidates, "LANDMARK")
    lmCandidates
  }

  def getTrajectorCandidates(phrases: List[Phrase]): List[Phrase] = {

    val trPosTagLex = List("NN", "JJR", "PRP$", "VBG", "JJ", "NNP", "NNS", "CD", "VBN", "VBD")
    val trCandidates = phrases.filter(x => trPosTagLex.exists(p => getPos(x).contains(p)))
    ReportHelper.reportRoleStats(phrases, trCandidates, "TRAJECTOR")
    trCandidates
  }

  private def createRelation(tr: Option[Phrase], sp: Option[Phrase], lm: Option[Phrase]): Relation = {

    val r = new Relation()
    r.setArgument(0, if (tr.nonEmpty && tr.get.getId != dummyPhrase.getId) tr.get else dummyPhrase)
    r.setArgumentId(0, r.getArgument(0).getId)
    r.setArgument(1, sp.get)
    r.setArgumentId(1, r.getArgument(1).getId)
    r.setArgument(2, if (lm.nonEmpty && lm.get.getId != dummyPhrase.getId) lm.get else dummyPhrase)
    r.setArgumentId(2, r.getArgument(2).getId)

    //set relation parent
    r.setParent(sp.get.getSentence)
    r.setId(r.getArgumentId(0) + "_" + r.getArgumentId(1) + "_" + r.getArgumentId(2))
    r
  }

}
