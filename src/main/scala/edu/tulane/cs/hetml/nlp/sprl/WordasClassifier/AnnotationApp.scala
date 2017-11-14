package edu.tulane.cs.hetml.nlp.sprl.WordasClassifier

import java.io.PrintWriter

import edu.tulane.cs.hetml.nlp.BaseTypes._
import edu.tulane.cs.hetml.nlp.sprl.Helpers.{CandidateGenerator, ReportHelper}
import edu.tulane.cs.hetml.nlp.sprl.MultiModalPopulateData.populateRoleDataFromAnnotatedCorpus
import edu.tulane.cs.hetml.nlp.sprl.MultiModalSpRLDataModel._
import edu.tulane.cs.hetml.nlp.sprl.mSpRLConfigurator._
import edu.tulane.cs.hetml.vision._

import scala.collection.JavaConversions._

/** Created by Umar on 2017-10-04.
  */

object AnnotationApp extends App {

  val ClefAnnReader = new CLEFAnnotationReader(imageDataPath)
  val testImages = ClefAnnReader.clefImages.toList
  val testSegments = ClefAnnReader.clefSegments.toList

  populateRoleDataFromAnnotatedCorpus()

  images.populate(testImages)
  segments.populate(testSegments)

  val docs = documents().filter(d=> testImages.exists(t=> {d.getId.endsWith("/" + t.getId + ".eng")}))

  val completeResult = new PrintWriter(s"data/mSprl/results/annotation/fullResult.txt")

  val sens = sentences().filter(s=> docs.exists(d=> d.getId.equals(s.getDocument.getId)))

  val trCandidates = CandidateGenerator.getTrajectorCandidates(phrases().filter(p => p!=dummyPhrase
    && testImages.exists(t=> {p.getDocument.getId.endsWith("/" + t.getId + ".eng")})).toList)

  val lmCandidates = CandidateGenerator.getLandmarkCandidates(phrases().filter(p => p!=dummyPhrase
    && testImages.exists(t=> {p.getDocument.getId.endsWith("/" + t.getId + ".eng")})).toList)

  var TrCandidateCount = 0
  var TrGoldCount = 0
  var TrCandidateMatch = 0
  var TrGoldMatch = 0

  var LmCandidateCount = 0;
  var LmGoldCount = 0;
  var LmCandidateMatch = 0;
  var LmGoldMatch = 0

  docs.foreach(d => {
    val name = (d.getId.split("/").takeRight(1))(0)

    completeResult.println(name)

    val writer = new PrintWriter(s"data/mSprl/results/annotation/${name}.txt")
    writer.println("Sentences:")
    sens.filter(s => s.getDocument.getId == d.getId).foreach(s => {
      writer.println(s.getId + ": " + s.getText)
    })
    writer.println("---------------------------------------------")
    writer.println("Trajector Phrases:")
    writer.println("---------------------------------------------")

    val trStats = generateStats(writer, d, trCandidates, "TRAJECTOR_id", "TR")
    TrCandidateCount += trStats(0)
    TrGoldCount += trStats(1)
    TrCandidateMatch += trStats(2)
    TrGoldMatch += trStats(3)

    writer.println("LandMark Phrases:")
    writer.println("---------------------------------------------")

    val lmStats = generateStats(writer, d, lmCandidates, "LANDMARK_id", "LM")

    LmCandidateCount += lmStats(0)
    LmGoldCount += lmStats(1)
    LmCandidateMatch += lmStats(2)
    LmGoldMatch += lmStats(3)

    completeResult.println("---------------------------------------------")

    writer.close()
  })

  val tr = List(TrCandidateCount, TrGoldCount, TrCandidateMatch, TrGoldMatch)
  val lm = List(LmCandidateCount, LmGoldCount, LmCandidateMatch, LmGoldMatch)

  writeSummary(completeResult, tr, lm)
  completeResult.close()

  def generateStats(writer: PrintWriter, d: Document, phrases: List[Phrase], roleId: String, role: String): List[Int] = {
    var roleCount = 0
    var size = 0
    var goldRole = 0
    var goldMatchedCount = 0
    phrases.filter(t => t.getDocument.getId == d.getId).foreach(t => {
      var goldMatched = "false"
      var line = t.getText
      size += 1
      if(t.containsProperty(roleId)) {
        line += ", Gold "
        goldRole += 1
        goldMatched = "true"
      }
      else
        line += ", Candidate "
      if(annotationAnalysis(t)=="true") {
        roleCount += 1
        line += ", M"
        if(goldMatched=="true")
          goldMatchedCount += 1
      }
      else {
        line += ", NM"
      }
      writer.println(line)
    })
    writer.println("")
    writer.println("-----------------Summary-------------------")
    writer.println(s"Candidate ${role}: ${size} Gold ${role}: ${goldRole} Candidate Matched: ${roleCount} Gold Matched: ${goldMatchedCount}")
    writer.println("-------------------------------------------")

    completeResult.println(s"-----------------${role}-------------------")
    completeResult.println(s"Candidate ${role}: ${size} Gold ${role}: ${goldRole} Candidate Matched: ${roleCount} Gold Matched: ${goldMatchedCount}")
    List(size, goldRole, roleCount, goldMatchedCount)
  }

  def writeSummary(writer: PrintWriter, tr: List[Int], lm: List[Int]): Unit = {
    completeResult.println("-------------------Summary--------------------------")
    completeResult.println(s"Documents: ${docs.size}, Sentences: ${sens.size}")
    completeResult.println(s"Candidate TR: ${tr(0)}, Gold TR: ${tr(1)}, Candidate Matched: ${tr(2)}, Gold Matched: ${tr(3)}")
    completeResult.println(s"Candidate LM: ${lm(0)}, Gold LM: ${lm(1)}, Candidate Matched: ${lm(2)}, Gold Matched: ${lm(3)}")
    completeResult.println("----------------------End----------------------------")
  }
}