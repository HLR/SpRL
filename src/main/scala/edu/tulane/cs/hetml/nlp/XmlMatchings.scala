package edu.tulane.cs.hetml.nlp

import edu.illinois.cs.cogcomp.saulexamples.nlp.SpatialRoleLabeling.Dictionaries
import edu.tulane.cs.hetml.nlp.BaseTypes.{ISpanElement, ISpanElementMatching, Phrase}
import edu.tulane.cs.hetml.nlp.LanguageBaseTypeSensors.{getHeadword, getTokens}

/** Created by Taher on 2016-12-28.
  */
object XmlMatchings {

  val xmlContainsElementHeadwordMatching = new ISpanElementMatching {

    override def matches(xmlElement: ISpanElement, element: ISpanElement) = {
      if (xmlElement.overlaps(element)) {
        val head = element match {
          case p: Phrase => getHeadword(p)
          case _ => element
        }
        xmlElement.contains(head)
      } else {
        false
      }
    }
  }

  val elementContainsXmlHeadwordMatching = new ISpanElementMatching {

    override def matches(xmlElement: ISpanElement, element: ISpanElement) = {
      if (xmlElement.overlaps(element)) {
        val (_, start, end) = getHeadword(xmlElement.getText)
        element.getStart <= start + xmlElement.getStart && element.getEnd >= end + xmlElement.getStart
      } else {
        false
      }
    }
  }

  val elementContainsXmlPrepositionMatching = new ISpanElementMatching {

    override def matches(xmlElement: ISpanElement, element: ISpanElement) = {
      if (xmlElement.overlaps(element)) {
        val prep = getTokens(xmlElement.getText).find(x => Dictionaries.isPreposition(x.getText))
        if (prep.isDefined) {
          val x = prep.get
          x.setStart(x.getStart + xmlElement.getStart)
          x.setEnd(x.getEnd + xmlElement.getStart)
          element.contains(x)
        }
        else {
          elementContainsXmlHeadwordMatching.matches(xmlElement, element)
        }
      } else {
        false
      }
    }
  }
}
