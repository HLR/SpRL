package edu.tulane.cs.hetml.nlp.sprl.Eval;


/**
 * Created by taher on 2017-04-18.
 */
public interface RelationTypeExtractor {
    String getType(RelationEval e, boolean isActual);
}
