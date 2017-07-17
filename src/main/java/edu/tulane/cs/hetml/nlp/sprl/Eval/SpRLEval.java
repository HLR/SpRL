package edu.tulane.cs.hetml.nlp.sprl.Eval;


/**
 * Created by Taher on 2016-09-20.
 */
public interface SpRLEval {
    boolean isEqual(SpRLEval b);

    boolean overlaps(SpRLEval b);

    boolean contains(SpRLEval b);

    boolean isPartOf(SpRLEval b);
}
