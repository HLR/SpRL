package edu.tulane.cs.hetml.nlp.sprl;

import edu.illinois.cs.cogcomp.core.datastructures.IntPair;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.Sentence;

import java.util.List;

/**
 * Created by Taher on 2016-09-06.
 */
public class SpRLSentence {
    private final IntPair offset;
    private final Sentence sentence;
    private final List<SpRLRelationContainer> relations;
    public SpRLSentence(IntPair offset, Sentence sentence, List<SpRLRelationContainer> relations) {
        this.offset = offset;
        this.sentence = sentence;
        this.relations = relations;
    }

    public Sentence getSentence() {
        return sentence;
    }

    public List<SpRLRelationContainer> getRelations() {
        return relations;
    }

    public IntPair getOffset() {
        return offset;
    }
}
