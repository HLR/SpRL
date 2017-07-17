package edu.tulane.cs.hetml.nlp.sprl.Eval;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Taher on 2016-09-19.
 */
@XmlRootElement(name = "Relations")
public class RelationsEvalDocument {

    public RelationsEvalDocument(List<RelationEval> relations) {
        Relations = relations;
    }

    public RelationsEvalDocument() {
        Relations = new ArrayList<>();
    }

    @XmlElement(name = "Relation", required = true)
    private List<RelationEval> Relations;

    public List<RelationEval> getRelations() {
        return Relations;
    }
}
