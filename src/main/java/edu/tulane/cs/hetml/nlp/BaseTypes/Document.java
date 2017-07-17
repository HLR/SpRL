package edu.tulane.cs.hetml.nlp.BaseTypes;


/**
 * Created by Taher on 2016-12-18.
 */
public class Document extends NlpBaseElement {
    public Document() {
    }

    public Document(String id) {
        super(id, -1, -1, "");
    }

    public Document(String id, Integer start, Integer end) {
        super(id, start, end, "");
    }

    public Document(String id, Integer start, Integer end, String text) {
        super(id, start, end, text);
    }

    @Override
    public NlpBaseElementTypes getType() {
        return NlpBaseElementTypes.Document;
    }
}
