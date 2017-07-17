package edu.tulane.cs.hetml.nlp.BaseTypes;

/**
 * Created by Taher on 2016-12-24.
 */
public class Phrase extends NlpBaseElement {

    private Sentence sentence;

    public Phrase(){

    }

    public Phrase(Sentence sentence, String id, Integer start, Integer end, String text) {
        super(id, start, end, text);
        this.sentence = sentence;
    }

    @Override
    public NlpBaseElementTypes getType() {
        return NlpBaseElementTypes.Phrase;
    }

    public Document getDocument() {
        return getSentence().getDocument();
    }

    public Sentence getSentence() {
        return sentence;
    }

    public void setSentence(Sentence sentence) {
        this.sentence = sentence;
    }
}