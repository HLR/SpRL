package edu.tulane.cs.hetml.nlp.BaseTypes;


/**
 * Created by Taher on 2016-12-18.
 */
public class Token extends NlpBaseElement {

    private Sentence sentence;
    private Phrase phrase;

    public Token() {

    }

    public Token(Sentence sentence, String id, Integer start, Integer end, String text) {
        this(sentence, null, id, start, end, text);
    }

    public Token(Phrase phrase, String id, Integer start, Integer end, String text) {
        this(phrase.getSentence(), phrase, id, start, end, text);
    }

    public Token(Sentence sentence, Phrase phrase, String id, Integer start, Integer end, String text) {
        super(id, start, end, text);
        this.setSentence(sentence);
        this.setPhrase(phrase);
    }

    @Override
    public NlpBaseElementTypes getType() {
        return NlpBaseElementTypes.Token;
    }

    public Document getDocument() {
        return getSentence().getDocument();
    }

    public Sentence getSentence() {
        if (sentence != null)
            return sentence;
        return phrase != null ? phrase.getSentence() : null;
    }

    public void setSentence(Sentence sentence) {
        this.sentence = sentence;
    }

    public Phrase getPhrase() {
        return phrase;
    }

    public void setPhrase(Phrase phrase) {
        this.phrase = phrase;
    }
}
