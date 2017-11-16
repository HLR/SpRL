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

    @Override
    public int getRelativeStart(){
        if(getStart() < 0)
            return getStart();

        return isGlobalSpan()? getStart() - sentence.getStart(): getStart();
    }

    @Override
    public int getRelativeEnd(){
        if(getEnd() < 0)
            return getEnd();

        return isGlobalSpan()? getEnd() - sentence.getStart(): getEnd();
    }

    @Override
    public int getGlobalStart(){
        if(getStart() < 0)
            return getStart();

        return isGlobalSpan()? getStart(): getStart() + sentence.getStart();
    }

    @Override
    public int getGlobalEnd(){
        if(getEnd() < 0)
            return getEnd();

        return isGlobalSpan()? getEnd(): getEnd() + sentence.getStart();
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