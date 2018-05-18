package edu.tulane.cs.hetml.vision;

import edu.tulane.cs.hetml.nlp.BaseTypes.Phrase;

public class WordSegment {

    private String word;
    private Segment segment;
    private Phrase phrase;
    private boolean isMatching;
    private boolean isHead;
    private String pos;

    public WordSegment(String word, Segment segment, Boolean isMatching) {
        this.word = word;
        this.segment = segment;
        this.isMatching = isMatching;
    }

    public WordSegment(String word, Segment segment, Boolean isMatching, boolean isHead, String pos) {
        this.word = word;
        this.segment = segment;
        this.isMatching = isMatching;
        this.isHead = isHead;
        this.setPos(pos);
    }

    public String getWord() {
        return word;
    }

    public Segment getSegment() {
        return segment;
    }

    public boolean isWordAndSegmentMatching() {
        return isMatching;
    }

    public boolean isHead() {
        return isHead;
    }

    public void setHead(boolean head) {
        isHead = head;
    }

    public String getPos() {
        return pos;
    }

    public void setPos(String pos) {
        this.pos = pos;
    }

    public Phrase getPhrase() {
        return phrase;
    }

    public void setPhrase(Phrase phrase) {
        this.phrase = phrase;
    }
}
