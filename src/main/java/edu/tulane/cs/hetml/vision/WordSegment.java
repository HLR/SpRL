package edu.tulane.cs.hetml.vision;

public class WordSegment {

    private String word;
    private Segment segment;
    private boolean isMatching;

    public WordSegment(String word, Segment segment, Boolean isMatching) {
        this.word = word;
        this.segment = segment;
        this.isMatching = isMatching;
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
}
