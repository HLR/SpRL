package edu.tulane.cs.hetml.vision;

import java.util.*;

public class WordSegment {

    private String word;
    private Segment segment;
    private boolean word2segment;

    public WordSegment(String word, Segment segment, Boolean word2segment) {
        this.word = word;
        this.segment = segment;
        this.word2segment = word2segment;
    }

    public String getWord() {
        return word;
    }

    public Segment getSegment() {
        return segment;
    }

    public boolean getWord2Segment() {
        return word2segment;
    }
}
