package edu.tulane.cs.hetml.vision;

import java.util.*;

public class WordSegment {

    private String word;
    private Segment segment;

    public WordSegment(String word, Segment segment) {
        this.word = word;
        this.segment = segment;
    }

    public String getWord() {
        return word;
    }

    public Segment getSegment() {
        return segment;
    }
}
