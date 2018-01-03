package edu.tulane.cs.hetml.vision;

public class SegmentPhraseHeadwordPair {

    private String imageId;
    private String segmentId;
    private String phraseHeadword;

    public SegmentPhraseHeadwordPair(String imageId, String segmentId, String phraseHeadword) {
        this.imageId = imageId;
        this.segmentId = segmentId;
        this.phraseHeadword = phraseHeadword;
    }

    public String getImageId() {
        return imageId;
    }

    public void setImageId(String imageId) {
        this.imageId = imageId;
    }

    public String getSegmentId() {
        return segmentId;
    }

    public void setSegmentId(String segmentId) {
        this.segmentId = segmentId;
    }

    public String getPhraseHeadword() {
        return phraseHeadword;
    }

    public void setPhraseHeadword(String phraseHeadword) {
        this.phraseHeadword = phraseHeadword;
    }
}
