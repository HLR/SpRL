package edu.tulane.cs.hetml.vision;



import org.bytedeco.javacpp.presets.opencv_core;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Umar Manzoor on 29/12/2016.
 */
public class Segment {
    private int segmentId;
    private int segmentCode;
    public String segmentFeatures;
    private String segmentConcept;
    private String imageId;
    public List<String> tagged;
    public String referItExpression;
    public String filteredTokens;
    private boolean isMatching;
    private String boxDimensions;

    public Segment(String ImageId, int segmentId, String segmentFeatures, String referItExpression, boolean isMatching) {
        this.imageId = ImageId;
        this.segmentId = segmentId;
        this.segmentFeatures = segmentFeatures;
        this.referItExpression = referItExpression;
        this.isMatching = isMatching;
        this.tagged = new ArrayList<>();
    }

    public Segment(String ImageId, int segmentId, int segmentCode, String segmentFeatures, String segmentConcept, String boxDimensions)
    {
        this.imageId = ImageId;
        this.segmentId = segmentId;
        this.segmentCode = segmentCode;
        this.segmentFeatures = segmentFeatures;
        this.segmentConcept = segmentConcept;
        this.boxDimensions = boxDimensions;
    }

    public String getAssociatedImageID()
    {
        return imageId;
    }

    public  int getSegmentId()
    {
        return segmentId;
    }

    public String getSegmentFeatures()
    {
        return segmentFeatures;

    }
    public  int getSegmentCode()
    {
        return segmentCode;
    }

    public String getSegmentConcept()
    {
        return segmentConcept;
    }

    public String getExpression() {
        return referItExpression;
    }

    public boolean isExpressionAndSegmentMatching() {
        return isMatching;
    }

    public void setFilteredTokens(String filteredTokens) {
        this.filteredTokens = filteredTokens;
    }

    public void setTagged(List<String> tagged) {
        this.tagged = tagged;
    }

    public String getUniqueId() {
        return imageId + "_" + segmentId;
    }

    public String getBoxDimensions() {
        return boxDimensions;
    }

    public void setBoxDimensions(String boxDimensions) {
        this.boxDimensions = boxDimensions;
    }

}