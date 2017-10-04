package edu.tulane.cs.hetml.vision;



import java.util.ArrayList;
import java.util.List;

/**
 * Created by Umar Manzoor on 29/12/2016.
 */
public class Segment {
    private int segmentId;
    private int segmentCode;
    private String segmentFeatures;
    private String segmentConcept;
    private String imageId;
    public List<String> tagged;
    public String refExp;

    public Segment(String ImageId, int segmentId, String segmentFeatures, String refExp) {
        this.imageId = ImageId;
        this.segmentId = segmentId;
        this.segmentFeatures = segmentFeatures;
        this.refExp = refExp;
        this.tagged = new ArrayList<>();
    }

    public Segment(String ImageId, int segmentId, int segmentCode, String segmentFeatures, String segmentConcept, List<String> tagged, List<String> refExp)
    {
        this.imageId = ImageId;
        this.segmentId = segmentId;
        this.segmentCode = segmentCode;
        this.segmentFeatures = segmentFeatures;
        this.segmentConcept = segmentConcept;
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
}