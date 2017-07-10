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
    private List<String> ontologyConcepts= new ArrayList<>();
    public double[] features;

    public Segment(String ImageId, int segmentId, int segmentCode, String segmentFeatures, String segmentConcept, List<String> ontologyConcepts)
    {
        this.imageId = ImageId;
        this.segmentId = segmentId;
        this.segmentCode = segmentCode;
        this.segmentFeatures = segmentFeatures;
        this.segmentConcept = segmentConcept;
        this.ontologyConcepts = ontologyConcepts;
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

    public List<String> getOntologyConcepts()
    {
        return ontologyConcepts;
    }

    @Override
    public String toString() {
        // TODO Auto-generated method stub
        return imageId + ", " + segmentId + ", " + segmentCode + ", " + segmentFeatures + ", " + segmentConcept;
    }

}
