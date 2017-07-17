package edu.tulane.cs.hetml.vision;


/**
 * Created by Umar Manzoor on 02/01/2017.
 */
public class SegmentRelation {
    private String imageId;
    private int firstSegmentId;
    private int secondSegmentId;
    private String relation;

    public SegmentRelation(String imageId, int firstSegmentId, int secondSegmentId, String relation) {
        this.imageId = imageId;
        this.firstSegmentId = firstSegmentId;
        this.secondSegmentId = secondSegmentId;
        this.relation = relation;
    }

    public String getRelation() {
        return relation;
    }

    public int getFirstSegmentId() {
        return firstSegmentId;
    }

    public int getSecondSegmentId() {
        return secondSegmentId;
    }
    public String getImageId()
    {
        return imageId;
    }

}
