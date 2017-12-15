package edu.tulane.cs.hetml.vision;

import java.awt.geom.Rectangle2D;

public class ImageTriplet {
    private String imageId;
    private int firstSegId;
    private int secondSegId;
    private String trajector;
    private String landmark;
    private String sp;
    private Rectangle2D trBox;
    private Rectangle2D lmBox;
    private double imageWidth;
    private double imageHeight;

    public void setTrBox(Rectangle2D trBox) {
        this.trBox = trBox;
    }

    public void setLmBox(Rectangle2D lmBox) {
        this.lmBox = lmBox;
    }

    public void setImageWidth(double imageWidth) {
        this.imageWidth = imageWidth;
    }

    public void setImageHeight(double imageHeight) {
        this.imageHeight = imageHeight;
    }

    public ImageTriplet(String sp, String trajector, String landmark, Rectangle2D trBox, Rectangle2D lmBox, double imageWidth,
                        double imageHeight) {
        this.setSp(sp);
        this.setTrajector(trajector);
        this.setLandmark(landmark);
        this.trBox = trBox;
        this.lmBox = lmBox;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
    }

    public ImageTriplet(String imageId, int firstSegId, int secondSegId, Rectangle2D trBox, Rectangle2D lmBox,
                        double imageWidth, double imageHeight) {
        this.setImageId(imageId);
        this.setFirstSegId(firstSegId);
        this.setSecondSegId(secondSegId);
        this.trBox = trBox;
        this.lmBox = lmBox;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
    }

    public String getSp() {
        return sp;
    }

    public double getImageWidth() {
        return imageWidth;
    }
    public double getImageHeight() {
        return imageHeight;
    }

    public String getLandmark() {
        return landmark;
    }

    public double getEuclideanDistance() {
        return RectangleHelper.getEuclideanDistance(trBox, lmBox);
    }

    public double getIou() {
        return RectangleHelper.getIntersectionOverUnion(trBox, lmBox);
    }

    public double getLmAreaBbox() {
        Rectangle2D b = RectangleHelper.generateBoundingBox(trBox, lmBox);
        double a = RectangleHelper.calculateArea(lmBox);
        double bArea = RectangleHelper.calculateArea(b);
        return RectangleHelper.normalizeArea(a, bArea);
    }

    public double getLmAreaImage() {
        double a = RectangleHelper.calculateArea(lmBox);
        double iArea = imageHeight * imageWidth;
        return RectangleHelper.normalizeArea(a, iArea);
    }

    public double getLmAspectRatio() {
        return RectangleHelper.calculateAspectRatio(lmBox);
    }

    public double getTrAreaBbox() {
        Rectangle2D b = RectangleHelper.generateBoundingBox(trBox, lmBox);
        double a = RectangleHelper.calculateArea(trBox);
        double bArea = RectangleHelper.calculateArea(b);
        return RectangleHelper.normalizeArea(a, bArea);
    }

    public double getTrAreaImage() {
        double a = RectangleHelper.calculateArea(trBox);
        double iArea = imageHeight * imageWidth;
        return RectangleHelper.normalizeArea(a, iArea);
    }

    public double getTrAreawrtLM() {
        double tr = RectangleHelper.calculateArea(trBox);
        double lm = RectangleHelper.calculateArea(lmBox);
        return  tr / lm;
    }

    public double getTrAspectRatio() {
        return RectangleHelper.calculateAspectRatio(trBox);
    }

    public Rectangle2D getLmBox() {
        return lmBox;
    }

    public String getTrajector() {
        return trajector;
    }

    public Rectangle2D getTrBox() {
        return trBox;
    }

    public double getTrVectorX() {
        Rectangle2D b = RectangleHelper.generateBoundingBox(trBox, lmBox);
        return RectangleHelper.getCentroidVector(trBox, lmBox, b)[0];
    }

    public double getTrVectorY() {
        Rectangle2D b = RectangleHelper.generateBoundingBox(trBox, lmBox);
        return RectangleHelper.getCentroidVector(trBox, lmBox, b)[1];
    }

    public String getImageId() {
        return imageId;
    }

    public void setImageId(String imageId) {
        this.imageId = imageId;
    }

    public int getFirstSegId() {
        return firstSegId;
    }

    public void setFirstSegId(int firstSegId) {
        this.firstSegId = firstSegId;
    }

    public int getSecondSegId() {
        return secondSegId;
    }

    public void setSecondSegId(int secondSegId) {
        this.secondSegId = secondSegId;
    }

    public void setSp(String sp) {
        this.sp = sp;
    }

    public void setTrajector(String trajector) {
        this.trajector = trajector;
    }

    public void setLandmark(String landmark) {
        this.landmark = landmark;
    }

    public double getAbove() {
        return RectangleHelper.getAbove(trBox, lmBox, imageHeight);
    }

    public double getBelow() {
        return RectangleHelper.getBelow(trBox, lmBox, imageHeight);
    }

    public double getLeft() {
        return RectangleHelper.getLeft(trBox, lmBox, imageWidth);
    }

    public double getRight() {
        return RectangleHelper.getRight(trBox, lmBox, imageWidth);
    }

    public double getIntersectionArea() {
        return RectangleHelper
                .getIntersectionArea(trBox, lmBox, imageHeight * imageWidth);
    }

    public double getUnionArea() {
        return RectangleHelper
                .getUnionArea(trBox, lmBox, imageHeight * imageWidth);
    }
}
