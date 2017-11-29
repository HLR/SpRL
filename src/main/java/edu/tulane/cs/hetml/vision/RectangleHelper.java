package edu.tulane.cs.hetml.vision;

import java.awt.geom.Rectangle2D;

public class RectangleHelper {

    public static double[] getCentroidVector(Rectangle2D trBox, Rectangle2D lmBox, Rectangle2D boundingbox) {
        return new double[]{
                (trBox.getCenterX() - lmBox.getCenterX()) / boundingbox.getWidth(),
                (trBox.getCenterY() - lmBox.getCenterY()) / boundingbox.getHeight()
        };
    }

    public static double getIntersectionOverUnion(Rectangle2D trBox, Rectangle2D lmBox) {
        Rectangle2D intersection = trBox.createIntersection(lmBox);
        Rectangle2D union = trBox.createUnion(lmBox);

        double intersectionArea = calculateArea(intersection);
        double unionArea = calculateArea(union);

        double iou = 0;
        if (unionArea != 0)
            iou = intersectionArea / unionArea;
        return iou;
    }

    public static Rectangle2D generateBoundingBox(Rectangle2D trBox, Rectangle2D lmBox) {
        double minX = Math.min(trBox.getMinX(), lmBox.getMinX());
        double maxX = Math.max(trBox.getMaxX(), lmBox.getMaxX());

        double minY = Math.min(trBox.getMinY(), lmBox.getMinY());
        double maxY = Math.max(trBox.getMaxY(), lmBox.getMaxY());

        Rectangle2D bBox = new Rectangle2D.Double(minX, minY, maxX - minX, maxY - minY);
        return bBox;
    }

    public static double calculateArea(Rectangle2D box) {
        return box.getHeight() * box.getWidth();
    }

    public static double calculateAspectRatio(Rectangle2D box) {
        return box.getWidth() / box.getHeight();
    }

    public static double normalizeArea(double boxArea, double imageArea) {
        return boxArea / imageArea;
    }

    public static double getEuclideanDistance(Rectangle2D trBox, Rectangle2D lmBox) {

        double xDiff = trBox.getX() - lmBox.getX();
        double xSqr = Math.pow(xDiff, 2);

        double yDiff = trBox.getX() - lmBox.getX();
        double ySqr = Math.pow(yDiff, 2);

        return Math.sqrt(xSqr + ySqr);
    }

    public static double getAbove(Rectangle2D trBox, Rectangle2D lmBox, double height) {
        return Math.max(0, (lmBox.getY()- trBox.getY()) / height);
    }

    public static double getBelow(Rectangle2D trBox, Rectangle2D lmBox, double height) {
        return Math.max(0, (trBox.getY() - lmBox.getY()) / height);
    }

    public static double getLeft(Rectangle2D trBox, Rectangle2D lmBox, double width) {
        return Math.max(0, (trBox.getX() - lmBox.getX()) / width);
    }

    public static double getRight(Rectangle2D trBox, Rectangle2D lmBox, double width) {
        return Math.max(0, (lmBox.getX() - trBox.getX()) / width);
    }

    public static Rectangle2D parseRectangle(String s, String delimeter) {
        String[] parts = s.split(delimeter);

        return new Rectangle2D.Double(
                Double.parseDouble(parts[0]),
                Double.parseDouble(parts[1]),
                Double.parseDouble(parts[2]),
                Double.parseDouble(parts[3])
        );
    }
}
