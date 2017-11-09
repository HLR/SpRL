package edu.tulane.cs.hetml.vision;

import org.bytedeco.javacpp.presets.opencv_core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.awt.Rectangle;
/**
 * Reads Image Triplets , given a directory
 * @author Umar Manzoor
 *
 */
public class ImageTripletReader {

    public List<ImageTriplet> trainImageTriplets;
    public List<ImageTriplet> testImageTriplets;
    public ImageTripletReader() {

    }
    public ImageTripletReader(String directory) throws IOException {

        trainImageTriplets = new ArrayList<>();
        testImageTriplets = new ArrayList<>();
        loadFile(directory +  "/MSCOCO.originalterm.train", true);
        loadFile(directory +  "/MSCOCO.originalterm.test", false);
    }

    private void loadFile(String filepath, boolean isTrain) throws IOException {
        String file = filepath;
        File d = new File(file);

        if (d.exists()) {
            String line;
            BufferedReader reader = new BufferedReader(new FileReader(file));
            while ((line = reader.readLine()) != null) {
                String[] tuple = line.split("\\t");
                String sp = tuple[2];
                String tr = tuple[3];
                String lm = tuple[4];
                String trBox = tuple[9];
                String lmBox = tuple[10];
                String imageBox = tuple[11];

                double trArea = calculateArea(trBox, false);
                double lmArea = calculateArea(lmBox, false);
                double imageArea = calculateArea(imageBox, true);

                String boundingBox = generateBoundingBox(trBox, lmBox);
                double boundingBoxArea = calculateArea(boundingBox, false);

                //Feature 1
                String trVector = getCentroidVector(trBox, lmBox, boundingBox);

                //Feature 2
                double trAreawrtLM = trArea / lmArea;

                // Feature 3
                double trAspectRatio = calculateAspectRatio(trBox);
                double lmAspectRatio = calculateAspectRatio(lmBox);

                //Feature 4
                double trAreaBbox = normalizeArea(trArea, boundingBoxArea);
                double lmAreaBbox = normalizeArea(lmArea, boundingBoxArea);

                //Feature 5
                double iou = getIntersectionOverUnion(trBox, lmBox);

                // Feature 6
                double euclideanDistance = normalizeArea(getEuclideanDistance(trBox, lmBox), imageArea);

                // Feature 7
                double trAreaImage = normalizeArea(trArea, imageArea);
                double lmAreaImage = normalizeArea(lmArea, imageArea);

                ImageTriplet it = new ImageTriplet(sp, tr, lm, trBox, lmBox, imageBox, trVector, trAreawrtLM,
                        trAspectRatio, lmAspectRatio, trAreaBbox, lmAreaBbox, iou, euclideanDistance, trAreaImage, lmAreaImage);

                if(isTrain)
                    trainImageTriplets.add(it);
                else
                    testImageTriplets.add(it);
            }
        }
    }

    private String getCentroidVector(String trBox, String lmBox, String Boundingbox) {
        String[] trInfo = trBox.split("-");
        String[] lmInfo = trBox.split("-");


        double trCenterX, trCenterY, lmCenterX, lmCenterY;
        trCenterX = Double.parseDouble(trInfo[0]) + Double.parseDouble(trInfo[2]) * 0.5;
        trCenterY = Double.parseDouble(trInfo[1]) + Double.parseDouble(trInfo[3]) * 0.5;

        lmCenterX = Double.parseDouble(lmInfo[0]) + Double.parseDouble(lmInfo[2]) * 0.5;
        lmCenterY = Double.parseDouble(lmInfo[1]) + Double.parseDouble(lmInfo[3]) * 0.5;

        String[] boundingBoxInfo = Boundingbox.split("-");
        String vector = (trCenterX - lmCenterX) / Double.parseDouble(boundingBoxInfo[2]) + "-"
                + (trCenterY - lmCenterY) / Double.parseDouble(boundingBoxInfo[3]);
        return vector;
    }

    private double getIntersectionOverUnion(String trBox, String lmBox) {
        String[] trInfo = trBox.split("-");
        String[] lmInfo = lmBox.split("-");

        Rectangle tr = new Rectangle((int) Double.parseDouble(trInfo[0]), (int) Double.parseDouble(trInfo[1]), (int) Double.parseDouble(trInfo[2]), (int) Double.parseDouble(trInfo[3]));
        Rectangle lm = new Rectangle((int) Double.parseDouble(lmInfo[0]), (int) Double.parseDouble(lmInfo[1]), (int) Double.parseDouble(lmInfo[2]), (int) Double.parseDouble(lmInfo[3]));
        Rectangle intersection = tr.intersection(lm);
        Rectangle union = tr.union(lm);

        double intersectionArea = intersection.width*intersection.height;
        double unionArea = union.width*union.height;

        double iou = 0;
        if(unionArea!=0)
            iou = intersectionArea / unionArea;
        return iou;
    }

    private String generateBoundingBox(String trBox, String lmBox) {
        String[] trRect = trBox.split("-");
        double trX = Double.parseDouble(trRect[0]);
        double trY = Double.parseDouble(trRect[1]);
        double trWidth = Double.parseDouble(trRect[2]);
        double trHeight = Double.parseDouble(trRect[3]);

        String[] lmRect = lmBox.split("-");
        double lmX = Double.parseDouble(lmRect[0]);
        double lmY = Double.parseDouble(lmRect[1]);
        double lmWidth = Double.parseDouble(lmRect[2]);
        double lmHeight = Double.parseDouble(lmRect[3]);

        double minX, minY, maxX, maxY;

        // Minimum X
        if(trX > lmX)
            minX = lmX;
        else
            minX = trX;

        //Maximum X
        if(trX + trWidth > lmX + lmWidth)
            maxX = trX + trWidth;
        else
            maxX = lmX + lmWidth;

        //Minimum Y
        if(trY < lmY)
            minY = trY;
        else
            minY = lmY;

        //Minimum Y
        if (trY+trHeight<lmY+lmHeight)
            maxY = lmY + lmHeight;
        else
            maxY = trY + trHeight;

        double bBoxWidth = maxX - minX;
        double bBoxHeight = maxY - minY;

        String bBox = minX + "-" + minY + "-" + bBoxWidth + "-" + bBoxHeight;
        return bBox;
    }

    private double calculateArea(String box, boolean isImage) {
        String[] rect = box.split("-");
        double width;
        double height;
        if (isImage) {
            width = Double.parseDouble(rect[0]);
            height = Double.parseDouble(rect[1]);
        }
        else {
            width = Double.parseDouble(rect[2]);
            height = Double.parseDouble(rect[3]);
        }
        return width * height;
    }

    private double calculateAspectRatio(String box) {
        String[] rect = box.split("-");
        double width = Double.parseDouble(rect[2]);
        double height = Double.parseDouble(rect[3]);
        return width / height;
    }
    private double normalizeArea(double boxArea, double imageArea) {
        return boxArea / imageArea;
    }
    private double getEuclideanDistance(String trBox, String lmBox) {

        String[] trInfo = trBox.split("-");
        String[] lmInfo = lmBox.split("-");

        double x1 = Double.parseDouble(trInfo[0]);
        double y1 = Double.parseDouble(trInfo[1]);
        double x2 = Double.parseDouble(lmInfo[0]);
        double y2 = Double.parseDouble(lmInfo[1]);

        double  xDiff = x1-x2;
        double  xSqr  = Math.pow(xDiff, 2);

        double yDiff = y1-y2;
        double ySqr = Math.pow(yDiff, 2);

        double distance = Math.sqrt(xSqr + ySqr);

        return distance;
    }
}
