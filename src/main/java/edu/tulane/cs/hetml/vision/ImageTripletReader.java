package edu.tulane.cs.hetml.vision;

import java.awt.geom.Rectangle2D;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
    public ImageTripletReader(String directory, String fileName) throws IOException {

        trainImageTriplets = new ArrayList<>();
        testImageTriplets = new ArrayList<>();

        loadFile(directory +  "/" + fileName + ".train", true);
        loadFile(directory +  "/" + fileName + ".test", false);
    }

    private void loadFile(String filepath, boolean isTrain) throws IOException {
        String file = filepath;
        File d = new File(file);

        if (d.exists()) {
            String line;
            BufferedReader reader = new BufferedReader(new FileReader(file));
            while ((line = reader.readLine()) != null) {
                String[] tuple = line.replaceAll("\\t\\t", "\t").split("\\t");
                String sp = tuple[2];
                String tr = tuple[3];
                String lm = tuple[4];
                Rectangle2D trBox = RectangleHelper.parseRectangle(tuple[9], "-");
                Rectangle2D lmBox = RectangleHelper.parseRectangle(tuple[10], "-");
                double imageWidth = Double.parseDouble(tuple[11].split("-")[0]);
                double imageHeight = Double.parseDouble(tuple[11].split("-")[1]);

                double trArea = RectangleHelper.calculateArea(trBox);
                double lmArea = RectangleHelper.calculateArea(lmBox);
                double imageArea = imageHeight * imageWidth;

                Rectangle2D boundingBox = RectangleHelper.generateBoundingBox(trBox, lmBox);
                double boundingBoxArea = RectangleHelper.calculateArea(boundingBox);

                //Feature 1
                double[] trVector = RectangleHelper.getCentroidVector(trBox, lmBox, boundingBox);

                //Feature 2
                double trAreawrtLM = trArea / lmArea;

                // Feature 3
                double trAspectRatio = RectangleHelper.calculateAspectRatio(trBox);
                double lmAspectRatio = RectangleHelper.calculateAspectRatio(lmBox);

                //Feature 4
                double trAreaBbox = RectangleHelper.normalizeArea(trArea, boundingBoxArea);
                double lmAreaBbox = RectangleHelper.normalizeArea(lmArea, boundingBoxArea);

                //Feature 5
                double iou = RectangleHelper.getIntersectionOverUnion(trBox, lmBox);

                // Feature 6
                double euclideanDistance = RectangleHelper
                        .normalizeArea(RectangleHelper.getEuclideanDistance(trBox, lmBox), imageArea);

                // Feature 7
                double trAreaImage = RectangleHelper.normalizeArea(trArea, imageArea);
                double lmAreaImage = RectangleHelper.normalizeArea(lmArea, imageArea);

                //Feature 8
                double above = RectangleHelper.getAbove(trBox, lmBox, imageHeight);

                //Feature 9
                double below = RectangleHelper.getBelow(trBox, lmBox, imageHeight);

                //Feature 10
                double left = RectangleHelper.getLeft(trBox, lmBox, imageWidth);

                //Feature 11
                double right = RectangleHelper.getRight(trBox, lmBox, imageWidth);

                ImageTriplet it = new ImageTriplet(sp, tr, lm, trBox, lmBox, imageWidth, imageHeight, trVector[0],
                        trVector[1], trAreawrtLM, trAspectRatio, lmAspectRatio, trAreaBbox, lmAreaBbox, iou,
                        euclideanDistance, trAreaImage, lmAreaImage, above, below, left, right);

                if(isTrain)
                    trainImageTriplets.add(it);
                else
                    testImageTriplets.add(it);
            }
        }
    }
}
