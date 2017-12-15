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

                ImageTriplet it = new ImageTriplet(sp, tr, lm, trBox, lmBox, imageWidth, imageHeight);

                if(isTrain)
                    trainImageTriplets.add(it);
                else
                    testImageTriplets.add(it);
            }
        }
    }
}
