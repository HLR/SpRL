package edu.tulane.cs.hetml.vision;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;
import edu.tulane.cs.hetml.nlp.BaseTypes.Document;
import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader;
import edu.tulane.cs.hetml.vision.*;

import java.awt.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

/**
 * Reads CLEF Image dataset, given a directory
 * @author Umar Manzoor
 *
 */
public class CLEFImageReader {

    private final String trainFilePath;
    private final String testFilePath;
    private String path;
    private Boolean readFullData;
    private List<String> trainingData;
    private List<String> testData;

    public List<Image> trainingImages;
    public List<Segment> trainingSegments;
    public List<SegmentRelation> trainingRelations;

    public List<Image> testImages;
    public List<Segment> testSegments;
    public List<SegmentRelation> testRelations;

    public List<Segment> allSegments;

    public List<ImageTriplet> trainImageTriplets;
    public List<ImageTriplet> testImageTriplets;

    private Hashtable<Integer, String> MapCode2Concept = new Hashtable<Integer, String>();
    private Hashtable<String, String> segmentReferitText = new Hashtable<String, String>();
    private Hashtable<String, String> segmentOntology = new Hashtable<String, String>();
    private Hashtable<String, String> redefindedRelations = new Hashtable<String, String>();
    private Hashtable<String, String> segmentBoxes = new Hashtable<String, String>();

    private double imageWidth, imageHeight;
    PrintWriter printWriterTest;

    public CLEFImageReader(String directory, String trainFilePath, String testFilePath, Boolean readFullData) throws IOException {

        this.trainFilePath = trainFilePath;
        this.testFilePath = testFilePath;
        File d = new File(directory);

        if (!d.exists()) {
            throw new IOException(directory + " does not exist!");
        }

        if (!d.isDirectory()) {
            throw new IOException(directory + " is not a directory!");
        }
        imageHeight = 360;
        imageWidth = 480;
        trainingData = new ArrayList<>();
        testData = new ArrayList<>();

        this.readFullData = readFullData;

        // Training Data
        trainingImages = new ArrayList<>();
        trainingSegments = new ArrayList<>();
        trainingRelations = new ArrayList<>();
        trainImageTriplets = new ArrayList<>();
        // Test Data
        testImages = new ArrayList<>();
        testSegments = new ArrayList<>();
        testRelations = new ArrayList<>();
        testImageTriplets = new ArrayList<>();

        // all Segment
        allSegments = new ArrayList<>();

        path = directory;
        // Load segment Boxes Information
        getSegmentsBox(directory);
        // Load redefined segment relations
        getRedefinedRelations(directory);
        // Load Concepts
        getConcepts(directory);
        //Load Referit Data
//        getReferitText(directory);
        // Load Training
        getTrainingImages();
        // Load Testing
        getTestImages();
        // Load all Images
        getallImages(directory);
        // Generate Visual Triplet Pairs
        generateVisualTripletSegmentPairs();
        // Save to File
        //printImageInformation();

        System.out.println("Total Train Data " + trainingData.size());

        System.out.println("Total Test Data " + testData.size());

        System.out.println("Total Train Images " + trainingImages.size());

        System.out.println("Total Test Images " + testImages.size());

        System.out.println("Total Train Segments " + trainingSegments.size());

        System.out.println("Total Test Segments " + testSegments.size());

        System.out.println("Total Train Relations " + trainingRelations.size());

        System.out.println("Total Test Relations " + testRelations.size());

    }

    /*****************************************/
    // Takes object code as input and returns
    // object concept

    /*****************************************/
    private String MappingCode2Concept(int code) {
        return MapCode2Concept.get(code);
    }

    /*******************************************************/
    // Loading Image Codes and its Corresponding Concept
    // Storing information in HashTable for quick retrieval

    /*******************************************************/
    private void getConcepts(String directory) throws IOException {
        String file = directory + "/wlist.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] CodesInfo = line.split("\\t");
            if (CodesInfo.length > 1) {
                MapCode2Concept.put(Integer.parseInt(CodesInfo[0]), CodesInfo[1]);
            } else {
                MapCode2Concept.put(Integer.parseInt(CodesInfo[0]), " ");
            }
        }
    }

    /*******************************************************/
    // Loading Referit Text for CLEF Segments
    // Storing information in HashTable for quick retrieval
    /*******************************************************/
    private void getReferitText(String directory) throws IOException {
        String file = directory + "/ReferGames.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] segReferitText = line.split("\\~");
            if (segReferitText.length > 1) {
                segmentReferitText.put(segReferitText[0], segReferitText[1]);
            } else {
                segmentReferitText.put(segReferitText[0], " ");
            }
        }
    }

    /*******************************************************/
    // Loading Segment Box Information for CLEF Segments
    // Storing information in HashTable for quick retrieval
    /*******************************************************/
    private void getSegmentsBox(String directory) throws IOException {
        String file = directory + "/SegmentBoxes/segmentBoxes.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] segBoxInfo = line.split(" ");
            String key = segBoxInfo[0] + "-" + segBoxInfo[1];
            segmentBoxes.put(key, segBoxInfo[2]);
        }
    }


    private void getRedefinedRelations(String directory) throws IOException {
        directory = directory + "/relations";
        File d = new File(directory);

        if (!d.exists()) {
            throw new IOException(directory + " does not exist!");
        }

        if (!d.isDirectory()) {
            throw new IOException(directory + " is not a directory!");
        }
        for (File f : d.listFiles()) {
            if (f.isFile()) {
                String name = f.getName();
                String imageFile = directory + "/" + name;
                String[] fileName = name.split("-");
                String[] imageId = fileName[1].split("\\.");
                String line;
                BufferedReader reader = new BufferedReader(new FileReader(imageFile));
                while ((line = reader.readLine()) != null) {
                    String[] relationInfo = line.split(",");
                    String key = (imageId[0]).trim() + "-" + (relationInfo[1]).trim() + "-" + (relationInfo[3]).trim() + "-" + (relationInfo[5]).trim();
                    redefindedRelations.put(key, relationInfo[0]);
                }
                reader.close();
            }
        }
    }

    /*******************************************************/
    // Load all Images in the CLEF Dataset

    /*******************************************************/
    private void getallImages(String directory) throws IOException {
        File d = new File(directory);

        if (!d.exists()) {
            throw new IOException(directory + " does not exist!");
        }

        if (!d.isDirectory()) {
            throw new IOException(directory + " is not a directory!");
        }

        for (File f : d.listFiles()) {
            if (f.isDirectory()) {

                String mainFolder = directory + "/" + f.getName();
                System.out.println(mainFolder);
                //Load all images
                String imageFolder = mainFolder + "/images";
                getImages(imageFolder);

                //Load all segments
                String ontologyfile = mainFolder + "/ontology_path.txt";
                getSegmentsOntology(ontologyfile);

                //Load all segments
                String file = mainFolder + "/features.txt";
                getSegments(file);

                //Load all relations
                String spatialRelations = mainFolder + "/spatial_rels";
                getSegmentsRelations(spatialRelations);
            }
        }
    }

    /*******************************************************/
    // Loading Images

    /*******************************************************/
    private void getImages(String folder) {
        File d = new File(folder);

        if (d.exists()) {

            for (File f : d.listFiles()) {
                String label = f.getName();
                String[] split = label.split("\\.");

                if (trainingData.contains(split[0]))
                    trainingImages.add(new Image(label, split[0], imageWidth, imageHeight));
                if (testData.contains(split[0]))
                    testImages.add(new Image(label, split[0], imageWidth, imageHeight));
            }
        }
    }
    /*******************************************************/
    // Loading Segments
    /*******************************************************/
    private void getSegments(String file) throws IOException {
        File d = new File(file);

        if (d.exists()) {

            String line;
            BufferedReader reader = new BufferedReader(new FileReader(file));
            while ((line = reader.readLine()) != null) {
                String[] segmentInfo = line.split("\\t");
                if (segmentInfo.length == 4) {
                    String imageId = segmentInfo[0];
                    int segmentId = Integer.parseInt(segmentInfo[1]);
                    int segmentCode = Integer.parseInt(segmentInfo[3]);
                    String segmentConcept = MappingCode2Concept(segmentCode);

                    if (segmentConcept != null) {
                        String segmentFeatures = segmentInfo[2];
                        segmentFeatures = segmentFeatures.trim().replaceAll(" +", " ");
                        if (trainingData.contains(imageId)) {
                            String key = imageId + "-" + segmentId;
                            trainingSegments.add(new Segment(imageId, segmentId, segmentCode, segmentFeatures, segmentConcept, segmentBoxes.get(key)));
                            allSegments.add(new Segment(imageId, segmentId, segmentCode, null, segmentConcept, segmentBoxes.get(key)));
                        }
                        if (testData.contains(imageId)) {
                            String key = imageId + "-" + segmentId;
                            testSegments.add(new Segment(imageId, segmentId, segmentCode, segmentFeatures, segmentConcept, segmentBoxes.get(key)));
                            allSegments.add(new Segment(imageId, segmentId, segmentCode, null, segmentConcept, segmentBoxes.get(key)));
                        }
                    }
                }
            }
            reader.close();
        }
    }

    /*******************************************************/
    // Loading Segment Relations

    /*******************************************************/
    private void getSegmentsRelations(String spatial_rels) throws IOException {
        File d = new File(spatial_rels);

        if (d.exists()) {
            for (File f : d.listFiles()) {
                String spatial_file = spatial_rels + "/" + f.getName();
                MatFileReader matFileReader = new MatFileReader(spatial_file);
                int val;
                int firstSegmentId;
                int secondSegmentId;
                String[] s = f.getName().split("\\.");
                String imgId = s[0];
                String rel;
                double[][] topo = ((MLDouble) matFileReader.getMLArray("topo")).getArray();
                double[][] xRels = ((MLDouble) matFileReader.getMLArray("x_rels")).getArray();
                double[][] yRels = ((MLDouble) matFileReader.getMLArray("y_rels")).getArray();
                /**************************************************/
                // Exemptional case
                // Sometimes mat file is returning only one value
                /**************************************************/
                if (topo.length > 1) {
                    // Finding Relationships
                    for (int x = 0; x < topo[0].length; x++)
                        for (int y = 0; y < topo[1].length; y++) {
                            //Ignoring same indexes
                            if (x != y) {
                                firstSegmentId = x + 1;
                                secondSegmentId = y + 1;
                                val = (int) topo[x][y];
                                if (val == 1)
                                    rel = "adjacent";
                                else if (val == 2)
                                    rel = "disjoint";
                                else
                                    rel = null;

                                if (rel != null) {
                                    if (trainingData.contains(imgId)) {
                                        //Creating new Relation between segments
                                        trainingRelations.add(new SegmentRelation(imgId, firstSegmentId, secondSegmentId, rel));
                                    }
                                    if (testData.contains(imgId)) {
                                        testRelations.add(new SegmentRelation(imgId, firstSegmentId, secondSegmentId, rel));
                                    }
                                }
                                val = (int) xRels[x][y];
                                if (val == 3)
                                    rel = "beside";
                                else if (val == 4) {
                                    // Original "x-aligned"
                                    //rel = "x-aligned";
                                    String key = imgId + "-" + firstSegmentId + "-" + secondSegmentId + "-" + "x-aligned";
                                    rel = redefindedRelations.get(key);
                                }

                                if (rel != null) {
                                    if (trainingData.contains(imgId)) {
                                        //Creating new Relation between segments
                                        trainingRelations.add(new SegmentRelation(imgId, firstSegmentId, secondSegmentId, rel));
                                    }
                                    if (testData.contains(imgId)) {
                                        testRelations.add(new SegmentRelation(imgId, firstSegmentId, secondSegmentId, rel));
                                    }
                                }

                                val = (int) yRels[x][y];
                                if (val == 5)
                                    rel = "above";
                                else if (val == 6)
                                    rel = "below";
                                else if (val == 7) {
                                    // Original "y-aligned"
                                    //rel= "y-aligned";
                                    String key = imgId + "-" + firstSegmentId + "-" + secondSegmentId + "-" + "x-aligned";
                                    rel = redefindedRelations.get(key);
                                }

                                if (rel != null) {
                                    if (trainingData.contains(imgId)) {
                                        //Creating new Relation between segments
                                        trainingRelations.add(new SegmentRelation(imgId, firstSegmentId, secondSegmentId, rel));
                                    }
                                    if (testData.contains(imgId)) {
                                        testRelations.add(new SegmentRelation(imgId, firstSegmentId, secondSegmentId, rel));
                                    }
                                }
                            }
                        }
                }
            }
        }
    }

    /*******************************************************/
    // Loading Training Images

    /*******************************************************/
    private void getTrainingImages() throws IOException {
        if (readFullData) {
            getMatData(path + "/training.mat", true, "training");
            getMatData(path + "/validation.mat", true, "validation");;
        } else {
            getXMLImages(trainFilePath, true);
        }
    }

    /*******************************************************/
    // Loading Testing Images

    /*******************************************************/
    private void getTestImages() throws IOException {
        if (readFullData) {
            getMatData(path + "/testing.mat", false, "testing");
        } else {
            getXMLImages(testFilePath, false);
        }
    }

    /*******************************************************/
    // Loading data from XML file
    // if choose = true, trainData will be populated
    // if choose = false, testData will be populated

    /*******************************************************/
    private void getXMLImages(String file, Boolean choose) throws IOException {

        File f = new File(file);

        if (!f.exists()) {
            throw new IOException(file + " does not exist!");
        }
        NlpXmlReader reader = new NlpXmlReader(file, "SCENE", "SENTENCE", null, null);
        reader.setIdUsingAnotherProperty("SCENE", "DOCNO");

        List<Document> documentList = reader.getDocuments();

        for (Document d : documentList) {
            String name = d.getPropertyFirstValue("IMAGE");
            String s = name.substring(name.lastIndexOf("/") + 1);
            String[] label = s.split("\\.");
            if (choose)
                trainingData.add(label[0]);
            else
                testData.add(label[0]);
        }
    }
    /*******************************************************/
    // Loading data from Mat file
    // if choose = true, trainData will be populated
    // if choose = false, testData will be populated

    /*******************************************************/
    private void getMatData(String file, Boolean choose, String name) throws IOException {

        File f = new File(file);

        if (!f.exists()) {
            throw new IOException(file + " does not exist!");
        }
        MatFileReader matReader = new MatFileReader(file);

        double[][] data;

        if (choose) {
            data = ((MLDouble) matReader.getMLArray(name)).getArray();
        }
        else
            data = ((MLDouble) matReader.getMLArray(name)).getArray();

        if (data.length > 1) {
            for (int i = 0; i < data.length; i++) {
                int imageId = (int) data[i][0];
                if (choose)
                    trainingData.add(Integer.toString(imageId));
                else
                    testData.add(Integer.toString(imageId));
            }
        }
    }

    /*******************************************************/
    // Loading Segments Ontology
    /*******************************************************/
    private void getSegmentsOntology(String file) throws IOException {

        File d = new File(file);

        if (d.exists()) {

            String line;
            BufferedReader reader = new BufferedReader(new FileReader(file));
            while ((line = reader.readLine()) != null) {
                String[] segmentOntologyInfo = line.split("\\t");
                if (segmentOntologyInfo.length == 3) {
                    String key = segmentOntologyInfo[0] + "-" + Integer.parseInt(segmentOntologyInfo[1]);
                    segmentOntology.put(key, segmentOntologyInfo[2].replaceAll("_", ""));
                }
            }
        }
    }

    /*******************************************************/
    // Loading Segments Ontology
    /*******************************************************/
    private void generateVisualTripletSegmentPairs() {
        generateVisualTripletsSegmentPairs(trainingImages, trainingSegments, true);
        generateVisualTripletsSegmentPairs(testImages, testSegments, false);
    }

    private void generateVisualTripletsSegmentPairs(List<Image> images, List<Segment> segments, boolean train) {
        for (Image i : images) {
            List<Segment> temp = new ArrayList<>();
            for (Segment s : segments) {
                if (i.getId().equals(s.getAssociatedImageID()))
                    temp.add(s);
            }

            // Generate Pairs
            for (int j =0; j<temp.size(); j++)
                for (int k =0; k<temp.size();k++) {
                    Segment trSeg = temp.get(j);
                    Segment lmSeg = temp.get(k);
                    if(trSeg.getSegmentId()!=lmSeg.getSegmentId()) { // Ignore same index
                        if (train)
                            trainImageTriplets.add(generateImageTriplet(i, trSeg, lmSeg));
                        else
                            testImageTriplets.add(generateImageTriplet(i, trSeg, lmSeg));
                    }
                }
        }
    }
    private ImageTriplet generateImageTriplet(Image i, Segment trSeg, Segment lmSeg) {

        String trBox = trSeg.getBoxDimensions();
        String lmBox = lmSeg.getBoxDimensions();

        double trArea = calculateArea(trBox);
        double lmArea = calculateArea(lmBox);
        double imageArea = imageWidth * imageHeight;

        String boundingBox = generateBoundingBox(trBox, lmBox);
        double boundingBoxArea = calculateArea(boundingBox);

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
        String imageBox = imageWidth + "-" + imageHeight;

        return new ImageTriplet(i.getId(), trSeg.getSegmentId(), lmSeg.getSegmentId(), trBox, lmBox, imageBox, trVector, trAreawrtLM,
                trAspectRatio, lmAspectRatio, trAreaBbox, lmAreaBbox, iou, euclideanDistance, trAreaImage, lmAreaImage);
    }

    private void printImageInformation() throws IOException {

        String path = "data/mSpRL/results/allImageSegments.txt";
        printWriterTest = new PrintWriter(path);

        for (Image i : testImages) {
            for (Segment s : testSegments) {
                if (i.getId().equals(s.getAssociatedImageID()))
                    printWriterTest.println(i.getId() + " " + s.getSegmentId());
            }
        }

        for (Image i : trainingImages) {
            for (Segment s : trainingSegments) {
                if (i.getId().equals(s.getAssociatedImageID()))
                    printWriterTest.println(i.getId() + " " + s.getSegmentId());
            }
        }
        printWriterTest.close();
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

    private double calculateArea(String box) {
        String[] rect = box.split("-");
        double width;
        double height;
        width = Double.parseDouble(rect[2]);
        height = Double.parseDouble(rect[3]);
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
