package edu.tulane.cs.hetml.vision;

import edu.tulane.cs.hetml.nlp.BaseTypes.Document;
import edu.tulane.cs.hetml.nlp.Xml.NlpXmlReader;
import sun.awt.image.ToolkitImage;

import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

public class CLEFNewSegmentCNNFeaturesReader {

    public List<Segment> clefSegments;
    public List<Segment> clefUniqueSegments;
    private List<String> xmlData;

    public List<Image> clefImages;

    public Hashtable<String, String> clefHeadwordSegment = new Hashtable<String, String>();
    public List<SegmentPhraseHeadwordPair> segmentPhraseHeadwordPair;

    private Hashtable<String, Rectangle2D> segmentBoxes = new Hashtable<String, Rectangle2D>();

    public CLEFNewSegmentCNNFeaturesReader() {

    }

    public void loadFeatures(String directory, String xmlFile, boolean isTrain) throws IOException {
        clefSegments = new ArrayList<>();
        clefUniqueSegments = new ArrayList<>();
        segmentPhraseHeadwordPair = new ArrayList<>();

        xmlData = new ArrayList<>();
        clefImages = new ArrayList<>();

        getXMLImages(xmlFile, isTrain);
        getallImages(directory);
        getSegmentsBox(directory);
        loadPhraseText(directory + "/PhraseSegmentPairs/", isTrain);
        loadNewSegmentFeatures(directory + "/SegmentCNNFeatures/", isTrain);
    }

    private void loadPhraseText(String directory, boolean isTrain) throws IOException {
        String file;
        if(isTrain) {
            file = directory + "SegmentsPhraseText_train_head.txt";
        } else {
            file = directory + "SegmentsPhraseText_test_head.txt";
        }

        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] segInfo = line.split("~");
            if (segInfo.length>=3) {
                segmentPhraseHeadwordPair.add(new SegmentPhraseHeadwordPair(segInfo[0], segInfo[1], segInfo[2]));
                clefHeadwordSegment.put(segInfo[0] + "-" + segInfo[1], segInfo[2]);
            }
            else {
                segmentPhraseHeadwordPair.add(new SegmentPhraseHeadwordPair(segInfo[0], segInfo[1], ""));
                clefHeadwordSegment.put(segInfo[0] + "-" + segInfo[1], "");
            }
        }
    }

    private void loadNewSegmentFeatures(String directory, boolean isTrain) throws IOException{
        String file;
        if(isTrain) {
            file = directory + "ImageSegmentsFeaturesNewTrain.txt";
            loadClefNewSegments(file);
        } else {
            file = directory + "ImageSegmentsFeaturesNewTest.txt";
            loadClefNewSegments(file);
        }
    }

    private void loadClefNewSegments(String file) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] segInfo = line.split(",");
            findSegmentPhrasePairs(segInfo[0], segInfo[1], segInfo[2]);
            String key = segInfo[0] + "-" + segInfo[1];
            Segment s = new Segment(segInfo[0], Integer.parseInt(segInfo[1]), segInfo[2], clefHeadwordSegment.get(key));
            s.setBoxDimensions(segmentBoxes.get(key));
            clefUniqueSegments.add(s);
        }
    }

    private void findSegmentPhrasePairs(String imageId, String segmentId, String segmentFeatures) {
        for(int i = 0; i < segmentPhraseHeadwordPair.size(); i++) {
            SegmentPhraseHeadwordPair sp = segmentPhraseHeadwordPair.get(i);
            if(sp.getImageId().equalsIgnoreCase(imageId) && sp.getSegmentId().equalsIgnoreCase(segmentId)) {
                Segment s = new Segment(sp.getImageId(), Integer.parseInt(sp.getSegmentId()), segmentFeatures,
                        sp.getPhraseHeadword());
                String key = sp.getImageId() + sp.getSegmentId();
                s.setBoxDimensions(segmentBoxes.get(key));
                clefSegments.add(s);
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
            Rectangle2D rec = RectangleHelper.parseRectangle(segBoxInfo[2], "-");
            segmentBoxes.put(key, rec);
        }
    }
    /*******************************************************/
    // Loading Images
    /*******************************************************/
    private void getImages(String folder) throws IOException {
        File d = new File(folder);

        if (d.exists()) {

            for (File f : d.listFiles()) {
                String label = f.getName();
                String[] split = label.split("\\.");
                ToolkitImage image = (ToolkitImage) Toolkit.getDefaultToolkit().getImage(f.getAbsolutePath());
                int width = image.getWidth();
                int height = image.getHeight();
                if (xmlData.contains(split[0]))
                    clefImages.add(new Image(label, split[0], width, height));
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
            }
        }
    }

    /*******************************************************/
    // Loading data from XML file
    /*******************************************************/
    private void getXMLImages(String file, Boolean isTrain) throws IOException {

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
            xmlData.add(label[0]);
        }
    }

}
