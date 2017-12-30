package edu.tulane.cs.hetml.vision;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

public class CLEFNewSegmentCNNFeaturesReader {

    public List<Segment> clefSegments;

    public Hashtable<String, String> clefHeadwordSegment = new Hashtable<String, String>();
    public CLEFNewSegmentCNNFeaturesReader() {

    }

    public void loadFeatures(String directory, boolean isTrain) throws IOException {
        clefSegments = new ArrayList<>();
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
            if (segInfo.length>=3)
                clefHeadwordSegment.put(segInfo[0] + "-" + segInfo[1], segInfo[2]);
            else
                clefHeadwordSegment.put(segInfo[0] + "-" + segInfo[1], "");
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
            String key = segInfo[0] + "-" + segInfo[1];
            if(clefHeadwordSegment.get(key)!=null) {
                Segment s = new Segment(segInfo[0], Integer.parseInt(segInfo[1]), segInfo[2],
                        clefHeadwordSegment.get(key));
                clefSegments.add(s);
            }
        }
    }
}
