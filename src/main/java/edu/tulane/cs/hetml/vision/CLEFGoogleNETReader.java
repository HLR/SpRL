package edu.tulane.cs.hetml.vision;

import edu.tulane.cs.hetml.nlp.BaseTypes.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.*;

/**
 * Reads CLEF Segment CNN Features, given a directory
 * @author Umar Manzoor
 *
 */
public class CLEFGoogleNETReader {

    public List<Image> allImages;
    public List<Image> trainImages;
    public List<Image> testImages;
    public List<Segment> allSegments;
    public Hashtable<String, String> segRefExp = new Hashtable<String, String>();

    public List<Document> allDocuments;
    public List<Sentence> allSentences;

    public CLEFGoogleNETReader(String directory) throws IOException {

        File d = new File(directory);
        if (!d.exists()) {
            throw new IOException(directory + " does not exist!");
        }

        if (!d.isDirectory()) {
            throw new IOException(directory + " is not a directory!");
        }

        allImages = new ArrayList<>();
        trainImages = new ArrayList<>();
        testImages = new ArrayList<>();
        allSegments = new ArrayList<>();
        allDocuments = new ArrayList<>();
        allSentences = new ArrayList<>();

        loadImages(directory + "/W2CtrainImages.txt", true);

        loadImages(directory + "/W2CtestImages.txt", false);

        getReferitText(directory);

        getFeatures(directory);

        generateNLPBaseClasses();

    }

    private void loadImages(String file, Boolean isTrain) throws IOException {
        File f = new File(file);
        if (f.exists()) {
            String line;
            BufferedReader reader = new BufferedReader(new FileReader(file));
            line = reader.readLine();
            String[] imageIDs = (line.trim()).split(",");
            for (String id : imageIDs) {
                Image i = new Image(id.trim(), id.trim());
                if (isTrain)
                    trainImages.add(i);
                else
                    testImages.add(i);
            }
            reader.close();
        }
    }

    private void getFeatures(String directory) throws IOException {
        String file = directory + "/ImageSegmentsFeatures.txt";
        File f = new File(file);
        if (f.exists()) {
            String line;
            BufferedReader reader = new BufferedReader(new FileReader(file));
            while ((line = reader.readLine()) != null) {
                String[] row = (line.trim()).split(",");
                Image i = new Image(row[0].trim(), row[0].trim());
                allImages.add(i);
                String key = row[0].trim() + "_" + row[1].trim() + ".jpg";
//                if(segRefExp.get(key)!=null) {
                    Segment s = new Segment(row[0].trim(), Integer.parseInt(row[1].trim()), row[2].trim(), segRefExp.get(key), true);
                    allSegments.add(s);
//                }
            }
            reader.close();
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
                String exp = removeDuplicates(segReferitText[1]);
                segRefExp.put(segReferitText[0], exp);

            } else {
                segRefExp.put(segReferitText[0], " ");
            }
        }
    }

    public void generateNLPBaseClasses() {
        for (Segment s :  allSegments) {
            String ID = s.getAssociatedImageID() + "_" + s.getSegmentId();
            Document d = new Document(ID);
            int len = 0;
            if(s.referItExpression!=null)
                len = s.referItExpression.length();
            Sentence sen = new Sentence(d, ID, 0, len, s.referItExpression);
            allDocuments.add(d);
            allSentences.add(sen);
        }
    }

    public String removeDuplicates(String s) {
        s = s.toLowerCase().replaceAll("[^a-z]", " ").replaceAll("( )+", " ").trim();
        return new LinkedHashSet<String>(Arrays.asList(s.split(" "))).toString().replaceAll("(^\\[|\\]$)", "").replace(", ", " ");
    }
}
