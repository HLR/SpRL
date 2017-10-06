package edu.tulane.cs.hetml.vision;

import org.bytedeco.javacpp.presets.opencv_core;

import java.io.*;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

public class CLEFAnnotationReader {

    private Hashtable<String, String> tokens = new Hashtable<String, String>();
    private Hashtable<String, String> pharseRemaining = new Hashtable<String, String>();
    private Hashtable<String, String> referit = new Hashtable<String, String>();

    public List<Image> testImages;
    public List<Segment> testSegments;

    PrintWriter printToFile;
    PrintWriter printToFileNames;

    public CLEFAnnotationReader(String directory) throws IOException {

        File d = new File(directory);

        if (!d.exists()) {
            throw new IOException(directory + " does not exist!");
        }

        if (!d.isDirectory()) {
            throw new IOException(directory + " is not a directory!");
        }

        testImages = new ArrayList<>();
        testSegments = new ArrayList<>();

        //Load Referit Text
        loadReferit(directory);

        //Annotated File Conversion
        String annDir = directory + "/annotatedFiles";
        annotatedFilesConversion(annDir);

        //Load Test Images
        loadTestImage(annDir);

        //Load Test Segments
        loadTestSegment(annDir);
    }

    private void annotatedFilesConversion(String directory) throws IOException {

        printToFile = new PrintWriter( directory + "/Output/ClefSegment.txt");
        printToFileNames = new PrintWriter( directory + "/Output/ClefImage.txt");

        File folder = new File(directory);
        File[] listOfFiles = folder.listFiles();

        for (File file : listOfFiles) {
            if (file.isFile()) {
                String fileName = file.getName().replaceFirst("[.][^.]+$", "");
                printToFileNames.println(fileName);
                String line;
                BufferedReader reader = new BufferedReader(new FileReader(file));
                while ((line = reader.readLine()) != null) {
                    String[] words = line.split("\\t");

                    if (words[0].startsWith("T")) { // Tokens
                        if (words[1].startsWith("P")) { // Phrase
                            tokens.put(words[0], words[2]);
                            pharseRemaining.put(words[0], words[2]);
                        }
                        else if (words[1].startsWith("S")) //Segment
                            tokens.put(words[0], words[2]);
                    } else if (words[0].startsWith("R")) { // Relations
                        String[] relTokens = words[1].split(" ");
                        String[] arg1 = relTokens[1].split(":");
                        String[] arg2 = relTokens[2].split(":");
                        String arg1Phrase = tokens.get(arg1[1].trim());
                        pharseRemaining.remove(arg1[1].trim());
                        String arg2Segment = tokens.get(arg2[1].trim());
                        String[] segCodeText = arg2Segment.split(" ");

                        // Referit Key
                        String referitKey = fileName + "_" + segCodeText[0] + ".jpg";
                        String[] oldData = referit.get(referitKey).split("~");

                        // Our Text in Referit
                        String newData = fileName + "~" + segCodeText[0] + "~" + arg1Phrase + "~" + oldData[1] + "~" + oldData[2];
                        //Save the new generated data to file
                        printToFile.println(newData);
                    }
                }
                for(String s : pharseRemaining.keySet()) {
                    // Writing Remaining Phrases
                    // 0 index doesn't exists
                    String referitKey = fileName + "_0.jpg";

                    String newData = fileName + "~" + "0" + "~" + pharseRemaining.get(s) + "~0.0"  + "~0.0";
                    //Save the new generated data to file
                    printToFile.println(newData);
                }
            }
        }
        printToFile.close();
        printToFileNames.close();
    }
    private void loadTestImage(String directory) throws IOException {
        String file = directory + "/Output/ClefImage.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String imageId = line.trim();
            Image i = new Image(imageId.trim(), imageId.trim());
            testImages.add(i);

        }
    }

    private void loadTestSegment(String directory) throws IOException {
        String file = directory + "/Output/ClefSegment.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] segInfo = line.split("\\~");
            Segment s = new Segment(segInfo[0], Integer.parseInt(segInfo[1]),"",segInfo[2]);
            testSegments.add(s);
        }
    }

    private void loadReferit(String directory) throws IOException {
        String file = directory + "/ReferGames.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] segInfo = line.split("\\~");
            referit.put(segInfo[0], segInfo[1] + "~" + segInfo[2] + "~" + segInfo[3]);
        }
     }
}
