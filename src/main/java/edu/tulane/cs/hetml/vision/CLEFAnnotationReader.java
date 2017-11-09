package edu.tulane.cs.hetml.vision;

/**
 * Reads CLEF Annotated dataset, given a directory
 * @author Umar Manzoor
 *
 */
import edu.tulane.cs.hetml.nlp.BaseTypes.Document;
import edu.tulane.cs.hetml.nlp.BaseTypes.Sentence;
import java.io.*;
import java.util.*;

public class CLEFAnnotationReader {

    private Hashtable<String, String> tokens = new Hashtable<String, String>();
    private Hashtable<String, String> pharseRemaining = new Hashtable<String, String>();
    private Hashtable<String, String> referit = new Hashtable<String, String>();

    public List<Image> clefImages;
    public List<Segment> clefSegments;
    private List<String> allPhrases;
    public List<Document> clefDocuments;
    public List<Sentence> clefSentences;

    PrintWriter printToFile;
    PrintWriter printToFileNames;

    public CLEFAnnotationReader(String directory) throws IOException {
        loadAnnotatedFiles(directory);
    }

    private void loadAnnotatedFiles(String directory) throws IOException {

        File d = new File(directory);

        if (!d.exists()) {
            throw new IOException(directory + " does not exist!");
        }

        if (!d.isDirectory()) {
            throw new IOException(directory + " is not a directory!");
        }

        clefImages = new ArrayList<>();
        clefSegments = new ArrayList<>();
        allPhrases = new ArrayList<>();
        clefSentences = new ArrayList<>();
        clefDocuments = new ArrayList<>();

        //Load Referit Text
        loadReferit(directory);

        //Annotated File Conversion
        String annDir = directory + "/annotatedFiles";
        annotatedFilesConversion(annDir);

        //Load Test Images
        loadTestImage(annDir);

        //Load Test Segments
        loadTestSegment(annDir);

        generateNLPBaseClasses();
    }
    private void annotatedFilesConversion(String directory) throws IOException {

        printToFile = new PrintWriter( directory + "/Output/ClefSegment.txt");
        printToFileNames = new PrintWriter( directory + "/Output/ClefImage.txt");

        File folder = new File(directory);
        File[] listOfFiles = folder.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return !name.equals(".DS_Store");
            }
        });

        for (File file : listOfFiles) {
            if (file.isFile()) {
                String fileName = file.getName().replaceFirst("[.][^.]+$", "");
                // Reading Text used for Annotated
                allPhrases.clear();
                tokens.clear();
                pharseRemaining.clear();
                completeText(directory, fileName);
                printToFileNames.println(fileName);
                String line;
                BufferedReader reader = new BufferedReader(new FileReader(file));
                while ((line = reader.readLine()) != null) {
                    String[] words = line.split("\\t");

                    if (words[0].startsWith("T")) { // Tokens
                        if (words[1].startsWith("P")) { // Phrase
                            tokens.put(words[0], words[2]);
                            String[] spans = words[1].split(" ");
                            String pharseWithSpan = words[2] + " " + spans[1] + " " + spans [2];
                            allPhrases.remove(pharseWithSpan);
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
                        String newData;
                        if(referit.get(referitKey)!=null) {
                            String[] oldData = referit.get(referitKey).split("~");
                            // Our Text in Referit
                            newData = fileName + "~" + segCodeText[0] + "~" + arg1Phrase + "~" + oldData[1] + "~" + oldData[2];
                        } else {
                            //Segment Description doesn't exist in ReferIt Dataset
                            newData = fileName + "~" + segCodeText[0] + "~" + arg1Phrase + "~0.0~0.0";
                        }
                        //Save the new generated data to file
                        printToFile.println(newData);
                    }
                }
                // Remaining Phrases with no Image Relation
//                for(String s : pharseRemaining.keySet()) {
//                    // Writing Remaining Phrases
//                    // 0 index doesn't exists
//                    String referitKey = fileName + "_0.jpg";
//
//                    String newData = fileName + "~" + "0" + "~" + pharseRemaining.get(s) + "~0.0"  + "~0.0";
//                    //Save the new generated data to file
//                    printToFile.println(newData);
//                }
            }
        }
        printToFile.close();
        printToFileNames.close();
    }

    private void completeText(String directory, String filename) throws IOException {
        String filePath = directory + "/textData/" + filename + ".txt";
        File file = new File(filePath);
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        int index = 0;
        while ((line = reader.readLine()) != null) {
            if(line.endsWith("-Phrases:")) {
                // Go to Phrases line
                index += line.length()+1;
                line = reader.readLine();
                String[] phrases = line.trim().split(",");
                for(String p : phrases) {
                    int startSpan = index +1;
                    int endSpan = index +1 + p.trim().length();
                    String phraseWithSpan = p.trim() + " " + startSpan + " " + endSpan;
                    allPhrases.add(phraseWithSpan);
                    index +=p.trim().length() + 2;
                }
                index += 1;
            }
            else {
                if (line.length()!=0)
                    index += line.length();
                else
                    index += 2;
            }
        }

    }

    private void loadTestImage(String directory) throws IOException {
        String file = directory + "/Output/ClefImage.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String imageId = line.trim();
            Image i = new Image(imageId.trim() + ".jpg", imageId.trim());
            clefImages.add(i);
        }
    }

    private void loadTestSegment(String directory) throws IOException {
        String file = directory + "/Output/ClefSegment.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] segInfo = line.split("\\~");
            System.out.println(line);
            String exp = removeDuplicates(segInfo[2]);
            Segment s = new Segment(segInfo[0], Integer.parseInt(segInfo[1]),"", exp,false);
            clefSegments.add(s);
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

    public void generateNLPBaseClasses() {
        for (Segment s :  clefSegments) {
            String ID = s.getAssociatedImageID() + "_" + s.getSegmentId();
            Document d = new Document(ID);
            int len = 0;
            if(s.referItExpression!=null)
                len = s.referItExpression.length();
            Sentence sen = new Sentence(d, ID, 0, len, s.referItExpression);
            clefDocuments.add(d);
            clefSentences.add(sen);
        }
    }

    public String removeDuplicates(String s) {
        s = s.toLowerCase().replaceAll("[^a-z]", " ").replaceAll("( )+", " ").trim();
        return new LinkedHashSet<String>(Arrays.asList(s.split(" "))).toString().replaceAll("(^\\[|\\]$)", "").replace(", ", " ");
    }
}
