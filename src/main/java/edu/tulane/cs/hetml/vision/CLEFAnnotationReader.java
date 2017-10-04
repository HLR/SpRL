package edu.tulane.cs.hetml.vision;

import java.io.*;
import java.util.ArrayList;
import java.util.Hashtable;

public class CLEFAnnotationReader {

    private Hashtable<String, String> tokens = new Hashtable<String, String>();
    private Hashtable<String, String> pharseRemaining = new Hashtable<String, String>();
    private Hashtable<String, String> referit = new Hashtable<String, String>();

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

        printToFile = new PrintWriter( directory + "/OutputDataSet/AnnotatedClef.txt");
        printToFileNames = new PrintWriter( directory + "/OutputDataSet/Clef_test.txt");
        //Load Referit Text
        loadReferit(directory);

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
                        String newData = referitKey + "~" + arg1Phrase + "~" + oldData[1] + "~" + oldData[2];
                        //Save the new generated data to file
                        printToFile.println(newData);
                    }
                }
                for(String s : pharseRemaining.keySet()) {
                    // Negative Examples
                    // 0 index doesn't exists
                    String referitKey = fileName + "_0.jpg";

                    String newData = referitKey + "~" + pharseRemaining.get(s) + "~0.0"  + "~0.0";
                    //Save the new generated data to file
                    printToFile.println(newData);
                }
            }
            break;
        }
        printToFile.close();
        printToFileNames.close();
    }

    private void loadReferit(String directory) throws IOException {
        String file = directory + "/Referit/ReferGames.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] segInfo = line.split("\\~");
            referit.put(segInfo[0], segInfo[1] + "~" + segInfo[2] + "~" + segInfo[3]);
        }
     }
}
