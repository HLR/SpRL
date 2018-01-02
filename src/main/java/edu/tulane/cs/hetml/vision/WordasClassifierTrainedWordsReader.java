package edu.tulane.cs.hetml.vision;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class WordasClassifierTrainedWordsReader {

    public List<String> filteredWords;
    public List<String> clefWords;
    public List<String> missingWordsClefExamples;

    public WordasClassifierTrainedWordsReader() {

    }

    public void loadTrainedWords(String directory) throws IOException {
        String frequencyWords = directory + "newFrequencyWords.txt"; //"TrainedWords.txt";
        String clefWordspath = directory + "clefWords.txt";
        String clefMissedWordExamples = directory + "missedWordsClef.txt";
        filteredWords = new ArrayList<>();
        clefWords = new ArrayList<>();
        missingWordsClefExamples = new ArrayList<>();
        //readWordsFromFile(frequencyWords, true, false);
        readWordsFromFile(clefWordspath, false, false);
        //readWordsFromFile(clefMissedWordExamples, false, true);
    }

    private void readWordsFromFile(String filepath, boolean isFrequencyWords, boolean useClef) throws IOException {
        File d = new File(filepath);
        if (d.exists()) {
            String line;
            BufferedReader reader = new BufferedReader(new FileReader(d));
            while ((line = reader.readLine()) != null) {
                if(isFrequencyWords) {
                    String[] words = line.split("-");
                    //if(Integer.parseInt(words[1])>=1)
                        filteredWords.add(words[0]);
                }
                else {
//                    String[] word = line.split("_");
//                    if(!useClef)
                        clefWords.add(line);
//                    else
//                        missingWordsClefExamples.add(word[0]);
                }
            }
        }
    }


}
