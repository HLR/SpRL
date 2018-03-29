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
    public List<String> missingWords;

    public WordasClassifierTrainedWordsReader() {

    }

    public void loadTrainedWords(String directory) throws IOException {
        String frequencyWords = directory + "TrainedWords.txt";
        String clefWords = directory + "missedWords.txt";
        filteredWords = new ArrayList<>();
        missingWords = new ArrayList<>();
        readWordsFromFile(frequencyWords, true);
        readWordsFromFile(clefWords, false);
    }

    private void readWordsFromFile(String filepath, boolean isFrequencyWords) throws IOException {
        File d = new File(filepath);
        if (d.exists()) {
            String line;
            BufferedReader reader = new BufferedReader(new FileReader(d));
            while ((line = reader.readLine()) != null) {
                if(isFrequencyWords)
                    filteredWords.add(line.trim());
                else {
                    String[] word = line.split("_");
                    missingWords.add(word[0]);
                }
            }
        }
    }


}
