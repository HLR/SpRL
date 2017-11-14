package edu.tulane.cs.hetml.vision;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class RefExpTrainedWordReader {

    public List<String> filteredWords;
    public HashMap<String, String> expSegScores;
    public HashMap<String, String> wordSegScores;

    public RefExpTrainedWordReader() {

    }
    public RefExpTrainedWordReader(String directory) throws IOException {
        String file = directory +  "/TrainedWords.txt";
        File d = new File(file);

        if (d.exists()) {
            filteredWords = new ArrayList<>();
            String line;
            BufferedReader reader = new BufferedReader(new FileReader(file));
            while ((line = reader.readLine()) != null) {
                filteredWords.add(line.trim());
            }
        }
    }

    public void ExpClsOutput(String directory) throws IOException {
        String file = directory +  "/EC-InstanceResults.txt";
        File d = new File(file);

        if (d.exists()) {
            expSegScores = new HashMap<>();
            String line;
            BufferedReader reader = new BufferedReader(new FileReader(file));
            while ((line = reader.readLine()) != null) {
                String scoreList = reader.readLine();
                expSegScores.put(line.trim(), scoreList);
                String Id = reader.readLine();
            }
        }
    }

    public void WordClsOutput(String directory) throws IOException {
        String file = directory +  "/WC-InstanceResults.txt";
        File d = new File(file);

        if (d.exists()) {
            wordSegScores = new HashMap<>();
            String line;
            BufferedReader reader = new BufferedReader(new FileReader(file));
            while ((line = reader.readLine()) != null) {
                String scoreList = reader.readLine();
                wordSegScores.put(line.trim(), scoreList);
                String Id = reader.readLine();
            }
        }
    }

    public void findDiff() {
        int count = 0, wrong = 0;
        for (String key : expSegScores.keySet()) {
            String expScore = expSegScores.get(key);
            String wordScore = wordSegScores.get(key);
            if (expScore.equalsIgnoreCase(wordScore)) {
                count++;
            }
            else
                wrong++;
        }
        System.out.println("Same: " + count + "Diff" + wrong);
    }

}
