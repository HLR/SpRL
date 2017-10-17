package edu.tulane.cs.hetml.vision;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class RefExpFilteredWordReader {

    public List<String> filteredWords;

    public RefExpFilteredWordReader(String directory) throws IOException {
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
}
