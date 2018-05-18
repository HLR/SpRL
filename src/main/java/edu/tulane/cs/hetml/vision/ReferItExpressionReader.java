package edu.tulane.cs.hetml.vision;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

public class ReferItExpressionReader {

    public List<String> referitExpressions;
    public Hashtable<String, String> referit = new Hashtable<String, String>();

    private String filename = "/ReferGames.txt";

    public void loadReferitExpressions(String directory) throws IOException {
        String file = directory + filename;
        referitExpressions = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] segInfo = line.split("\\~");
            referit.put(segInfo[0], segInfo[1] + "~" + segInfo[2] + "~" + segInfo[3]);
            referitExpressions.add(line);
        }
    }

}
