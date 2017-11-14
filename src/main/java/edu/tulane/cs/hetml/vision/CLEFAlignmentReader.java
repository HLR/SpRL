package edu.tulane.cs.hetml.vision;

import org.apache.commons.io.FileUtils;

import java.io.*;
import java.util.*;

public class CLEFAlignmentReader {
    final String _annotationDir;
    final String _textDir;

    public CLEFAlignmentReader(String annotationDir, String textDir) {
        _annotationDir = annotationDir;
        _textDir = textDir;
    }

    public List<String[]> getAlignments() throws Exception {
        Map<String, File> listOfTextFiles = new HashMap<>();
        Map<String, File> listOfAnnFiles = new HashMap<>();
        loadFiles(listOfAnnFiles, listOfTextFiles);

        List<String[]> alignments = new ArrayList<>();

        for (String fileName : listOfAnnFiles.keySet()) {
            if (!listOfTextFiles.containsKey(fileName))
                throw new Exception("Cannot find the text file for " + fileName + ".ann");

            Map<Integer, String> sentenceSpans = getSentences(listOfTextFiles.get(fileName));
            String line;
            BufferedReader annReader = new BufferedReader(new FileReader(listOfAnnFiles.get(fileName)));
            Map<String, String> phraseMap = new HashMap<String, String>();
            Map<String, String> phraseSentenceMap = new HashMap<String, String>();
            Map<String, String> segmentMap = new HashMap<String, String>();

            while ((line = annReader.readLine()) != null) {

                String[] parts = line.split("\\t");
                String[] labelParts = parts[1].split(" ");
                String label = labelParts[0].toLowerCase();

                switch (label) {
                    case "phrase":
                        int start = Integer.parseInt(labelParts[1]);
                        String sentence = getSentence(sentenceSpans, start);
                        parts[2] = parts[2].trim();
                        if(parts[2].endsWith(","))
                            parts[2] = parts[2].substring(0, parts[2].length() - 1);
                        phraseMap.put(parts[0], parts[2]);
                        phraseSentenceMap.put(parts[0], sentence);
                        break;

                    case "segment":
                        segmentMap.put(parts[0], parts[2].split(" ")[0]);
                        break;

                    case "related":
                        String arg1 = labelParts[1].split(":")[1];
                        String arg2 = labelParts[2].split(":")[1];
                        alignments.add(new String[]{
                                fileName,
                                phraseSentenceMap.get(arg1),
                                phraseMap.get(arg1),
                                segmentMap.get(arg2)
                        });
                        break;

                    default:
                        throw new Exception("malformed format");
                }
            }
        }
        return alignments;
    }

    private String getSentence(Map<Integer, String> sentenceSpans, int start) {
        List keys = Arrays.asList(sentenceSpans.keySet().toArray());
        keys.sort(Comparator.comparingInt(o -> (int) o));
        for (int i = 0; i < keys.size(); i++) {
            int sStart = (int) keys.get(i);
            int sEnd = i < keys.size() - 1 ? (int) keys.get(i + 1): 1000000;
            if (sStart <= start && start < sEnd)
                return sentenceSpans.get(sStart);

        }
        return null;
    }

    private Map<Integer, String> getSentences(File f) throws IOException {
        String text = FileUtils.readFileToString(f, "utf-8");
        int index = 0;
        int num = 1;
        Map<Integer, String> sentenceMap = new HashMap<>();
        while (true) {
            String label = "S" + num + "-Phrases:\n";
            index = text.indexOf(label, index);
            if (index < 0)
                break;
            index += label.length();
            String sLabel = "S" + num + ": ";
            int sStart = text.indexOf(sLabel) + sLabel.length();
            int sEnd = text.indexOf("\n", sStart);
            String sent = text.substring(sStart, sEnd);
            sentenceMap.put(index, sent);
            num++;
        }
        return sentenceMap;
    }

    private void loadFiles(Map<String, File> annFiles, Map<String, File> textFiles) throws IOException {
        File annDir = new File(_annotationDir);
        File textDir = new File(_textDir);

        if (!annDir.exists()) {
            throw new IOException(_annotationDir + " does not exist!");
        }

        if (!annDir.isDirectory()) {
            throw new IOException(_annotationDir + " is not a directory!");
        }

        if (!textDir.exists()) {
            throw new IOException(_textDir + " does not exist!");
        }

        if (!textDir.isDirectory()) {
            throw new IOException(_textDir + " is not a directory!");
        }

        for (File f : annDir.listFiles())
            if (f.isFile()) {
                annFiles.put(f.getName().split("\\.")[0], f);
            }

        for (File f : textDir.listFiles())
            if (f.isFile())
                textFiles.put(f.getName().split("\\.")[0], f);

    }

}
