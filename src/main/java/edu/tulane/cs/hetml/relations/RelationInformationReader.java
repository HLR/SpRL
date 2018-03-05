package edu.tulane.cs.hetml.relations;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class RelationInformationReader {

    public List<VisualGenomeRelations> visualgenomeRelations;
    public List<VisualGenomeStats> visualGenomeStats;
    public List<ClefRelationsHeadwords> clefHeadwordRelations;

    public RelationInformationReader() {

    }

    public void loadClefRelationHeadwords(String directory, Boolean isTrain) throws IOException {
        System.out.println("Loading Clef Relation Headwords from File...");
        clefHeadwordRelations = new ArrayList<>();
        String file;
        if(isTrain)
            file = directory + "/tripletHeadwords/tripletsHeadWordsTrain.txt";
        else
            file = directory + "/tripletHeadwords/tripletsHeadWordsTest.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] relInfo = line.split("~");
            String id = relInfo[0];
            String tr = relInfo[1];
            String sp = relInfo[2];
            String lm = relInfo[3];

            ClefRelationsHeadwords crh = new ClefRelationsHeadwords(id, tr, lm, sp);
            clefHeadwordRelations.add(crh);
        }
    }

    public void loadRelations(String directory) throws IOException {
        System.out.println("Loading Visual Genome Relations from File...");
        visualgenomeRelations = new ArrayList<>();
        String file = directory + "/VGData/flat_relations.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        line = reader.readLine();
        while ((line = reader.readLine()) != null) {
            String[] relInfo = line.split(",");
            String predicate = relInfo[1];
            String object = relInfo[2];
            String subject = relInfo[3];

            String[] labels = getLabels(relInfo);

            VisualGenomeRelations vgr = new VisualGenomeRelations();
            vgr.setValues(labels[0],labels[1], predicate, subject, object);

            visualgenomeRelations.add(vgr);
        }
    }

    public void loadStats(String directory, Boolean isTrain) throws IOException {
        visualGenomeStats = new ArrayList<>();
        String file;
        if(isTrain)
            file = directory + "/extactMatchRelsTrain.txt";
        else
            file = directory + "/extactMatchRelsTest.txt";
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        line = reader.readLine();

        while ((line = reader.readLine()) != null) {
            String[] relInfo = line.split(",");
            String predicate = relInfo[0];
            String subject = relInfo[1];
            String object = relInfo[2];
            int totalInstances = Integer.parseInt(relInfo[3]);
            Double ecScore = Double.parseDouble(relInfo[4]);
            Double dcScore = Double.parseDouble(relInfo[5]);
            Double tppScore = Double.parseDouble(relInfo[6]);
            Double tppiScore = Double.parseDouble(relInfo[7]);
            Double nttpScore = Double.parseDouble(relInfo[8]);
            Double nttpiScore = Double.parseDouble(relInfo[9]);
            Double eqScore = Double.parseDouble(relInfo[10]);
            Double poScore = Double.parseDouble(relInfo[11]);
            Double aboveScore = Double.parseDouble(relInfo[12]);
            Double belowScore = Double.parseDouble(relInfo[13]);
            Double leftScore = Double.parseDouble(relInfo[14]);
            Double rightScore = Double.parseDouble(relInfo[15]);

            VisualGenomeStats vgs = new VisualGenomeStats();
            vgs.setValues(predicate, subject, object, ecScore,dcScore,tppScore,tppiScore,nttpScore,nttpiScore,eqScore,poScore,aboveScore,belowScore,leftScore,rightScore, totalInstances);

            visualGenomeStats.add(vgs);
        }

    }

    private String[] getLabels(String[] rcc8) {
        String[] labels = new String[2];
        if(Integer.parseInt(rcc8[12])==1)
            labels[0]="DC";
        else if(Integer.parseInt(rcc8[13])==1)
            labels[0]="EC";
        else if(Integer.parseInt(rcc8[14])==1)
            labels[0]="TPP";
        else if(Integer.parseInt(rcc8[15])==1)
            labels[0]="TPPi";
        else if(Integer.parseInt(rcc8[16])==1)
            labels[0]="NTPP";
        else if(Integer.parseInt(rcc8[17])==1)
            labels[0]="NTPPi";
        else if(Integer.parseInt(rcc8[18])==1)
            labels[0]="EQ";
        else if(Integer.parseInt(rcc8[19])==1)
            labels[0]="PO";

        if(Integer.parseInt(rcc8[20])==1)
            labels[1]="ABOVE";
        else if(Integer.parseInt(rcc8[21])==1)
            labels[1]="BELOW";
        else if(Integer.parseInt(rcc8[22])==1)
            labels[1]="LEFT";
        else if(Integer.parseInt(rcc8[23])==1)
            labels[1]="RIGHT";

        return labels;
    }
}
