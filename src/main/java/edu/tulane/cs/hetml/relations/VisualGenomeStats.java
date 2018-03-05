package edu.tulane.cs.hetml.relations;

public class VisualGenomeStats {
    private String predicate;
    private String subject;
    private String object;
    private int totalInstances;
    private Double ecScore;
    private Double dcScore;
    private Double eqScore;
    private Double poScore;
    private Double tppScore;
    private Double tppiScore;
    private Double ntppScore;
    private Double ntppiScore;
    private Double aboveScore;
    private Double belowScore;
    private Double leftScore;
    private Double rightScore;

    public void setValues(String predicate, String subject, String object, Double ecScore, Double dcScore, Double tppScore, Double tppiScore, Double ntppScore, Double ntppiScore, Double eqScore, Double poScore, Double aboveScore, Double belowScore, Double leftScore, Double rightScore, int totalInstances) {
        this.predicate=predicate;
        this.subject=subject;
        this.object=object;
        this.ecScore=ecScore;
        this.dcScore=dcScore;
        this.ntppScore=ntppScore;
        this.ntppiScore=ntppiScore;
        this.tppScore=tppScore;
        this.tppiScore=tppiScore;
        this.poScore=poScore;
        this.eqScore=eqScore;
        this.aboveScore=aboveScore;
        this.belowScore=belowScore;
        this.leftScore=leftScore;
        this.rightScore=rightScore;
        this.setTotalInstances(totalInstances);
    }

    public Double[] getScoreArray() {
        Double[] scores = new Double[12];
        scores[0] = ecScore;
        scores[1] = dcScore;
        scores[2] = tppScore;
        scores[3] = tppiScore;
        scores[4] = ntppScore;
        scores[5] = ntppiScore;
        scores[6] = eqScore;
        scores[7] = poScore;
        scores[8] = aboveScore;
        scores[9] = belowScore;
        scores[10] = leftScore;
        scores[11] = rightScore;

        return scores;
    }

    public Double[] getScoreRCC8() {
        Double[] scores = new Double[8];
        scores[0] = ecScore;
        scores[1] = dcScore;
        scores[2] = tppScore;
        scores[3] = tppiScore;
        scores[4] = ntppScore;
        scores[5] = ntppiScore;
        scores[6] = eqScore;
        scores[7] = poScore;

        return scores;
    }

    public Double[] getScoreDirection() {
        Double[] scores = new Double[4];
        scores[0] = aboveScore;
        scores[1] = belowScore;
        scores[2] = leftScore;
        scores[3] = rightScore;

        return scores;
    }

    public String getPredicate() {
        return predicate;
    }

    public void setPredicate(String predicate) {
        this.predicate = predicate;
    }

    public String getSubject() {
        return subject;
    }

    public void setSubject(String subject) {
        this.subject = subject;
    }

    public String getObject() {
        return object;
    }

    public void setObject(String object) {
        this.object = object;
    }

    public Double getEcScore() {
        return ecScore;
    }

    public void setEcScore(Double ecScore) {
        this.ecScore = ecScore;
    }

    public Double getDcScore() {
        return dcScore;
    }

    public void setDcScore(Double dcScore) {
        this.dcScore = dcScore;
    }

    public Double getEqScore() {
        return eqScore;
    }

    public void setEqScore(Double eqScore) {
        this.eqScore = eqScore;
    }

    public Double getPoScore() {
        return poScore;
    }

    public void setPoScore(Double poScore) {
        this.poScore = poScore;
    }

    public Double getTppScore() {
        return tppScore;
    }

    public void setTppScore(Double tppScore) {
        this.tppScore = tppScore;
    }

    public Double getTppiScore() {
        return tppiScore;
    }

    public void setTppiScore(Double tppiScore) {
        this.tppiScore = tppiScore;
    }

    public Double getNtppScore() {
        return ntppScore;
    }

    public void setNtppScore(Double ntppScore) {
        this.ntppScore = ntppScore;
    }

    public Double getNtppiScore() {
        return ntppiScore;
    }

    public void setNtppiScore(Double ntppiScore) {
        this.ntppiScore = ntppiScore;
    }

    public Double getAboveScore() {
        return aboveScore;
    }

    public void setAboveScore(Double aboveScore) {
        this.aboveScore = aboveScore;
    }

    public Double getBelowScore() {
        return belowScore;
    }

    public void setBelowScore(Double belowScore) {
        this.belowScore = belowScore;
    }

    public Double getLeftScore() {
        return leftScore;
    }

    public void setLeftScore(Double leftScore) {
        this.leftScore = leftScore;
    }

    public Double getRightScore() {
        return rightScore;
    }

    public void setRightScore(Double rightScore) {
        this.rightScore = rightScore;
    }

    public int getTotalInstances() {
        return totalInstances;
    }

    public void setTotalInstances(int totalInstances) {
        this.totalInstances = totalInstances;
    }
}
