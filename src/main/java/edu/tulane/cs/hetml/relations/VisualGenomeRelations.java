package edu.tulane.cs.hetml.relations;

public class VisualGenomeRelations {
    private String rcc8Label;
    private String directionLabel;
    private String predicate;
    private String subject;
    private String object;

    public void setValues(String rec8Label, String directionLabel, String predicate, String subject, String object) {
        this.setRcc8Label(rec8Label);
        this.setDirectionLabel(directionLabel);
        this.setPredicate(predicate);
        this.setObject(object);
        this.setSubject(subject);
    }

    public String getRcc8Label() {
        return rcc8Label;
    }

    public void setRcc8Label(String rcc8Label) {
        this.rcc8Label = rcc8Label;
    }

    public String getDirectionLabel() {
        return directionLabel;
    }

    public void setDirectionLabel(String directionLabel) {
        this.directionLabel = directionLabel;
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
}
