package edu.tulane.cs.hetml.nlp.BaseTypes;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Taher on 2016-12-18.
 */
public abstract class NlpBaseElement extends SpanBasedElement {
    private String id;
    private String text;
    private Map<String, List<String>> properties = new HashMap<>();
    private boolean globalSpan;

    public NlpBaseElement() {
        setStart(-1);
        setEnd(-1);
    }

    public NlpBaseElement(String id, Integer start, Integer end, String text) {
        this.setId(id);
        this.setStart(start);
        this.setEnd(end);
        this.setText(text);
    }

    public abstract NlpBaseElementTypes getType();

    public boolean containsProperty(String name) {
        return properties.containsKey(name) && !properties.get(name).isEmpty();
    }

    public String getPropertyFirstValue(String name) {
        if (containsProperty(name))
            return properties.get(name).get(0);
        return null;
    }

    public List<String> getPropertyValues(String name) {
        if (containsProperty(name))
            return properties.get(name);
        return new ArrayList<>();
    }

    public void addPropertyValue(String name, String value) {
        if (!containsProperty(name))
            properties.put(name, new ArrayList<>());
        properties.get(name).add(value);
    }

    public void removeProperty(String name) {
        if (containsProperty(name))
            properties.remove(name);
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public static NlpBaseElement create(NlpBaseElementTypes type) {

        switch (type) {
            case Document:
                return new Document();
            case Sentence:
                return new Sentence();
            case Phrase:
                return new Phrase();
            case Token:
                return new Token();
        }
        return null;
    }

    public int getRelativeStart() {
        return getStart();
    }

    public int getRelativeEnd() {
        return getEnd();
    }

    public int getGlobalStart() {
        return getStart();
    }

    public int getGlobalEnd() {
        return getEnd();
    }

    @Override
    public String toString() {
        return getText();
    }

    public boolean isGlobalSpan() {
        return globalSpan;
    }

    public void setGlobalSpan(boolean globalSpan) {
        this.globalSpan = globalSpan;
    }

}
