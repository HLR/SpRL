package edu.tulane.cs.hetml.vision;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Umar Manzoor on 26/12/2016.
 */
public class Image {
    private final String label;
    private String id;
    private double width;
    private double height;

    public Image(String label, String id) {
        this.label = label;
        this.id = id;
    }

    public Image(String label, String id, double width, double height) {
        this.label = label;
        this.id = id;
        this.width = width;
        this.height = height;
    }

    public String getId()
    {
        return id;
    }

    public String getLabel()
    {
        return label;
    }

    public void setID(String ID)
    {
        id = ID;
    }

    public double getHeight() {
        return height;
    }

    public void setHeight(double height) {
        this.height = height;
    }

    public double getWidth() {
        return width;
    }

    public void setWidth(double width) {
        this.width = width;
    }

}
