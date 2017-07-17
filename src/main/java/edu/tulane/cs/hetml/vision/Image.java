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

    public Image(String label, String id) {
        this.label = label;
        this.id = id;
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

}
