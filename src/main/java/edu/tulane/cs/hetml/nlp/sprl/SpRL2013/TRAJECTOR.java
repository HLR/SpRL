package edu.tulane.cs.hetml.nlp.sprl.SpRL2013;


import edu.tulane.cs.hetml.nlp.sprl.SpRLAnnotation;

import javax.xml.bind.annotation.*;


@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "")
@XmlRootElement(name = "TRAJECTOR")
public class TRAJECTOR implements SpRLAnnotation {

    @XmlAttribute(name = "id", required = true)
    protected String id;
    @XmlAttribute(name = "start", required = true)
    protected Integer start;
    @XmlAttribute(name = "end", required = true)
    protected Integer end;
    @XmlAttribute(name = "text", required = true)
    protected String text;

    /**
     * Gets the value of the id property.
     *
     * @return possible object is
     * {@link String }
     */
    public String getId() {
        return id;
    }

    /**
     * Sets the value of the id property.
     *
     * @param value allowed object is
     *              {@link String }
     */
    public void setId(String value) {
        this.id = value;
    }

    /**
     * Gets the value of the start property.
     *
     * @return possible object is
     * {@link Integer }
     */
    public Integer getStart() {
        return start;
    }

    /**
     * Sets the value of the start property.
     *
     * @param value allowed object is
     *              {@link Integer }
     */
    public void setStart(Integer value) {
        this.start = value;
    }

    /**
     * Gets the value of the end property.
     *
     * @return possible object is
     * {@link Integer }
     */
    public Integer getEnd() {
        return end;
    }

    /**
     * Sets the value of the end property.
     *
     * @param value allowed object is
     *              {@link Integer }
     */
    public void setEnd(Integer value) {
        this.end = value;
    }

    /**
     * Gets the value of the text property.
     *
     * @return possible object is
     * {@link String }
     */
    public String getText() {
        return text;
    }

    /**
     * Sets the value of the text property.
     *
     * @param value allowed object is
     *              {@link String }
     */
    public void setText(String value) {
        this.text = value;
    }
}
