package edu.tulane.cs.hetml.nlp.sprl.SpRL2013;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;

@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
    "trajector",
    "landmark",
    "spatialindicator",
    "motionindicator",
    "path",
    "direction",
    "distance",
    "relation"
})
@XmlRootElement(name = "TAGS")
public class TAGS {

    @XmlElement(name = "TRAJECTOR", required = true)
    protected List<TRAJECTOR> trajector;
    @XmlElement(name = "LANDMARK", required = true)
    protected List<LANDMARK> landmark;
    @XmlElement(name = "SPATIAL_INDICATOR", required = false)
    protected List<SPATIALINDICATOR> spatialindicator;
    @XmlElement(name = "MOTION_INDICATOR", required = false)
    protected List<MOTIONINDICATOR> motionindicator;
    @XmlElement(name = "PATH", required = false)
    protected List<PATH> path;
    @XmlElement(name = "DIRECTION", required = false)
    protected List<DIRECTION> direction;
    @XmlElement(name = "DISTANCE", required = false)
    protected List<DISTANCE> distance;
    @XmlElement(name = "RELATION", required = true)
    protected List<RELATION> relation;


    /**
     * Gets the value of the trajector property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the trajector property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getTRAJECTOR().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link TRAJECTOR }
     * 
     * 
     */
    public List<TRAJECTOR> getTRAJECTOR() {
        if (trajector == null) {
            trajector = new ArrayList<TRAJECTOR>();
        }
        return this.trajector;
    }

    /**
     * Gets the value of the landmark property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the landmark property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getLANDMARK().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link LANDMARK }
     * 
     * 
     */
    public List<LANDMARK> getLANDMARK() {
        if (landmark == null) {
            landmark = new ArrayList<LANDMARK>();
        }
        return this.landmark;
    }

    /**
     * Gets the value of the spatialindicator property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the spatialindicator property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getSPATIALINDICATOR().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link SPATIALINDICATOR }
     * 
     * 
     */
    public List<SPATIALINDICATOR> getSPATIALINDICATOR() {
        if (spatialindicator == null) {
            spatialindicator = new ArrayList<SPATIALINDICATOR>();
        }
        return this.spatialindicator;
    }

    /**
     * Gets the value of the motionindicator property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the motionindicator property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getMOTIONINDICATOR().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link MOTIONINDICATOR }
     * 
     * 
     */
    public List<MOTIONINDICATOR> getMOTIONINDICATOR() {
        if (motionindicator == null) {
            motionindicator = new ArrayList<MOTIONINDICATOR>();
        }
        return this.motionindicator;
    }

    /**
     * Gets the value of the path property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the path property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getPATH().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link PATH }
     * 
     * 
     */
    public List<PATH> getPATH() {
        if (path == null) {
            path = new ArrayList<PATH>();
        }
        return this.path;
    }

    /**
     * Gets the value of the direction property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the direction property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getDIRECTION().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link DIRECTION }
     * 
     * 
     */
    public List<DIRECTION> getDIRECTION() {
        if (direction == null) {
            direction = new ArrayList<DIRECTION>();
        }
        return this.direction;
    }

    /**
     * Gets the value of the distance property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the distance property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getDISTANCE().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link DISTANCE }
     * 
     * 
     */
    public List<DISTANCE> getDISTANCE() {
        if (distance == null) {
            distance = new ArrayList<DISTANCE>();
        }
        return this.distance;
    }

    /**
     * Gets the value of the relation property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the relation property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getRELATION().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link RELATION }
     * 
     * 
     */
    public List<RELATION> getRELATION() {
        if (relation == null) {
            relation = new ArrayList<RELATION>();
        }
        return this.relation;
    }


}
