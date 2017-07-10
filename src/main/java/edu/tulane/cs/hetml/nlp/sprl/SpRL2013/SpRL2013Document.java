package edu.tulane.cs.hetml.nlp.sprl.SpRL2013;

import edu.illinois.cs.cogcomp.core.datastructures.IntPair;
import edu.tulane.cs.hetml.nlp.sprl.*;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlTransient;
import javax.xml.bind.annotation.XmlType;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
    "url",
    "cp",
    "text",
    "tags"
})
@XmlRootElement(name = "SpRL")
public class SpRL2013Document implements SpRLXmlDocument {

    @XmlElement(name = "URL", required = false)
    protected URL url;
    @XmlElement(name = "CP", required = false)
    protected CP cp;
    @XmlElement(name = "TEXT", required = true)
    protected TEXT text;
    @XmlElement(name = "TAGS", required = true)
    protected TAGS tags;
    
    @XmlTransient
    protected String filename;
    @XmlTransient
    private HashMap<String, SPATIALINDICATOR> spatialIndicatorMap;
    @XmlTransient
    private HashMap<String, TRAJECTOR> trajectorHashMap;
    @XmlTransient
    private HashMap<String, LANDMARK> landmarkHashMap;
    @XmlTransient
    private HashMap<IntPair, List<RELATION>> tagRelationMap;

    /**
     * Gets the value of the url property.
     * 
     * @return
     *     possible object is
     *     {@link URL }
     *     
     */
    public URL getURL() {
        return url;
    }

    /**
     * Sets the value of the url property.
     * 
     * @param value
     *     allowed object is
     *     {@link URL }
     *     
     */
    public void setURL(URL value) {
        this.url = value;
    }

    /**
     * Gets the value of the cp property.
     * 
     * @return
     *     possible object is
     *     {@link CP }
     *     
     */
    public CP getCP() {
        return cp;
    }

    /**
     * Sets the value of the cp property.
     * 
     * @param value
     *     allowed object is
     *     {@link CP }
     *     
     */
    public void setCP(CP value) {
        this.cp = value;
    }

    /**
     * Gets the value of the text property.
     * 
     * @return
     *     possible object is
     *     {@link TEXT }
     *     
     */
    public TEXT getTEXT() {
        return text;
    }

    /**
     * Sets the value of the text property.
     * 
     * @param value
     *     allowed object is
     *     {@link TEXT }
     *     
     */
    public void setTEXT(TEXT value) {
        this.text = value;
    }

    /**
     * Gets the value of the tags property.
     * 
     * @return
     *     possible object is
     *     {@link TAGS }
     *     
     */
    public TAGS getTAGS() {
        return tags;
    }

    /**
     * Sets the value of the tags property.
     * 
     * @param value
     *     allowed object is
     *     {@link TAGS }
     *     
     */
    public void setTAGS(TAGS value) {
        this.tags = value;
    }

	/**
	 * @return the filename
	 */
	public String getFilename() {
		return filename;
	}

	/**
	 * @param filename the filename to set
	 */
	public void setFilename(String filename) {
		this.filename = filename;
	}

    public HashMap<String, SPATIALINDICATOR> getSpatialIndicatorMap() {
        if(spatialIndicatorMap == null) {
            spatialIndicatorMap = new HashMap<>();
            for (SPATIALINDICATOR sp : getTAGS().getSPATIALINDICATOR()) {
                spatialIndicatorMap.put(sp.getId(), sp);
            }
        }
        return spatialIndicatorMap;
    }

    public HashMap<String, TRAJECTOR> getTrajectorHashMap() {
        if(trajectorHashMap == null){
            trajectorHashMap = new HashMap<>();
            for(TRAJECTOR t : getTAGS().getTRAJECTOR()){
                trajectorHashMap.put(t.getId(), t);
            }
        }
        return trajectorHashMap;
    }

    public HashMap<String , LANDMARK> getLandmarkHashMap() {
        if(landmarkHashMap == null){
            landmarkHashMap = new HashMap<>();
            for(LANDMARK l : getTAGS().getLANDMARK()){
                landmarkHashMap.put(l.getId(), l);
            }
        }
        return landmarkHashMap;
    }

    public HashMap<IntPair, List<RELATION>> getTagRelationMap() {
        if(tagRelationMap == null){
            tagRelationMap = new HashMap<>();
            for(RELATION r : getTAGS().getRELATION()){
                addTagRelation(r, getSpatialIndicatorMap().get(r.spatialIndicatorId));
                addTagRelation(r, getTrajectorHashMap().get(r.trajectorId));
                addTagRelation(r, getLandmarkHashMap().get(r.landmarkId));
            }
        }
        return tagRelationMap;
    }

    private void addTagRelation(RELATION r, SpRLAnnotation tag) {
        if(tag == null || tag.getStart().intValue() == -1)
            return;

        IntPair p = new IntPair(tag.getStart().intValue(), tag.getEnd().intValue());
        if(!tagRelationMap.containsKey(p)){
            tagRelationMap.put(p, new ArrayList<>());
        }
        tagRelationMap.get(p).add(r);
    }
}
