package edu.tulane.cs.hetml.nlp.sprl.SpRL2013;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.*;

@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "")
@XmlRootElement(name = "RELATION")
public class RELATION {

    @XmlAttribute(name = "id", required = true)
    protected String id;
    @XmlAttribute(name = "trajector_id", required = true)
    protected String trajectorId;
    @XmlAttribute(name = "landmark_id")
    protected String landmarkId;
    @XmlAttribute(name = "spatial_indicator_id")
    protected String spatialIndicatorId;
    @XmlAttribute(name = "motion_id")
    protected String motionId;
    @XmlAttribute(name = "direction_id")
    protected String directionId;
    @XmlAttribute(name = "distance_id")
    protected String distanceId;
    @XmlAttribute(name = "path_id")
    protected String pathId;
    @XmlAttribute(name = "quantitative_value")
    protected String quantitativeValue;
    @XmlAttribute(name = "qualitative_value")
    protected String qualitativeValue;
    @XmlAttribute(name = "general_type")
    protected String generalType;
    @XmlAttribute(name = "specific_type", required = true)
    protected String specificType;
    @XmlAttribute(name = "RCC8_value")
    protected String rcc8Value;
    @XmlAttribute(name = "absolute_value")
    protected String absoluteValue;
    @XmlAttribute(name = "relative_value")
    protected String relativeValue;
    @XmlAttribute(name = "FoR")
    protected String foR;

    /**
     * Gets the value of the id property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getId() {
        return id;
    }

    /**
     * Sets the value of the id property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setId(String value) {
        this.id = value;
    }

    /**
     * Gets the value of the trajectorId property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getTrajectorId() {
        return trajectorId;
    }

    /**
     * Sets the value of the trajectorId property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setTrajectorId(String value) {
        this.trajectorId = value;
    }

    /**
     * Gets the value of the landmarkId property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getLandmarkId() {
        return landmarkId;
    }

    /**
     * Sets the value of the landmarkId property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setLandmarkId(String value) {
        this.landmarkId = value;
    }

    /**
     * Gets the value of the spatialIndicatorId property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSpatialIndicatorId() {
        return spatialIndicatorId;
    }

    /**
     * Sets the value of the spatialIndicatorId property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSpatialIndicatorId(String value) {
        this.spatialIndicatorId = value;
    }

    /**
     * Gets the value of the motionId property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getMotionId() {
        return motionId;
    }

    /**
     * Sets the value of the motionId property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setMotionId(String value) {
        this.motionId = value;
    }

    /**
     * Gets the value of the directionId property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDirectionId() {
        return directionId;
    }

    /**
     * Sets the value of the directionId property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDirectionId(String value) {
        this.directionId = value;
    }

    /**
     * Gets the value of the distanceId property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDistanceId() {
        return distanceId;
    }

    /**
     * Sets the value of the distanceId property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDistanceId(String value) {
        this.distanceId = value;
    }

    /**
     * Gets the value of the pathId property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getPathId() {
        return pathId;
    }

    /**
     * Sets the value of the pathId property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setPathId(String value) {
        this.pathId = value;
    }

    /**
     * Gets the value of the quantitativeValue property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getQuantitativeValue() {
        return quantitativeValue;
    }

    /**
     * Sets the value of the quantitativeValue property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setQuantitativeValue(String value) {
        this.quantitativeValue = value;
    }

    /**
     * Gets the value of the qualitativeValue property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getQualitativeValue() {
        return qualitativeValue;
    }

    /**
     * Sets the value of the qualitativeValue property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setQualitativeValue(String value) {
        this.qualitativeValue = value;
    }

    /**
     * Gets the value of the generalType property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getGeneralType() {
        return generalType;
    }

    /**
     * Sets the value of the generalType property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setGeneralType(String value) {
        this.generalType = value;
    }

    /**
     * Gets the value of the specificType property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSpecificType() {
        return specificType;
    }

    /**
     * Sets the value of the specificType property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSpecificType(String value) {
        this.specificType = value;
    }

    /**
     * Gets the value of the rcc8Value property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getRCC8Value() {
        return rcc8Value;
    }

    /**
     * Sets the value of the rcc8Value property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setRCC8Value(String value) {
        this.rcc8Value = value;
    }

    /**
     * Gets the value of the absoluteValue property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getAbsoluteValue() {
        return absoluteValue;
    }

    /**
     * Sets the value of the absoluteValue property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setAbsoluteValue(String value) {
        this.absoluteValue = value;
    }

    /**
     * Gets the value of the relativeValue property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getRelativeValue() {
        return relativeValue;
    }

    /**
     * Sets the value of the relativeValue property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setRelativeValue(String value) {
        this.relativeValue = value;
    }

    /**
     * Gets the value of the foR property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getFoR() {
        return foR;
    }

    /**
     * Sets the value of the foR property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setFoR(String value) {
        this.foR = value;
    }

}
