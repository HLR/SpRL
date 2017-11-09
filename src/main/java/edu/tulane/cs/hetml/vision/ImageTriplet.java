package edu.tulane.cs.hetml.vision;

import org.bytedeco.javacpp.presets.opencv_core;

public class ImageTriplet {
    private String trajector;
    private String landmark;
    private String sp;
    private String trBox;
    private String lmBox;
    private String imageBox;
    private String trVector; // F1
    private double trAreawrtLM; // F2
    private double trAspectRatio; // F3
    private double lmAspectRatio; // F3
    private double trAreaBbox; // F4
    private double lmAreaBbox; //F4
    private double iou; // F5
    private double euclideanDistance; // F6
    private double trAreaImage; // F7
    private double lmAreaImage; // F7

    public ImageTriplet() {

    }

    public ImageTriplet(String sp, String trajector, String landmark, String trBox, String lmBox, String imageBox,
                        String trVector, double trAreawrtLM, double trAspectRatio, double lmAspectRatio,
                        double trAreaBbox, double lmAreaBbox, double iou, double euclideanDistance, double trAreaImage,
                        double lmAreaImage) {
        this.sp = sp;
        this.trajector = trajector;
        this.landmark = landmark;
        this.trBox = trBox;
        this.lmBox = lmBox;
        this.imageBox = imageBox;
        this.trVector = trVector;
        this.trAreawrtLM = trAreawrtLM;
        this.trAspectRatio = trAspectRatio;
        this.lmAspectRatio = lmAspectRatio;
        this.trAreaBbox = trAreaBbox;
        this.lmAreaBbox = lmAreaBbox;
        this.iou = iou;
        this.euclideanDistance = euclideanDistance;
        this.trAreaImage = trAreaImage;
        this.lmAreaImage = lmAreaImage;
    }
}
