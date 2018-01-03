package edu.tulane.cs.hetml.vision;

import java.awt.geom.Rectangle2D;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

public class JSONReader {

    public List<ImageTriplet> allImageTriplets;
    public Hashtable<Integer, String> imageInfo;

    public void readJsonFile(String directory) throws IOException, InterruptedException {
        allImageTriplets = new ArrayList<>();
        imageInfo = new Hashtable<Integer, String>();

        String imagefile = directory + "image_data.json";
        readImageInfo(imagefile);

        String file = directory + "relationships.json";
        JSONParser parser = new JSONParser();
        try {
            Object obj = parser.parse(new FileReader(file));
            JSONArray jsonObject = (JSONArray) obj;

            for(int i = 0 ; i< jsonObject.size() ;i++){
              JSONObject arr = (JSONObject) jsonObject.get(i);
              System.out.println(i);
              JSONArray arguments = (JSONArray) arr.get("relationships");

              int imageId = Integer.parseInt(arr.get("image_id").toString());

              String[] imageWidthHeight = imageInfo.get(imageId).split("-");
              double imageWidth = Double.parseDouble(imageWidthHeight[0]);
              double imageHeight = Double.parseDouble(imageWidthHeight[1]);

                for(int j = 0 ; j< arguments.size() ;j++) {
                    JSONObject instance = (JSONObject) arguments.get(j);

                    String sp = instance.get("predicate").toString().toLowerCase();

                    JSONObject subjectFeatures = (JSONObject)instance.get("subject");
                    JSONObject objectFeatures = (JSONObject)instance.get("object");

                    String tr;
                    if(subjectFeatures.size()==8)
                        tr = ((JSONArray)subjectFeatures.get("names")).get(0).toString();
                    else
                        tr = subjectFeatures.get("name").toString();

                    String trBoxString = subjectFeatures.get("x").toString() + "-" + subjectFeatures.get("y").toString() + "-" +
                            subjectFeatures.get("w").toString() + "-" + subjectFeatures.get("h").toString();

                    String lm;
                    if(objectFeatures.size()==8)
                        lm = ((JSONArray)objectFeatures.get("names")).get(0).toString();
                    else
                        lm = objectFeatures.get("name").toString();

                    String lmBoxString = objectFeatures.get("x").toString() + "-" + objectFeatures.get("y").toString() + "-" +
                            objectFeatures.get("w").toString() + "-" + objectFeatures.get("h").toString();

                    Rectangle2D trBox = RectangleHelper.parseRectangle(trBoxString, "-");
                    Rectangle2D lmBox = RectangleHelper.parseRectangle(lmBoxString, "-");

                    ImageTriplet it = new ImageTriplet(sp, tr, lm, trBox, lmBox, imageWidth, imageHeight);
                    it.setImageId(imageId  + "");
                    allImageTriplets.add(it);
                }
            }
        } catch (Exception e) {
           e.printStackTrace();
        }
    }

    private void readImageInfo(String file) throws IOException, InterruptedException {
        JSONParser parser = new JSONParser();
        try {
            Object obj = parser.parse(new FileReader(file));
            JSONArray jsonObject = (JSONArray) obj;

            for(int i = 0 ; i< jsonObject.size() ;i++){
                JSONObject instance = (JSONObject) jsonObject.get(i);

                int width = Integer.parseInt(instance.get("width").toString());
                int height = Integer.parseInt(instance.get("height").toString());
                int imageId = Integer.parseInt(instance.get("image_id").toString());

                imageInfo.put(imageId, width + "-" + height);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
