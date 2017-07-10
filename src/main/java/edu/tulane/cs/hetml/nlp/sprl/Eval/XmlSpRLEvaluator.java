package edu.tulane.cs.hetml.nlp.sprl.Eval;

import edu.illinois.cs.cogcomp.saulexamples.nlp.SpatialRoleLabeling.SpRL2013.*;
import edu.illinois.cs.cogcomp.saulexamples.nlp.SpatialRoleLabeling.SpRLAnnotation;
import edu.tulane.cs.hetml.nlp.sprl.SpRL2017.*;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by taher on 2017-04-17.
 */
public class XmlSpRLEvaluator {
    private File actualFile;
    private File predictedFile;
    private SpRL2017Document actualDoc;
    private SpRL2017Document predictedDoc;
    private EvalComparer comparer;

    public XmlSpRLEvaluator(String actualPath, String predictedPath, EvalComparer comparer) {
        actualFile = new File(actualPath);
        if (!actualFile.exists())
            error("actual file doesn't exist.");

        predictedFile = new File(predictedPath);
        if (!predictedFile.exists())
            error("predictions file doesn't exist.");

        this.comparer = comparer;

        readDocs();
    }

    public List<SpRLEvaluation> evaluateRoles() {
        SpRLEvaluator evaluator = new SpRLEvaluator();

        RolesEvalDocument actualRoles = getRoleEvals(getActualDoc());
        RolesEvalDocument predictedRoles = getRoleEvals(getPredictedDoc());
        List<SpRLEvaluation> roleResults = evaluator.evaluateRoles(actualRoles, predictedRoles, comparer);

        return roleResults;
    }

    public List<SpRLEvaluation> evaluateRelations() {
        SpRLEvaluator evaluator = new SpRLEvaluator();

        RelationsEvalDocument actualRelations = getRelationEvals(getActualDoc());
        RelationsEvalDocument predictedRelations = getRelationEvals(getPredictedDoc());
        List<SpRLEvaluation> relationResults =
                evaluator.evaluateRelations(actualRelations, predictedRelations, comparer);

        return relationResults;
    }

    private RolesEvalDocument getRoleEvals(SpRL2017Document doc) {
        return new RolesEvalDocument(
                getRoles(doc, doc.getAllTrajectors()),
                getRoles(doc, doc.getAllIndicators()),
                getRoles(doc, doc.getAllLandmarks()));
    }

    private RelationsEvalDocument getRelationEvals(SpRL2017Document doc) {
        List<RelationEval> relations = new ArrayList<>();
        for (Scene scene : doc.getScenes())
            for (Sentence sentence : scene.getSentences())
                for (RELATION r : sentence.getRelations()) {
                    TRAJECTOR tr = doc.getTrajector(r.getTrajectorId());
                    LANDMARK lm = doc.getLandmark(r.getLandmarkId());
                    SPATIALINDICATOR sp = doc.getIndicator(r.getSpatialIndicatorId());

                    int offset = sentence.getStart();
                    int trStart = tr == null ? -1 : offset + tr.getStart().intValue();
                    int trEnd = tr == null ? -1 : offset + tr.getEnd().intValue();
                    int lmStart = lm == null ? -1 : offset + lm.getStart().intValue();
                    int lmEnd = lm == null ? -1 : offset + lm.getEnd().intValue();
                    int spStart = sp == null ? -1 : offset + sp.getStart().intValue();
                    int spEnd = sp == null ? -1 : offset + sp.getEnd().intValue();

                    RelationEval eval = new RelationEval(trStart, trEnd, spStart, spEnd, lmStart, lmEnd);
                    eval.setFoR(r.getFoR());
                    eval.setGeneralType(r.getGeneralType());
                    eval.setRCC8(r.getRCC8Value());
                    eval.setSpecificType(r.getSpecificType());
                    relations.add(eval);
                }
        return new RelationsEvalDocument(relations);
    }

    private <T extends SpRLAnnotation> List<RoleEval> getRoles(SpRL2017Document doc, List<T> annotations) {
        List<RoleEval> roles = new ArrayList<>();
        for (SpRLAnnotation a : annotations) {
            int offset = doc.getRoleSentence(a.getId()).getStart();
            roles.add(new RoleEval(offset + a.getStart().intValue(), offset + a.getEnd().intValue()));
        }
        return roles;
    }

    private void readDocs() {
        try {
            JAXBContext jaxbContext = JAXBContext.newInstance(SpRL2017Document.class);
            Unmarshaller jaxbUnmarshaller = jaxbContext.createUnmarshaller();

            actualDoc = (SpRL2017Document) jaxbUnmarshaller.unmarshal(actualFile);
            predictedDoc = (SpRL2017Document) jaxbUnmarshaller.unmarshal(predictedFile);

        } catch (JAXBException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private void error(String message) {
        System.out.println(message);
        System.exit(1);
    }

    public SpRL2017Document getActualDoc() {
        return actualDoc;
    }

    public SpRL2017Document getPredictedDoc() {
        return predictedDoc;
    }
}
