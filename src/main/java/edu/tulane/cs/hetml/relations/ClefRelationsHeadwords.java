package edu.tulane.cs.hetml.relations;

public class ClefRelationsHeadwords {
    private String id;
    private String tr;
    private String lm;
    private String sp;

    ClefRelationsHeadwords(String id, String tr, String lm, String sp) {
        this.setId(id);
        this.setTr(tr);
        this.setLm(lm);
        this.setSp(sp);
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getTr() {
        return tr;
    }

    public void setTr(String tr) {
        this.tr = tr;
    }

    public String getLm() {
        return lm;
    }

    public void setLm(String lm) {
        this.lm = lm;
    }

    public String getSp() {
        return sp;
    }

    public void setSp(String sp) {
        this.sp = sp;
    }
}
