package org.example;

public class DocTuple {

    private String code;
    private String text;

    private double cosineSim;

    public DocTuple(String code, String text) {
        this.code = code;
        this.text = text;
    }

    public DocTuple(String code, double cosineSim) {
        this.code = code;
        this.cosineSim = cosineSim;
    }

    public double getCosineSim() {
        return cosineSim;
    }

    public String getCode() { return code; }

    public String getText() {
        return text;
    }

}
