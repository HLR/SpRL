package edu.tulane.cs.hetml.vision;

public class ExpressionSegment {
    private String expression;
    private Segment segment;
    private boolean isMatching;

    public ExpressionSegment(String expression, Segment segment, Boolean isMatching) {
        this.expression = expression;
        this.segment = segment;
        this.isMatching = isMatching;
    }

    public String getExpression() {
        return expression;
    }

    public Segment getSegment() {
        return segment;
    }

    public boolean isExpressionAndSegmentMatching() {
        return isMatching;
    }

}
