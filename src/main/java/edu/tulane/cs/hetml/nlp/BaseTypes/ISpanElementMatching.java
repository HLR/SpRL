package edu.tulane.cs.hetml.nlp.BaseTypes;


/**
 * Created by Taher on 2016-12-28.
 */
public interface ISpanElementMatching {
    boolean matches(ISpanElement xmlElement, ISpanElement element);
}
