package edu.tulane.cs.hetml.vision;

import org.languagetool.JLanguageTool;
import org.languagetool.language.*;
import org.languagetool.rules.*;

import java.io.IOException;
import java.util.List;


public class LanguageHelper {
    public LanguageHelper() {

    }
    public String wordSpellVerifier(String word) throws IOException {
        JLanguageTool langTool = new JLanguageTool(new BritishEnglish());
        for (Rule rule : langTool.getAllRules()) {
            if (!rule.isDictionaryBasedSpellingRule()) {
                langTool.disableRule(rule.getId());
            }
        }
        List<RuleMatch> matches = langTool.check(word);
        if(matches.size()==0 || matches.get(0).getSuggestedReplacements().size()==0)
            return "true";
        else
            return matches.get(0).getSuggestedReplacements().get(0);
    }
}
