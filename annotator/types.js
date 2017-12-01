var Phrase = function (imFolder, imId, sentId, sentence, start, end, text) {
    var self = this;
    self.imFolder = imFolder;
    self.imId = imId;
    self.sentId = sentId;
    self.sentence = sentence;
    self.start = start;
    self.end = end;
    self.text = text;
};

var Relation = function (line, i) {
    var self = this;
    self.parts = line.split("\t\t");
    self.imFolder = parts(0);
    self.imId = parts(1);
    self.sentId = parts(2);
    
    self.trStart = parts(3);
    self.trEnd = parts(4);
    self.trText = parts(5);
    
    self.spStart = parts(6);
    self.spEnd = parts(7);
    self.spText = parts(8);
    
    self.lmStart = parts(9);
    self.lmEnd = parts(10);
    self.lmText = parts(11);


    function parts(i) {
        if (i >= self.parts.length)
            return "";
        return self.parts[i];
    }
};
function parsePhrase(line, i) {
    var p = Phrase();
    p.parts = line.split("\t\t");
    p.imFolder = parts(0);
    p.imId = parts(1);
    p.sentId = parts(2);
    p.sentence = parts(3);
    p.start = parseInt(parts(4));
    p.end = parseInt(parts(5));
    p.text = parts(6);

    function parts(i) {
        if (i >= p.parts.length)
            return "";
        return p.parts[i];
    }
};
var Model = function (phrases, relations){
    var self = this;
    self.images = {};
    for(var i in phrases){
        var imId = phrases[i].imId;
        var imFolder = phrases[i].imFolder;
        if(self.images[imId]){
            self.images[imId].phrases.push(phrases[i])
        }
        else{
            self.images[imId] = {
                'imFolder' : imFolder,
                'phrases': [],
            }
        }

    }
    self.relations = relations;
    self.phrases = phrases;
}
