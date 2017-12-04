var Phrase = function (line, i) {
    var self = this;
    self.resetSeg = function() {
        self.segX = self.origSegX;
        self.segY = self.origSegY;
        self.segWidth = self.origSegWidth;
        self.segHeight = self.origSegHeight;
    };

    if (line == null) {
        self.parts = [];
        self.imFolder = null;
        self.imId = null;
        self.sentId = null;
        self.sentence = null;
        self.start = -1;
        self.end = -1;
        self.text = null;
        self.id = null;
        self.segId = -1;
        self.segX = -1;
        self.segY = -1;
        self.segWidth = -1;
        self.segHeight = -1;
        self.origSegX = -1;
        self.origSegY = -1;
        self.origSegWidth = -1;
        self.origSegHeight = -1;
    }
    else {
        self.parts = line.split("\t\t");
        self.imFolder = part(0);
        self.imId = part(1);
        self.sentId = part(2);
        self.sentence = part(3);
        self.start = parseInt(part(4));
        self.end = parseInt(part(5));
        self.text = part(6);
        self.id = self.imFolder + "_" + self.imId + "_" + self.sentId + "_" + self.start + "_" + self.end;
        self.segId = parseInt(part(7));
        self.segX = parseInt(part(8));
        self.segY = parseInt(part(9));
        self.segWidth = parseInt(part(10));
        self.segHeight = parseInt(part(11));
        self.origSegX = self.segX;
        self.origSegY = self.segY;
        self.origSegWidth = self.segWidth;
        self.origSegHeight = self.segHeight;
    }

    function part(i) {
        if (i >= self.parts.length)
            return "";
        return self.parts[i];
    }
};

var Relation = function (line, i) {
    var self = this;
    self.parts = line.split("\t\t");
    self.imFolder = parts(0);
    self.imId = parts(1);
    self.sentId = parts(2);
    self.sentence = parts(3);

    self.trId = self.imFolder + "_" + self.imId + "_" + self.sentId + "_" + parseInt(parts(4)) + "_" + parseInt(parts(5));
    self.spId = self.imFolder + "_" + self.imId + "_" + self.sentId + "_" + parseInt(parts(7)) + "_" + parseInt(parts(8));
    self.lmId = self.imFolder + "_" + self.imId + "_" + self.sentId + "_" + parseInt(parts(10)) + "_" + parseInt(parts(11));

    self.tr = null
    self.sp = null
    self.lm = null


    function parts(i) {
        if (i >= self.parts.length)
            return "";
        return self.parts[i];
    }
};

var Model = function (phrases, relations) {
    var self = this;
    self.images = {};
    phraseDic = {};
    for (var i in phrases) {
        var imId = phrases[i].imId;
        var imFolder = phrases[i].imFolder;
        if (self.images[imId]) {
            self.images[imId].phrases.push(phrases[i])
        }
        else {
            self.images[imId] = {
                'imFolder': imFolder,
                'imId': imId,
                'phrases': [phrases[i]],
                'relations': []
            }
        }
        phraseDic[phrases[i].id] = phrases[i];
    }
    for (var i in relations) {
        var imId = relations[i].imId;
        var imFolder = relations[i].imFolder;
        self.images[imId].relations.push(relations[i])
        relations[i].tr = phraseDic[relations[i].trId];
        relations[i].sp = phraseDic[relations[i].spId];
        relations[i].lm = phraseDic[relations[i].lmId];
        if (relations[i].lm === undefined) {
            relations[i].lm = new Phrase();
            relations[i].lm.imFolder = relations[i].imFolder;
            relations[i].lm.imId = relations[i].imId;
            relations[i].lm.sentId = relations[i].sentId;
            relations[i].lm.sentence = relations[i].sentence;
            relations[i].lm.id = relations[i].lmId;
        }
    }
}
