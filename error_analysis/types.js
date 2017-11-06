var Instance = function (line, i) {
    var self = this;
    self.parts = line.split("\t\t");
    self.docId = "";
    self.docId = parts(0);
    self.sentId = parts(1);
    self.sent = parts(2);
    self.actualRel = parts(3);
    self.predictedRel = parts(4);
    self.tr = parts(5);
    self.sp = parts(6);
    self.lm = parts(7);
    self.trApproved = parts(8) == 'true';
    self.spApproved = parts(9) == 'true';
    self.lmApproved = parts(10) == 'true';
    self.segments = parts(11).split(",");
    self.trDis = parts(12);
    self.lmDis = parts(13);
    self.index = i;
    self.image = self.docId.substr(0, self.docId.lastIndexOf(".")) + ".jpg";
    self.image = self.image.substr(0, self.image.lastIndexOf("/")) + "/segmented_images" + self.image.substr(self.image.lastIndexOf("/"));
    self.image = self.image.replace("annotations", "saiapr_tc-12");

    self.tp = self.actualRel === "Relation" && self.predictedRel === "Relation";
    self.tn = self.actualRel === "None" && self.predictedRel === "None";
    self.fp = self.actualRel === "None" && self.predictedRel === "Relation";
    self.fn = self.actualRel === "Relation" && self.predictedRel === "None";
    self.correct = self.tp || self.tn;
    self.errortype = self.tp ? "tp" : self.tn ? "tn" : self.fp ? "fp" : "fn";

    self.equalRoles = function(d){
        return self.trApproved == d.trApproved && self.lmApproved == d.lmApproved && self.spApproved == d.spApproved;
    }

    self.equalRels = function(d){
        return self.predictedRel == d.predictedRel;
    }

    function parts(i){
        if(i >= self.parts.length)
            return "";
        return self.parts[i];
    }
};

var Model = function (name, data) {
    var self = this;
    self.name = name;
    self.data = data;
    self.selected = false;
    self.tp = self.data.filter(function (x) { return x.tp }).length;
    self.fp = self.data.filter(function (x) { return x.fp }).length;
    self.fn = self.data.filter(function (x) { return x.fn }).length;
    self.tn = self.data.filter(function (x) { return x.tn }).length;

    self.accuracy = ((self.tp + self.tn) / data.length).toFixed(3);
    var p = (self.tp / (self.tp + self.fp));
    self.precision = p.toFixed(3);
    var r = (self.tp / (self.tp + self.fn));
    self.recall = r.toFixed(3);
    self.f1 = (2 * p * r / (p + r)).toFixed(3);

    self.compareWith = function (m) {
        var c = {
            first: self,
            second: m,
            cc: [],
            ci: [],
            ic: [],
            ii: [],
            
        };
        for (var i =0; i<self.data.length; i++) {
            if (self.data[i].correct) {
                if(m.data[i].correct){
                    c.cc.push(i);
                }
                else{
                    c.ci.push(i);
                }
            }
            else{
                if(m.data[i].correct){
                    c.ic.push(i);
                }
                else{
                    c.ii.push(i);
                }  
            }
        }
        return c;
    }
};
