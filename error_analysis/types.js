var Instance = function (line, i) {
    var parts = line.split("\t\t");
    this.docId = "";
    this.docId = parts[0];
    this.sentId = parts[1];
    this.sent = parts[2];
    this.actualRel = parts[3];
    this.predictedRel = parts[4];
    this.tr = parts[5];
    this.sp = parts[6];
    this.lm = parts[7];
    this.trApproved = parts[8] == 'true';
    this.spApproved = parts[9] == 'true';
    this.lmApproved = parts[10] == 'true';
    this.segments = parts[11].split(",");
    this.index = i;
    this.image = this.docId.substr(0, this.docId.lastIndexOf(".")) + ".jpg";
    this.image = this.image.substr(0, this.image.lastIndexOf("/")) + "/segmented_images" + this.image.substr(this.image.lastIndexOf("/"));
    this.image = this.image.replace("annotations", "saiapr_tc-12");

    this.tp = this.actualRel === "Relation" && this.predictedRel === "Relation";
    this.tn = this.actualRel === "None" && this.predictedRel === "None";
    this.fp = this.actualRel === "None" && this.predictedRel === "Relation";
    this.fn = this.actualRel === "Relation" && this.predictedRel === "None";
    this.correct = this.tp || this.tn;
    this.errortype = this.tp ? "tp" : this.tn ? "tn" : this.fp ? "fp" : "fn";

    this.equalRoles = function(d){
        return this.trApproved == d.trApproved && this.lmApproved == d.lmApproved && this.spApproved == d.spApproved;
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
            if (this.data[i].correct) {
                if(m.data[i].correct){
                    c.cc.push(i);
                }else{
                    c.ci.push(i);
                }
            }
            else{
                if(m.data[i].correct){
                    c.ic.push(i);
                }else{
                    c.ii.push(i);
                }  
            }
        }
        return c;
    }
};
