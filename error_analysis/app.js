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
    this.trApproved = parts[8];
    this.spApproved = parts[9];
    this.lmApproved = parts[10];
    this.segments = parts[11].split(",");
    this.index = i;
    this.image = this.docId.substr(0, this.docId.lastIndexOf(".")) + ".jpg";
    this.image = this.image.substr(0, this.image.lastIndexOf("/")) + "/segmented_images" + this.image.substr(this.image.lastIndexOf("/"));
    this.image = this.image.replace("annotations", "saiapr_tc-12");

    this.tp = this.actualRel === "Relation" && this.predictedRel === "Relation";
    this.tn = this.actualRel === "None" && this.predictedRel === "None";
    this.fp = this.actualRel === "None" && this.predictedRel === "Relation";
    this.fn = this.actualRel === "Relation" && this.predictedRel === "None";
    this.errortype = this.tp ? "tp" : this.tn ? "tn" : this.fp ? "fp" : "fn";
};
var Model = function (name, data) {
    this.name = name;
    this.data = data;
    this.selected = false;
    this.tp = this.data.filter(function (x) { return x.tp }).length;
    this.fp = this.data.filter(function (x) { return x.fp }).length;
    this.fn = this.data.filter(function (x) { return x.fn }).length;
    this.tn = this.data.filter(function (x) { return x.tn }).length;

    this.accuracy = ((this.tp + this.tn) / data.length).toFixed(3);
    var p = (this.tp / (this.tp + this.fp));
    this.precision = p.toFixed(3);
    var r = (this.tp / (this.tp + this.fn));
    this.recall = r.toFixed(3);
    this.f1 = (2 * p * r / (p + r)).toFixed(3);
};
var App = function () {
    var self = this;
    self.models = ko.observableArray([]);
    self.exploreData = ko.observableArray([]);
    self.explorePage = ko.observable(0);
    self.pageSize = 10;
    self.exploreTotal = ko.observable(0);
    self.exploreTP = ko.observable(true);
    self.exploreTN = ko.observable(true);
    self.exploreFP = ko.observable(false);
    self.exploreFN = ko.observable(false);
    self.currentRel = ko.observable({index:-1})

    self.addModel = function (m) {
        self.models.push(m);
    };

    self.removeModel = function (m) {
        self.models.remove(m);
    };

    self.setSelected = function (i, selected) {
        self.models()[i].selected = selected;
    }

    self.explore = function () {
        var selected = ko.utils.arrayFirst(self.models(), function (item) {
            return item.selected;
        });
        self.exploreData([]);
        if (selected) {
            var d = selected.data.filter(function (item) {
                if (self.exploreTP() && item.tp)
                    return true;
                if (self.exploreTN() && item.tn)
                    return true;
                if (self.exploreFP() && item.fp)
                    return true;
                if (self.exploreFN() && item.fn)
                    return true;
                return false;
            });

            self.exploreTotal(Math.ceil(d.length / self.pageSize));
            var from = self.explorePage() * self.pageSize;
            var to = Math.min(d.length, (self.explorePage() + 1) * self.pageSize);
            self.exploreData(d.slice(from, to));
        }
    }

    self.toggleExploreTP = function () {
        self.exploreTP(!self.exploreTP());
        self.explore();
    }

    self.toggleExploreTN = function () {
        self.exploreTN(!self.exploreTN());
        self.explore();
    }

    self.toggleExploreFP = function () {
        self.exploreFP(!self.exploreFP());
        self.explore();
    }

    self.toggleExploreFN = function () {
        self.exploreFN(!self.exploreFN());
        self.explore();
    }

    self.nextExplorePage = function () {
        self.explorePage(Math.min(self.explorePage() + 1, self.exploreTotal()));
        self.explore();
    }

    self.prevExplorePage = function () {
        self.explorePage(Math.max(self.explorePage() - 1, 0));
        self.explore();
    }

    self.showRelDetail = function(m){
        self.currentRel(m);
        self.explore();
    }

    self.compare = function () {

    }
};

var app = new App();
ko.applyBindings(app);
