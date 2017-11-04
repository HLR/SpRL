var App = function () {
    var self = this;
    self.models = ko.observableArray([]);
    self.data = ko.observableArray([]);
    self.page = ko.observable(0);
    self.pageSize = 15;
    self.total = ko.observable(0);
    self.totalRels = ko.observable(0);
    self.TP = ko.observable(false);
    self.TN = ko.observable(false);
    self.FP = ko.observable(false);
    self.FN = ko.observable(true);
    self.CC = ko.observable(false);
    self.CI = ko.observable(false);
    self.IC = ko.observable(false);
    self.II = ko.observable(false);
    self.diffRole = ko.observable(false);
    self.currentRel = ko.observable({ index: -1 });
    self.allRolesCorrect = ko.observable(false);
    self.comparison = ko.observable({
        first: {},
        second: {},
        cc: [],
        ci: [],
        ic: [],
        ii: []
    });

    self.addModel = function (m) {
        self.models.push(m);
    };

    self.removeModel = function (m) {
        self.models.remove(m);
    };

    self.setSelected = function (i, selected) {
        self.models()[i].selected = selected;
    };

    self.explore = function () {
        var selected = ko.utils.arrayFirst(self.models(), function (item) {
            return item.selected;
        });
        self.data([]);
        if (selected) {
            var d = selected.data.filter(function (item, i) {
                if (self.allRolesCorrect() && !(item.trApproved && item.spApproved && item.lmApproved))
                    return false;

                if (self.comparison().first !== {} && self.comparison().second !== {}) {
                    if (self.diffRole() && self.models()[0].data[i].equalRoles(self.models()[1].data[i]))
                        return false;
                    if (self.CC() && self.comparison().cc.indexOf(i) < 0)
                        return false;

                    if (self.CI() && self.comparison().ci.indexOf(i) < 0)
                        return false;

                    if (self.IC() && self.comparison().ic.indexOf(i) < 0)
                        return false;

                    if (self.II() && self.comparison().ii.indexOf(i) < 0)
                        return false;

                }

                if (self.TP() && item.tp)
                    return true;
                if (self.TN() && item.tn)
                    return true;
                if (self.FP() && item.fp)
                    return true;
                if (self.FN() && item.fn)
                    return true;
                return false;
            });

            self.totalRels(d.length);
            self.total(Math.ceil(d.length / self.pageSize));
            var from = self.page() * self.pageSize;
            var to = Math.min(d.length, (self.page() + 1) * self.pageSize);
            self.data(d.slice(from, to));
        }
    };


    self.toggleAllRolesCorrect = function () {
        self.allRolesCorrect(!self.allRolesCorrect());
        self.explore();
        return true;
    };

    self.toggleCC = function () {
        self.CC(!self.CC());
        self.explore();
        return true;
    };

    self.toggleCI = function () {
        self.CI(!self.CI());
        self.explore();
        return true;
    };

    self.toggleIC = function () {
        self.IC(!self.IC());
        self.explore();
        return true;
    };

    self.toggleII = function () {
        self.II(!self.II());
        self.explore();
        return true;
    };

    self.toggleTP = function () {
        self.TP(!self.TP());
        self.explore();
        return true;
    };

    self.toggleTN = function () {
        self.TN(!self.TN());
        self.explore();
        return true;
    };

    self.toggleFP = function () {
        self.FP(!self.FP());
        self.explore();
        return true;
    };

    self.toggleFN = function () {
        self.FN(!self.FN());
        self.explore();
        return true;
    };

    self.toggleDiffRole = function () {
        self.diffRole(!self.diffRole());
        self.explore();
        return true;
    };

    self.nextPage = function () {
        self.page(Math.min(self.page() + 1, self.total() - 1));
        self.explore();
    };

    self.prevPage = function () {
        self.page(Math.max(self.page() - 1, 0));
        self.explore();
    };

    self.showRelDetail = function (m) {
        self.currentRel(m);
        self.explore();
    };

    self.compare = function () {
        var selected = self.models().filter(function (item) {
            return item.selected;
        });
        if (selected.length != 2) {
            alert("please select two models to compare!");
            return;
        }
        self.comparison(selected[0].compareWith(selected[1]));
    };

    self.trApproved = function (rel) {
        var str = "";
        for (i in self.models()) {
            str += self.models()[i].data[rel.index].trApproved ? '\u2713' : '\u00D7';
        }
        return str;
    }

    self.lmApproved = function (rel) {
        var str = "";
        for (i in self.models()) {
            str += self.models()[i].data[rel.index].lmApproved ? '\u2713' : '\u00D7';
        }
        return str;
    }

    self.spApproved = function (rel) {
        var str = "";
        for (i in self.models()) {
            str += self.models()[i].data[rel.index].spApproved ? '\u2713' : '\u00D7';
        }
        return str;
    }

    self.predicted = function (rel) {
        var str = "";
        for (i in self.models()) {
            str += self.models()[i].data[rel.index].predictedRel + " ";
        }
        return str;
    }

    self.reset = function () {
        self.models([]);
        self.data([]);
        self.page(0);
        self.total(0);
        self.totalRels(0);
        self.TP(false);
        self.TN(false);
        self.FP(false);
        self.FN(true);
        self.CC(false);
        self.CI(false);
        self.IC(false);
        self.II(false);
        self.currentRel({ index: -1 });
        self.allRolesCorrect(false);
        self.comparison({
            first: {},
            second: {},
            cc: [],
            ci: [],
            ic: [],
            ii: []
        });
    }
};

var app = new App();
ko.applyBindings(app);
