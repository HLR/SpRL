var App = function () {
    var self = this;
    self.data = ko.observableArray([]);
    self.fullData = []
    self.page = ko.observable(0);
    self.pageSize = 15;
    self.selected = ko.observable(0);
    self.total = ko.observable(0);
    self.totalRels = ko.observable(0);

    self.refresh = function(){
        var start = self.page() * self.pageSize;
        var end = start + self.pageSize;
        self.data(self.fullData.slice(start, end));
    }
    self.setData = function (d) {
        self.fullData = d;
        self.totalRels(d.length);
        self.total(Math.ceil(d.length / self.pageSize));
        self.refresh();
    };

    self.nextPage = function () {
        self.page(Math.min(self.page() + 1, self.total() - 1));
        self.refresh();
    };

    self.prevPage = function () {
        self.page(Math.max(self.page() - 1, 0));
        self.refresh();
    };

    self.setSeleced = function(i){
        self.selected(i);
    }
};

var app = new App();
ko.applyBindings(app);
