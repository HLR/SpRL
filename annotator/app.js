var App = function () {
    var self = this;
    self.images = ko.observableArray([]);
    self.currentImageIndex = ko.observable(-1);
    self.currentImage = ko.observable({
        relations:[]
    });
    self.currentRelation = ko.observable({});
    self.currentRelationIndex = ko.observable(-1);
    self.page = ko.observable(0);
    self.pageSize = 15;

    self.goToImage = function (i) {
        self.currentImageIndex(i);
        self.currentImage(self.images()[i]);
        self.currentRelation(-1);
    };

    self.goToRelation = function (i) {
        self.currentRelation(self.currentImage().relations[i]);
        self.currentRelationIndex(i);
    };

    self.addModel = function (m) {
        var images = Object.keys(m.images).map(function (key) { return m.images[key]; });
        self.images(images);
    };
};

var app = new App();
ko.applyBindings(app);
