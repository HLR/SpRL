var App = function () {
    var self = this;
    self.images = ko.observableArray([]);
    self.currentImageIndex = ko.observable(-1);
    self.currentImage = ko.observable({
        relations:[]
    });
    self.currentPhrase = ko.observable({
    });
    self.page = ko.observable(0);
    self.pageSize = 15;

    self.setCurrentPhrase = function (id) {
        self.currentPhrase({});
        var image = self.currentImage();
        for(var i in image.phrases){
            if(image.phrases[i].id == id){
                self.currentPhrase(image.phrases[i]);
                break;
            }
        }
    };

    self.goToImage = function (i) {
        self.currentImageIndex(i);
        self.currentImage(self.images()[i]);
        self.currentPhrase({});
    };

    self.addModel = function (m) {
        var images = Object.keys(m.images).map(function (key) { return m.images[key]; });
        self.images(images);
    };
};

var app = new App();
ko.applyBindings(app);
