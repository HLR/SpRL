var App = function () {
    var self = this;
    self.images = ko.observableArray([]);
    self.currentImageIndex = ko.observable(-1);
    self.currentImage = ko.observable({
        relations: []
    });
    self.currentPhrase = ko.observable({
    });
    self.page = ko.observable(0);
    self.pageSize = 15;

    self.setCurrentPhrase = function (id) {
        self.currentPhrase({
            resetSeg: function () {
            }
        });
        var image = self.currentImage();
        for (var i in image.phrases) {
            if (image.phrases[i].id == id) {
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

    self.getAnnotations = function () {
        var images = self.images();
        var txt = "";        
        for (var i in images) {
            var phrases = images[i].phrases;
            for (var j in phrases) {
                var p = phrases[j];
                txt += p.imFolder + "\t\t";
                txt += p.imId + "\t\t";
                txt += p.sentId + "\t\t";
                txt += p.sentence + "\t\t";
                txt += p.start + "\t\t";
                txt += p.end + "\t\t";
                txt += p.text + "\t\t";
                txt += p.segId + "\t\t";
                txt += p.segX + "\t\t";
                txt += p.segY + "\t\t";
                txt += p.segWidth + "\t\t";
                txt += p.segHeight + "\r\n";
            }
        }
        return txt;
    }
};

var app = new App();
ko.applyBindings(app);
