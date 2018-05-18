var Instance = function (line, i) {
    var self = this;
    self.parts = line.split("\t\t");
    self.imId = parts(0);
    self.relId = parts(1);
    self.trSegId = parts(2);
    self.tr = parts(3);
    self.lmSegId = parts(4);
    self.lm = parts(5);
    self.sp = parts(6);
    self.predicted = parts(7);
    self.actual = parts(8);
    self.trX = parts(9);
    self.trY = parts(10);
    self.trW = parts(11);
    self.trH = parts(12);
    self.lmX = parts(13);
    self.lmY = parts(14);
    self.lmW = parts(15);
    self.lmH = parts(16);

    function parts(i) {
        if (i >= self.parts.length)
            return "";
        return self.parts[i];
    }
};