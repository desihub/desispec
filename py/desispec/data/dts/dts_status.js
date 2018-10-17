$(function() {
    var status = [];
    var onDataReceived = function(data) { return status = data; };
    var startNight = function(nights, night) {
        nights.push(night);
        return $("<div/>", {"id": ""+night});
    };
    var finishNight = function(night, rows) {
        if (rows.length > 0) {
            rows.push("</ul>");
            night.append(rows.join(""));
            night.appendTo("#container");
        }
    };
    var padExpid = function(expid) {
        var e = "" + expid;
        var z = [];
        for (var k = 0; k < 8 - e.length; k++) { z.push("0"); }
        return z.join("") + e;
    }
    var display = function() {
        $("#container").empty();
        var nights = [];
        var rows = [];
        var night;
        for (var k = 0; k < status.length; k++) {
            var n = status[k][0]
            if (nights.indexOf(n) == -1) {
                //
                // Finish previous night
                //
                finishNight(night, rows);
                //
                // Start a new night
                //
                night = startNight(nights, n)
                rows = ["<h2>Night " + n + "</h2>",
                        "<ul>"];
            }
            //
            // Add to existing night
            //
            var p = padExpid(status[k][1])
            var c = status[k][2] ? "success" : "failure";
            var l = status[k][3].length > 0 ? "Last " + status[k][3] + " exposure." : "";
            var r = "<li class=\"" + c + "\" id=\"" +
                    n + "/" + p + "\">" +
                    p + l + "</li>";
            // console.log(r);
            rows.push(r);
        }
        //
        // Finish the final night
        //
        finishNight(night, rows);
    };
    $.getJSON("dts_status.json", {}, onDataReceived).always(display);
    return true;
});
