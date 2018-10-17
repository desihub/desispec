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
    var display = function() {
        $("#container").empty();
        var nights = [];
        var rows = [];
        var night;
        for (var k = 0; k < status.length; k++) {
            if (nights.indexOf(status[k][0]) == -1) {
                //
                // Finish previous night
                //
                finishNight(night, rows);
                //
                // Start a new night
                //
                night = startNight(nights, status[k][0])
                rows = ["<h2>Night " + status[k][0] + "</h2>",
                        "<ul>"];
            }
            //
            // Add to existing night
            //
            var r = "<li class=\"" + status[k][2] + "\" id=\"" +
                    status[k][0] + "/" + status[k][1] + "\">" +
                    status[k][1] + status[k][3] + "</li>";
            console.log(r);
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
