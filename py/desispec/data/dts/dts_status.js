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
            night.appendTo("#content");
        }
    };
    var padExpid = function(expid) {
        var e = ("" + expid).split("");
        while (e.length < 8) e.unshift("0");
        return e.join("");
    };
    var nightButton = function(n, role, success) {
        var color = success ? "btn-success" : "btn-danger";
        if (role == "show") {
            return "<button type=\"button\" class=\"btn " + color +
                " btn-sm\" id=\"show" + n +
                "\" style=\"display:inline;\" onclick=\"$('#ul" + n +
                "').css('display', 'block');$('#hide" + n +
                "').css('display', 'inline');$('#show" + n +
                "').css('display', 'none');\">Show</button>";
        } else {
            return "<button type=\"button\" class=\"btn " + color +
                " btn-sm\" id=\"hide" + n +
                "\" style=\"display:none;\" onclick=\"$('#ul" + n +
                "').css('display', 'none');$('#show" + n +
                "').css('display', 'inline');$('#hide" + n +
                "').css('display', 'none');\">Hide</button>";
        }
    };
    var display = function() {
        $("#content").empty();
        var nights = [];
        var rows = [];
        var night;
        for (var k = 0; k < status.length; k++) {
            var n = status[k][0];
            if (nights.indexOf(n) == -1) {
                //
                // Finish previous night
                //
                finishNight(night, rows);
                //
                // Start a new night
                //
                night = startNight(nights, n);
                rows = ["<p>Night " + n,
                        nightButton(n, "show", true),
                        nightButton(n, "hide", true),
                        "</p>",
                        "<ul id=\"ul" + n + "\" style=\"display:none;\">"];
            }
            //
            // Add to existing night
            //
            var p = padExpid(status[k][1]);
            var c = status[k][2] ? "bg-success" : "bg-danger";
            if (!status[k][2]) {
                rows[1] = nightButton(n, "show", false);
                rows[2] = nightButton(n, "hide", false);
            }
            var l = status[k][3].length > 0 ? " Last " + status[k][3] + " exposure." : "";
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
