$(function() {
    var status = [];
    var onDataReceived = function(data) { return status = data; };
    var startNight = function(nights, night) {
        nights.push(night);
        return $("<div/>", {"id": ""+night, "class": "row"});
    };
    var finishNight = function(night, buttons, b_rows, ul, ul_rows) {
        if (b_rows.length > 0) {
            ul_rows.push("</ul>");
            buttons.append(b_rows.join(""));
            ul.append(ul_rows.join(""));
            night.append(buttons);
            night.append(ul);
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
        var b_rows = [];
        var ul_rows = [];
        var night, buttons, ul;
        for (var k = 0; k < status.length; k++) {
            var n = status[k][0];
            if (nights.indexOf(n) == -1) {
                //
                // Finish previous night
                //
                finishNight(night, buttons, b_rows, ul, ul_rows);
                //
                // Start a new night
                //
                night = startNight(nights, n);
                buttons = $("<div/>", {"class": "col-4"});
                ul = $("<div/>", {"class": "col-8"});
                b_rows = ["<p id=\"p" + n + "\">Night " + n + "&nbsp;",
                          nightButton(n, "show", true),
                          nightButton(n, "hide", true),
                          "</p>"];
                ul_rows = ["<ul id=\"ul" + n + "\" style=\"display:none;\">"];
            }
            //
            // Add to existing night
            //
            var p = padExpid(status[k][1]);
            var c = status[k][2] ? "bg-success" : "bg-danger";
            if (!status[k][2]) {
                b_rows[1] = nightButton(n, "show", false);
                b_rows[2] = nightButton(n, "hide", false);
            }
            var l = status[k][3].length > 0 ? " Last " + status[k][3] + " exposure." : "";
            var r = "<li class=\"" + c + "\" id=\"" +
                    n + "/" + p + "\">" +
                    p + l + "</li>";
            // console.log(r);
            ul_rows.push(r);
        }
        //
        // Finish the final night
        //
        finishNight(night, buttons, b_rows, ul, ul_rows);
    };
    $.getJSON("dts_status.json", {}, onDataReceived).always(display);
    return true;
});
