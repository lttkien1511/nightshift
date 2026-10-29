"use strict";
(function () {
  var KEY = "nightshift-theme";
  var root = document.documentElement;
  var PHASES = {dawn: "Dawn", day: "Day", dusk: "Dusk", night: "Night"};

  function stored() {
    try { return localStorage.getItem(KEY) || "auto"; } catch (e) { return "auto"; }
  }
  function phaseAt(hour) {
    if (hour >= 5 && hour < 9) return "dawn";
    if (hour >= 9 && hour < 16) return "day";
    if (hour >= 16 && hour < 19) return "dusk";
    return "night";
  }
  function apply() {
    var mode = stored();
    var now = new Date();
    var hour = now.getHours() + now.getMinutes() / 60;
    root.setAttribute("data-phase", phaseAt(hour));
    root.setAttribute("data-theme", mode === "auto" ? (hour >= 6 && hour < 18 ? "light" : "dark") : mode);
    root.setAttribute("data-mode", mode);
    var label = document.getElementById("phase");
    if (label) label.textContent = PHASES[phaseAt(hour)] + " · " + now.toTimeString().slice(0, 5);
    var buttons = document.querySelectorAll(".theme button");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute("aria-pressed", String(buttons[i].getAttribute("data-mode") === mode));
    }
  }
  apply();
  setInterval(apply, 30000);

  function relative(iso) {
    var t = Date.parse(iso);
    if (isNaN(t)) return iso;
    var s = Math.round((Date.now() - t) / 1000);
    if (s < 90) return "just now";
    var m = Math.round(s / 60);
    if (m < 90) return m + " min ago";
    var h = Math.round(m / 60);
    return h < 48 ? h + " h ago" : Math.round(h / 24) + " days ago";
  }

  function ready() {
    apply();
    var modes = document.querySelectorAll(".theme button");
    for (var i = 0; i < modes.length; i++) {
      modes[i].addEventListener("click", function (e) {
        try { localStorage.setItem(KEY, e.currentTarget.getAttribute("data-mode")); } catch (err) { /* storage may be blocked */ }
        apply();
      });
    }

    var times = document.querySelectorAll("time[data-rel]");
    for (var j = 0; j < times.length; j++) times[j].textContent = relative(times[j].getAttribute("datetime"));

    var forms = document.querySelectorAll("form[data-confirm]");
    for (var k = 0; k < forms.length; k++) {
      forms[k].addEventListener("submit", function (e) {
        if (!window.confirm(e.currentTarget.getAttribute("data-confirm"))) e.preventDefault();
      });
    }

    var text = document.getElementById("text");
    var counter = document.querySelector(".counter");
    if (text && counter) {
      var max = parseInt(counter.getAttribute("data-max"), 10) || 0;
      var count = document.getElementById("count");
      var update = function () {
        var bytes = new Blob([text.value]).size;
        count.textContent = bytes.toLocaleString();
        counter.classList.toggle("over", max > 0 && bytes > max);
      };
      text.addEventListener("input", update);
      update();
      var file = document.getElementById("file");
      var name = document.getElementById("name");
      if (file) {
        file.addEventListener("change", function () {
          var chosen = file.files && file.files[0];
          if (!chosen) return;
          var reader = new FileReader();
          reader.onload = function () {
            text.value = String(reader.result || "");
            if (name && !name.value) name.value = chosen.name.replace(/\.(md|markdown|txt)$/i, "");
            update();
          };
          reader.readAsText(chosen);
        });
      }
    }

    var merge = document.getElementById("merge");
    if (merge) {
      var boxes = merge.querySelectorAll("input[name=keep]");
      var kept = document.getElementById("kept");
      var refresh = function () {
        var on = 0;
        for (var b = 0; b < boxes.length; b++) if (boxes[b].checked) on++;
        if (kept) kept.textContent = on + " of " + boxes.length + " edits kept";
      };
      for (var c = 0; c < boxes.length; c++) boxes[c].addEventListener("change", refresh);
      var pickers = merge.querySelectorAll("button[data-keep]");
      for (var p = 0; p < pickers.length; p++) {
        pickers[p].addEventListener("click", function (e) {
          var all = e.currentTarget.getAttribute("data-keep") === "all";
          for (var d = 0; d < boxes.length; d++) boxes[d].checked = all;
          refresh();
        });
      }
      refresh();
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", ready);
  else ready();
})();
