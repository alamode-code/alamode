/* 3D structures and phonon-mode animations for the ALAMODE docs
   (see sphinxext/alamode_structures.py). Each <figure class="alamode-structure">
   carries the cell and the frames as JSON; 3Dmol.js draws them on a transparent
   background, and the cell box and arrows follow the page theme. */
(function () {
  "use strict";
  var viewers = [];
  var ARROW = "#4a90d9";  // reads on both the light and the dark page (as alamode_plots.js)

  // the theme text colour as #rrggbb (the theme may give oklch())
  function textColor() {
    var ctx = document.createElement("canvas").getContext("2d");
    ctx.fillStyle = getComputedStyle(document.body).color || "#333";
    ctx.fillRect(0, 0, 1, 1);
    var m = ctx.getImageData(0, 0, 1, 1).data;
    return "#" + [m[0], m[1], m[2]].map(function (v) { return (v | 256).toString(16).slice(1); }).join("");
  }

  function elementColor(el) {
    var c = $3Dmol.elementColors.Jmol[el];
    return c === undefined ? "#ff1493" : "#" + (c | 0x1000000).toString(16).slice(1);
  }

  function xyz(d, f) {
    var s = d.elements.length + "\n\n";
    d.elements.forEach(function (el, i) {
      s += el + " " + f[3 * i] + " " + f[3 * i + 1] + " " + f[3 * i + 2] + "\n";
    });
    return s;
  }

  // quaternions [x, y, z, w]: rotation by deg about axis, and the product a*b
  function qaxis(u, deg) {
    var h = deg * Math.PI / 360, s = Math.sin(h);
    return [u[0] * s, u[1] * s, u[2] * s, Math.cos(h)];
  }
  function qmul(a, b) {
    return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
  }

  function vec(p) { return { x: p[0], y: p[1], z: p[2] }; }

  // cell box and displacement arrows, redrawn when the theme changes
  function drawShapes(v) {
    var L = v.data.lattice, color = textColor();
    v.viewer.removeAllShapes();
    var corner = function (i, j, k) {
      return [0, 1, 2].map(function (c) { return i * L[0][c] + j * L[1][c] + k * L[2][c]; });
    };
    for (var a = 0; a < 3; a++)
      for (var p = 0; p < 2; p++)
        for (var q = 0; q < 2; q++) {
          var s = [0, 0, 0], e;
          s[(a + 1) % 3] = p; s[(a + 2) % 3] = q;
          e = s.slice(); e[a] = 1;
          v.viewer.addCylinder({ start: vec(corner.apply(null, s)), end: vec(corner.apply(null, e)),
                                 radius: 0.03, color: color, fromCap: 1, toCap: 1 });
        }
    // arrows along each atom's displacement, starting at about the sphere surface (0.5 A)
    (v.data.arrows || []).forEach(function (r) {
      var n = Math.hypot(r[3], r[4], r[5]);
      var at = function (t) { return vec([0, 1, 2].map(function (k) { return r[k] + t * r[k + 3] / n; })); };
      v.viewer.addArrow({ start: at(0.5), end: at(0.75 + 0.9 * n), radius: 0.07, radiusRatio: 2.2,
                          mid: 0.65, color: ARROW });
    });
  }

  function button(label, title, onclick) {
    var b = document.createElement("button");
    b.type = "button"; b.textContent = label; b.title = title;
    b.setAttribute("aria-label", title);
    b.addEventListener("click", onclick);
    return b;
  }

  function create(fig) {
    var d = JSON.parse(fig.querySelector(".alamode-structure-data").textContent);
    var box = fig.querySelector(".alamode-structure-canvas");
    box.style.height = d.height + "px";
    var host = document.createElement("div");
    host.className = "alamode-structure-gl";
    box.appendChild(host);
    var viewer = $3Dmol.createViewer(host, { backgroundColor: "white", backgroundAlpha: 0, antialias: true });
    if (!viewer) throw new Error("WebGL is not available");
    viewer.setBackgroundColor(0xffffff, 0);
    var text = d.frames.map(function (f) { return xyz(d, f); }).join("");
    if (d.frames.length > 1) viewer.addModelsAsFrames(text, "xyz");
    else viewer.addModel(text, "xyz");
    var ball = { sphere: { scale: 0.28, colorscheme: "Jmol" } };
    var stick = { sphere: ball.sphere, stick: { radius: 0.11, colorscheme: "Jmol" } };
    viewer.setStyle({}, d.bonds ? ball : stick);
    if (d.bonds) viewer.setStyle({ elem: d.bonds }, stick);
    var v = { viewer: viewer, data: d };
    drawShapes(v);
    // z up, seen slightly from above and from the side
    viewer.zoomTo();
    var home = viewer.getView();
    home.splice(4, 4, ...qmul(qaxis([1, 0, 0], 15), qmul(qaxis([0, 1, 0], -25), qaxis([1, 0, 0], -90))));
    viewer.setView(home);
    home = viewer.getView();
    viewer.render();

    var legend = document.createElement("div");
    legend.className = "alamode-structure-legend";
    d.elements.filter(function (el, i, a) { return a.indexOf(el) === i; }).forEach(function (el) {
      var item = document.createElement("span"), dot = document.createElement("span");
      dot.className = "alamode-structure-dot";
      dot.style.background = elementColor(el);
      item.appendChild(dot);
      item.appendChild(document.createTextNode(el));
      legend.appendChild(item);
    });
    box.appendChild(legend);

    var bar = document.createElement("div");
    bar.className = "alamode-structure-controls";
    bar.appendChild(button("↺", "Reset view", function () { viewer.setView(home); viewer.render(); }));
    if (d.frames.length > 1) {
      var still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      var play = button("", "", function () {
        if (viewer.isAnimated()) viewer.pauseAnimate(); else viewer.resumeAnimate();
        label();
      });
      var label = function () {
        var on = viewer.isAnimated();
        play.textContent = on ? "❚❚" : "▶";
        play.title = on ? "Pause" : "Play";
        play.setAttribute("aria-label", play.title);
      };
      viewer.animate({ loop: "forward", interval: 80 });
      if (still) viewer.pauseAnimate();
      label();
      bar.appendChild(play);
      // Pause while the viewer is off-screen; resume only if it was playing.
      if ("IntersectionObserver" in window) {
        var wasOn = false;
        new IntersectionObserver(function (entries) {
          entries.forEach(function (e) {
            if (!e.isIntersecting) { wasOn = viewer.isAnimated(); if (wasOn) viewer.pauseAnimate(); }
            else if (wasOn) { viewer.resumeAnimate(); wasOn = false; }
            label();
          });
        }).observe(box);
      }
    }
    box.appendChild(bar);
    viewers.push(v);
  }

  function restyle() {
    viewers.forEach(function (v) { drawShapes(v); v.viewer.render(); });
  }

  // Show the static image kept in <noscript> instead of an empty canvas.
  function showFallback(fig) {
    var el = fig.querySelector(".alamode-structure-canvas");
    var ns = fig.querySelector("noscript");
    if (el && ns) el.outerHTML = ns.textContent;
  }

  function init() {
    var figs = document.querySelectorAll("figure.alamode-structure");
    if (typeof $3Dmol === "undefined") { figs.forEach(showFallback); return; }
    figs.forEach(function (fig) {
      try { create(fig); } catch (err) {
        console.error("alamode_structures:", err);
        fig.querySelectorAll(".alamode-structure-canvas > *").forEach(function (c) { c.remove(); });
        showFallback(fig);
      }
    });
    new MutationObserver(restyle).observe(document.documentElement,
      { attributes: true, attributeFilter: ["data-theme"] });
    if (window.matchMedia)
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", restyle);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
