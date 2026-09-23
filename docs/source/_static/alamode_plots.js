/* Interactive plots for the ALAMODE docs (see sphinxext/alamode_plots.py).
   Each <figure class="alamode-plot"> carries its data as JSON; Plotly draws it
   and the colours follow the page theme (breeze's data-theme on <html>). */
(function () {
  "use strict";
  // mid-lightness colours that read on both the light and the dark page, in
  // the order of tools/plotband.py (blue, green, red, magenta) so that the
  // tutorial text ("the green curve ...") matches
  var PALETTE = ["#4a90d9", "#3aa876", "#e0524f", "#c451c9", "#e6862e", "#9b7ae0"];
  var plots = [];

  function themeColors() {
    // the theme may give oklch() colours: let a canvas convert to rgb
    var ctx = document.createElement("canvas").getContext("2d");
    ctx.fillStyle = getComputedStyle(document.body).color || "#333";
    ctx.fillRect(0, 0, 1, 1);
    var m = ctx.getImageData(0, 0, 1, 1).data;
    var text = "rgb(" + m[0] + "," + m[1] + "," + m[2] + ")";
    var rgba = function (a) { return "rgba(" + m[0] + "," + m[1] + "," + m[2] + "," + a + ")"; };
    return { text: text, grid: rgba(0.15), line: rgba(0.55) };
  }

  function axis(c, extra) {
    var a = {
      color: c.text, gridcolor: c.grid, linecolor: c.line, zerolinecolor: c.line,
      showline: true, mirror: true, ticks: "outside", tickcolor: c.line, automargin: true
    };
    for (var k in extra) a[k] = extra[k];
    return a;
  }

  // x axis with the high-symmetry labels and vertical lines at them
  function kAxis(d, c) {
    return axis(c, {
      tickmode: "array", tickvals: d.ticks, ticktext: d.ticklabels, showgrid: true,
      gridcolor: c.grid, range: [d.x[0], d.x[d.x.length - 1]], zeroline: false, ticklen: 4
    });
  }

  function kLabels(d) {
    // hover text: the high-symmetry label when on one, else the path coordinate
    return d.x.map(function (x) {
      for (var i = 0; i < d.ticks.length; i++)
        if (Math.abs(d.ticks[i] - x) < 1e-3) return d.ticklabels[i] + " (" + x.toFixed(3) + ")";
      return "k = " + x.toFixed(3);
    });
  }

  function bandTraces(d, name, color, dash, group, xaxis) {
    var text = kLabels(d);
    return d.branches.map(function (y, i) {
      return {
        type: "scatter", mode: "lines", x: d.x, y: y, text: text, name: name,
        legendgroup: group, showlegend: i === 0, xaxis: xaxis || "x",
        line: { color: color, width: dash ? 1.5 : 2, dash: dash || "solid" },
        hovertemplate: "%{text}<br>%{y:.1f} cm⁻¹<br>" + name + ", branch " + (i + 1) + "<extra></extra>"
      };
    });
  }

  function legend(c) {
    // above the plot area, so it never covers a curve
    return { orientation: "h", x: 0, y: 1.02, xanchor: "left", yanchor: "bottom",
             font: { color: c.text }, bgcolor: "rgba(0,0,0,0)" };
  }

  function build(d, c) {
    var traces = [], layout = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: c.text, size: 13 }, height: d.height,
      margin: { l: 60, r: 15, t: 30, b: 45 }, hovermode: "closest",
      hoverlabel: { namelength: -1 }, showlegend: false
    };
    if (d.kind === "dispersion" || d.kind === "dispersion-compare") {
      d.series.forEach(function (s, i) {
        traces = traces.concat(bandTraces(s, s.name, PALETTE[i % PALETTE.length], null, "s" + i));
      });
      layout.xaxis = kAxis(d.series[0], c);
      var lo = 0, hi = 0;
      d.series.forEach(function (s) { s.branches.forEach(function (b) {
        lo = Math.min(lo, Math.min.apply(null, b)); hi = Math.max(hi, Math.max.apply(null, b)); }); });
      // the DOS may extend past the bands: keep the band range
      layout.yaxis = axis(c, { title: { text: d.ylabel }, zeroline: false,
                               range: [lo - 0.03 * (hi - lo), hi + 0.05 * (hi - lo)] });
      if (d.series.length > 1) { layout.showlegend = true; layout.legend = legend(c); }
      if (d.dos) {
        layout.xaxis.domain = [0, 0.76];
        layout.xaxis2 = axis(c, { domain: [0.79, 1], showticklabels: false, ticks: "",
                                  showgrid: false, zeroline: false, title: { text: "DOS" }, rangemode: "tozero" });
        traces.push({
          type: "scatter", mode: "lines", x: d.dos.dos, y: d.dos.e, xaxis: "x2", name: "DOS",
          line: { color: c.text, width: 1.5 }, fill: "tozerox", fillcolor: c.grid,
          hovertemplate: "%{y:.0f} cm⁻¹<br>DOS %{x:.3g}<extra></extra>"
        });
        layout.margin.t = 15;
      }
    } else if (d.kind === "dispersion-temperature") {
      var i0 = Math.max(0, d.temps.indexOf(d.t0));
      var n = d.frames[0].length;
      traces = bandTraces({ x: d.x, ticks: d.ticks, ticklabels: d.ticklabels, branches: d.frames[i0] },
                          d.name, PALETTE[0], null, "scph");
      traces.forEach(function (t) {
        t.hovertemplate = t.hovertemplate.replace("<extra>", "<br>T = %{meta} K<extra>");
        t.meta = d.temps[i0];
      });
      if (d.reference)
        traces = traces.concat(bandTraces(d.reference, d.reference.name, PALETTE[1], "dash", "ref"));
      // legend entry for the shaded half-plane below (after the band traces:
      // the slider restyles traces 0..n-1)
      traces.push({ type: "scatter", mode: "markers", x: [null], y: [null], hoverinfo: "skip",
                    name: "Imaginary (plotted as ω < 0)",
                    marker: { symbol: "square", size: 12, color: "rgba(214,85,85,0.30)" } });
      var idx = []; for (var j = 0; j < n; j++) idx.push(j);
      layout.xaxis = kAxis(d, c);
      layout.yaxis = axis(c, { title: { text: d.ylabel }, zeroline: true, zerolinewidth: 1.5 });
      // imaginary (soft) modes are negative: shade that half-plane
      layout.shapes = [{ type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0: -1e4, y1: 0,
                         fillcolor: "rgba(214,85,85,0.12)", line: { width: 0 }, layer: "below" }];
      var lo = 0;
      if (d.reference) d.reference.branches.forEach(function (b) { lo = Math.min(lo, Math.min.apply(null, b)); });
      d.frames.forEach(function (f) { f.forEach(function (b) { lo = Math.min(lo, Math.min.apply(null, b)); }); });
      var hi = -Infinity;
      d.frames.forEach(function (f) { f.forEach(function (b) { hi = Math.max(hi, Math.max.apply(null, b)); }); });
      var pad = 0.04 * (hi - lo);
      layout.yaxis.range = [lo - pad, hi + pad];
      layout.showlegend = true; layout.legend = legend(c);
      layout.margin.b = 40;
      layout.sliders = [{
        active: i0, pad: { t: 30 }, len: 1, x: 0,
        currentvalue: { prefix: "T = ", suffix: " K", font: { color: c.text } },
        font: { color: c.text }, bgcolor: c.grid, bordercolor: c.line, activebgcolor: PALETTE[0],
        tickcolor: c.line,
        steps: d.temps.map(function (t, k) {
          return { label: String(t), method: "restyle", args: [{ y: d.frames[k], meta: t }, idx] };
        })
      }];
      layout.height = d.height + 90;
    } else {
      var logx = d.log.indexOf("x") >= 0, logy = d.log.indexOf("y") >= 0;
      d.series.forEach(function (s, i) {
        var x = [], y = [];
        s.x.forEach(function (v, k) {
          if ((!logx || v > 0) && (!logy || s.y[k] > 0)) { x.push(v); y.push(s.y[k]); }
        });
        traces.push({
          type: "scatter", mode: d.markers ? "lines+markers" : "lines", x: x, y: y, name: s.name,
          line: { color: PALETTE[i % PALETTE.length], width: 2 }, marker: { size: 5 },
          hovertemplate: d.xlabel + ": %{x:.4g}<br>" + d.ylabel + ": %{y:.4g}<br>" + s.name + "<extra></extra>"
        });
      });
      layout.xaxis = axis(c, { title: { text: d.xlabel }, type: logx ? "log" : "linear", zeroline: false });
      layout.yaxis = axis(c, { title: { text: d.ylabel }, type: logy ? "log" : "linear",
                               exponentformat: "power", zeroline: false });
      if (d.series.length > 1) { layout.showlegend = true; layout.legend = legend(c); }
    }
    return { data: traces, layout: layout };
  }

  var CONFIG = {
    responsive: true, displaylogo: false,
    modeBarButtons: [["zoom2d", "resetScale2d", "toImage"]],
    toImageButtonOptions: { format: "png", scale: 2 }
  };

  function restyle() {
    var c = themeColors();
    plots.forEach(function (p) {
      var fig = build(p.data, c);
      // keep the current slider position and legend toggles
      if (p.el.layout && p.el.layout.sliders) fig.layout.sliders[0].active = p.el.layout.sliders[0].active;
      if (p.el.data) p.el.data.forEach(function (t, i) {
        if (fig.data[i]) { fig.data[i].visible = t.visible; if (t.meta !== undefined) { fig.data[i].y = t.y; fig.data[i].meta = t.meta; } }
      });
      Plotly.react(p.el, fig.data, fig.layout, CONFIG);
    });
  }

  // Show the static image kept in <noscript> instead of an empty plot area.
  function showFallback(fig) {
    var el = fig.querySelector(".alamode-plot-canvas");
    var ns = fig.querySelector("noscript");
    if (el && ns) el.outerHTML = ns.textContent;
  }

  function init() {
    var figs = document.querySelectorAll("figure.alamode-plot");
    if (typeof Plotly === "undefined") { figs.forEach(showFallback); return; }
    var c = themeColors();
    figs.forEach(function (fig) {
      var el = fig.querySelector(".alamode-plot-canvas");
      try {
        var data = JSON.parse(fig.querySelector(".alamode-plot-data").textContent);
        var f = build(data, c);
        Plotly.newPlot(el, f.data, f.layout, CONFIG);
        plots.push({ el: el, data: data });
      } catch (err) {
        console.error("alamode_plots:", err);
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
