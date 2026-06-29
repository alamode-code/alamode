// Per-language PDF download link for ALAMODE.
//
// Companion to version-switcher.js / language-switcher.js. It derives the
// published layout from its own URL and adds a "Download PDF" link pointing at
// the PDF published next to the current language's HTML:
//
//   <site root>/<version>/<lang>/ALAMODE.pdf
//
// This file lives at <site root>/<version>/<lang>/_static/pdf-download.js. The
// link is shown only if that PDF actually exists (checked with a HEAD request),
// so legacy/English-only versions and local builds without a PDF are
// unaffected -- the page simply shows no link.
(function () {
  "use strict";

  var thisScript = document.currentScript;
  var MARKER = "/_static/pdf-download.js";

  function start() {
    if (!thisScript || thisScript.src.indexOf(MARKER) === -1) {
      return; // Unexpected layout; fail quietly rather than break the page.
    }
    // langRoot = <site root>/<version>/<lang>
    var langRoot = thisScript.src.slice(0, thisScript.src.indexOf(MARKER));
    if (!langRoot) {
      return;
    }
    var pdfUrl = langRoot + "/ALAMODE.pdf";

    // Only render the link when the PDF is actually published here.
    fetch(pdfUrl, { method: "HEAD", cache: "no-cache" })
      .then(function (resp) { if (resp.ok) render(pdfUrl); })
      .catch(function () { /* no PDF (e.g. local build): no link */ });
  }

  function render(pdfUrl) {
    var box = document.createElement("div");
    box.className = "pdf-download";

    var link = document.createElement("a");
    link.className = "pdf-download__link";
    link.href = pdfUrl;
    link.setAttribute("download", "");
    link.rel = "nofollow";
    link.textContent = "↓ Download PDF"; // ↓ Download PDF

    box.appendChild(link);
    document.body.appendChild(box);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
