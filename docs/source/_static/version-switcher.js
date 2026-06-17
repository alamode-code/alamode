// Multi-version documentation switcher for ALAMODE.
//
// This script is theme- and domain-agnostic. It derives the published layout
// purely from its own URL, so it works under any base path (e.g.
// https://alamode-team.github.io/alamode/<version>/...) and behind a custom
// domain without any hard-coded URLs.
//
// The deploy workflow publishes one directory per version plus a versions.json
// manifest at the site root:
//
//   <site root>/versions.json      ->  ["master", "develop", "2.0dev"]
//   <site root>/<version>/...      ->  the built docs for each version
//
// This file lives at <site root>/<version>/_static/version-switcher.js, which
// is how we recover both the current version and the site root below.
(function () {
  "use strict";

  // document.currentScript is only valid during initial synchronous execution,
  // so capture it now and use it later from the DOMContentLoaded callback.
  var thisScript = document.currentScript;
  var MARKER = "/_static/version-switcher.js";

  function start() {
    if (!thisScript || thisScript.src.indexOf(MARKER) === -1) {
      return; // Unexpected layout; fail quietly rather than break the page.
    }
    var versionRoot = thisScript.src.slice(0, thisScript.src.indexOf(MARKER));
    var siteRoot = versionRoot.slice(0, versionRoot.lastIndexOf("/"));
    var current = decodeURIComponent(versionRoot.slice(versionRoot.lastIndexOf("/") + 1));

    fetch(siteRoot + "/versions.json", { cache: "no-cache" })
      .then(function (resp) { return resp.json(); })
      .then(function (versions) { render(versions, siteRoot, current); })
      .catch(function () { /* manifest missing (e.g. local build): no switcher */ });
  }

  function render(versions, siteRoot, current) {
    if (!Array.isArray(versions) || versions.length === 0) {
      return;
    }

    var box = document.createElement("div");
    box.className = "version-switcher";

    var label = document.createElement("label");
    label.className = "version-switcher__label";
    label.textContent = "Version";
    label.setAttribute("for", "version-switcher-select");

    var select = document.createElement("select");
    select.id = "version-switcher-select";
    select.className = "version-switcher__select";

    versions.forEach(function (entry) {
      var name = typeof entry === "string" ? entry : entry.name;
      var option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      if (name === current) {
        option.selected = true;
      }
      select.appendChild(option);
    });

    select.addEventListener("change", function () {
      // Jump to the chosen version's landing page; same-page paths are not
      // guaranteed to exist across versions, so the version root is safest.
      window.location.href = siteRoot + "/" + encodeURIComponent(select.value) + "/";
    });

    box.appendChild(label);
    box.appendChild(select);
    document.body.appendChild(box);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
