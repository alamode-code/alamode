// Multi-version documentation switcher for ALAMODE.
//
// This script is theme- and domain-agnostic. It derives the published layout
// purely from its own URL, so it works under any base path (e.g.
// https://alamode-team.github.io/alamode/<version>/<lang>/...) and behind a
// custom domain without any hard-coded URLs.
//
// The deploy workflow publishes one directory per version, each containing one
// directory per language, plus a versions.json manifest at the site root:
//
//   <site root>/versions.json            ->  ["master", "develop", "2.0dev"]
//   <site root>/<version>/               ->  redirect to the default language
//   <site root>/<version>/<lang>/...     ->  the built docs for each language
//
// This file lives at <site root>/<version>/<lang>/_static/version-switcher.js,
// which is how we recover the site root and the current version below.
//
// Switching version jumps to the chosen version's landing page (which then
// redirects to its default language), because the current language is not
// guaranteed to exist in every version (older versions may be English-only).
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
    // langRoot    = <site root>/<version>/<lang>
    // versionRoot = <site root>/<version>
    // siteRoot    = <site root>
    var langRoot = thisScript.src.slice(0, thisScript.src.indexOf(MARKER));
    var versionRoot = langRoot.slice(0, langRoot.lastIndexOf("/"));
    var siteRoot = versionRoot.slice(0, versionRoot.lastIndexOf("/"));
    if (!versionRoot || !siteRoot) {
      return; // Layout shallower than expected; bail rather than guess.
    }
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
      // Jump to the chosen version's landing page; same-page paths and the
      // current language are not guaranteed to exist across versions, so the
      // version root (which redirects to that version's default language) is
      // the safest target.
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
