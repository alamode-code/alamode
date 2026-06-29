// Multi-language documentation switcher for ALAMODE.
//
// Companion to version-switcher.js. It derives the published layout from its
// own URL and reads the per-version languages.json manifest emitted by the
// deploy workflow:
//
//   <site root>/<version>/languages.json ->  ["en", "ja"]
//   <site root>/<version>/<lang>/...      ->  the built docs for each language
//
// This file lives at <site root>/<version>/<lang>/_static/language-switcher.js.
// Switching language stays within the current version (every language a
// version declares is built for that version, so the target always exists).
//
// Only versions that ship translations carry a languages.json, so on English-
// only versions (or local builds) the manifest is absent and no switcher is
// shown -- the page is unaffected.
(function () {
  "use strict";

  // Display names for language codes. Codes without an entry fall back to the
  // raw code, so a new language renders (just untranslated) before it is added.
  var LANG_LABELS = {
    en: "English",
    ja: "日本語",       // 日本語
    fr: "Français",
    de: "Deutsch",
    es: "Español",
    zh: "中文",             // 中文
    ko: "한국어",       // 한국어
  };

  var thisScript = document.currentScript;
  var MARKER = "/_static/language-switcher.js";

  function labelFor(code) {
    return Object.prototype.hasOwnProperty.call(LANG_LABELS, code)
      ? LANG_LABELS[code]
      : code;
  }

  function start() {
    if (!thisScript || thisScript.src.indexOf(MARKER) === -1) {
      return; // Unexpected layout; fail quietly rather than break the page.
    }
    // langRoot    = <site root>/<version>/<lang>
    // versionRoot = <site root>/<version>
    var langRoot = thisScript.src.slice(0, thisScript.src.indexOf(MARKER));
    var versionRoot = langRoot.slice(0, langRoot.lastIndexOf("/"));
    if (!versionRoot) {
      return; // Layout shallower than expected; bail rather than guess.
    }
    var current = decodeURIComponent(langRoot.slice(langRoot.lastIndexOf("/") + 1));

    fetch(versionRoot + "/languages.json", { cache: "no-cache" })
      .then(function (resp) { return resp.json(); })
      .then(function (languages) { render(languages, versionRoot, current); })
      .catch(function () { /* manifest missing: English-only or local build */ });
  }

  function render(languages, versionRoot, current) {
    // Nothing to switch between unless there are at least two languages.
    if (!Array.isArray(languages) || languages.length < 2) {
      return;
    }

    var box = document.createElement("div");
    box.className = "language-switcher";

    var label = document.createElement("label");
    label.className = "language-switcher__label";
    label.textContent = "Language";
    label.setAttribute("for", "language-switcher-select");

    var select = document.createElement("select");
    select.id = "language-switcher-select";
    select.className = "language-switcher__select";

    languages.forEach(function (entry) {
      var code = typeof entry === "string" ? entry : entry.code;
      var option = document.createElement("option");
      option.value = code;
      option.textContent = labelFor(code);
      if (code === current) {
        option.selected = true;
      }
      select.appendChild(option);
    });

    select.addEventListener("change", function () {
      // Stay in the current version; jump to the chosen language's landing page.
      // Same-page paths are not guaranteed to exist across languages, so the
      // language root is the safest target.
      window.location.href =
        versionRoot + "/" + encodeURIComponent(select.value) + "/";
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
