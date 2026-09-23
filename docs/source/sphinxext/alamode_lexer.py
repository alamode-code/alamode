"""Pygments lexers for the ALAMODE documentation.

``alamode``       ALAMODE input files: ``&field`` ... ``/`` blocks and ``TAG = value`` lines.
``alamode-auto``  the default for plain ``::`` blocks: picks a lexer from the content
                  (shell session or script, ALAMODE input, Python, or plain text), so that commands
                  and input files are not lexed as Python.
"""

import re

from pygments.lexer import Lexer, RegexLexer, bygroups
from pygments.lexers.python import PythonLexer
from pygments.lexers.shell import BashLexer, BashSessionLexer
from pygments.lexers.special import TextLexer
from pygments.token import (
    Comment,
    Error,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    Text,
)


class AlamodeInputLexer(RegexLexer):
    name = "ALAMODE input"
    aliases = ["alamode"]
    flags = re.MULTILINE

    tokens = {
        "root": [
            (r"#.*$", Comment.Single),
            (r"^(\s*)(&\w+)", bygroups(Text, Keyword.Namespace)),
            (r"^(\s*)(/)(\s*)$", bygroups(Text, Keyword.Namespace, Text)),
            (r"\b([A-Z][A-Z0-9_]*)(\s*)(=)", bygroups(Name.Attribute, Text, Operator)),
            (r"[-+]?(\d+\.\d*|\.\d+|\d+)([eEdD][-+]?\d+)?(?![\w.])", Number),
            (r";", Punctuation),
            (r"[^\s#;=]+", Text),
            (r"[=#]", Text),
            (r"\s+", Text),
        ]
    }


_SHELL = re.compile(r"\A\s*[$%] ")
_FIELD = re.compile(r"^\s*&[A-Za-z]\w*", re.MULTILINE)
_TAG_LINE = re.compile(r"^\s*[A-Z][A-Z0-9_]*\s*=")
_PYTHON = re.compile(
    r"^\s*(import |from \S+ import |def |class |print\(|with )", re.MULTILINE
)


def _pick(text):
    lines = [
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if _SHELL.match(text):
        return BashSessionLexer
    if re.match(r"\A\s*#!.*sh\b", text):
        return BashLexer
    if _FIELD.search(text) or (lines and all(_TAG_LINE.match(ln) for ln in lines)):
        return AlamodeInputLexer
    if _PYTHON.search(text):
        return PythonLexer
    return TextLexer


class AlamodeAutoLexer(Lexer):
    name = "ALAMODE auto"
    aliases = ["alamode-auto"]

    def get_tokens_unprocessed(self, text):
        yield from _pick(text)(**self.options).get_tokens_unprocessed(text)


def setup(app):
    app.add_lexer("alamode", AlamodeInputLexer)
    app.add_lexer("alamode-auto", AlamodeAutoLexer)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


if __name__ == "__main__":
    # Self-check of the content detection.
    assert _pick("$ alm alm.in > alm.log\n") is BashSessionLexer
    assert _pick("% cmake ..\n") is BashSessionLexer
    assert _pick("#!/bin/bash\nmpirun vasp\n") is BashLexer
    assert _pick("&general\n  PREFIX = si\n/\n") is AlamodeInputLexer
    assert _pick("L1_ALPHA = 2.5e-06\nCV = 0 # off\n") is AlamodeInputLexer
    assert _pick("import h5py\nf = h5py.File('a.h5')\n") is PythonLexer
    assert _pick("Bohr = 0.529\nprint(Bohr)\n") is PythonLexer
    assert _pick("  0.0  0.1  0.2\n  1.0  1.1  1.2\n") is TextLexer
    toks = list(
        AlamodeInputLexer().get_tokens("&general\n  NAT = 64; KD = Si # c\n/\n")
    )
    assert (Keyword.Namespace, "&general") in toks and (Name.Attribute, "NAT") in toks
    assert (Comment.Single, "# c") in toks
    assert all(
        t is not Error for t, _ in AlamodeInputLexer().get_tokens("a = b == c\n")
    )
    print("alamode_lexer self-check: ok")
