# Disimpy JOSS paper

The manuscript (`paper.md`), bibliography (`paper.bib`), and two figures are
unchanged copies from the repository's [`v0.1.1` tag](https://github.com/kerkelae/disimpy/tree/v0.1.1),
the version named in the [JOSS review](https://github.com/openjournals/joss-reviews/issues/2527).

`paper.pdf` is the [published JOSS PDF](https://joss.theoj.org/papers/10.21105/joss.02527.pdf),
downloaded on 11 October 2026. It was not rebuilt locally.

To build a new PDF from the source files, run from this directory:

```bash
docker run --rm -v "$PWD":/data -u "$(id -u):$(id -g)" \
  -e JOURNAL=joss openjournals/inara -o pdf paper.md
```

This command writes `paper.pdf`, replacing the published copy if it is still
present. See the [JOSS paper instructions](https://joss.readthedocs.io/en/latest/paper.html#checking-that-your-paper-compiles).
