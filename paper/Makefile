# ── LaTeX build targets ───────────────────────────────────────
TEXBIN   ?= /Library/TeX/texbin
PDFLATEX  = $(TEXBIN)/pdflatex
BIBTEX    = $(TEXBIN)/bibtex
TLMGR     = $(TEXBIN)/tlmgr

PAPER     = paper
DEPS      = $(PAPER).tex references.bib

.PHONY: all clean distclean install-deps

all: $(PAPER).pdf

# Full build: latex → bibtex → latex × 2
$(PAPER).pdf: $(DEPS)
	$(PDFLATEX) -interaction=nonstopmode $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(PDFLATEX) -interaction=nonstopmode $(PAPER).tex
	$(PDFLATEX) -interaction=nonstopmode $(PAPER).tex

# Quick single-pass (skips bibliography re-run)
quick: $(PAPER).tex
	$(PDFLATEX) -interaction=nonstopmode $(PAPER).tex

# Install required TeX packages via tlmgr
install-deps:
	sudo $(TLMGR) update --self
	sudo $(TLMGR) install \
	    ieeetran \
	    booktabs \
	    siunitx \
	    caption \
	    subcaption \
	    hyperref \
	    multirow \
	    xcolor \
	    tabularx

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).log $(PAPER).out $(PAPER).toc \
	      $(PAPER).lof $(PAPER).lot

distclean: clean
	rm -f $(PAPER).pdf
