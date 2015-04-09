#
# Use this shell to interpret shell commands, & pass its value to sub-make
#
export SHELL = /bin/sh
#
# This will cause LaTeX to print UTC timestamps!
#
export TZ = UTC
#
# This is like doing 'make -w' on the command line.  This tells make to
# print the directory it is in.
#
MAKEFLAGS = w
#
# This is a list of subdirectories that make should descend into.  Right now
# this only applies to the 'make clean' command.
#
SUBDIRS =
#
# Define compilation programs
#
TEX = pdflatex
BIB = bibtex
#
# Base name of the document.
#
DOCUMENT = coadd
#
# This line helps prevent make from getting confused in the case where you
# have a file named 'clean'.
#
.PHONY: clean
#
# Generic instruction for compiling tex files.
#
%.pdf: %.tex
	$(TEX) -halt-on-error $<
	$(BIB) $*
	$(TEX) -halt-on-error $<
	$(TEX) -halt-on-error $<
#
# This will compile the proposal
#
all: $(DOCUMENT).pdf
#
# GNU make pre-defines $(RM).  The - in front of $(RM) causes make to
# ignore any errors produced by $(RM).
#
clean:
	- $(RM) *.aux *.toc *.out *.log *.bbl *.blg *~ $(DOCUMENT).pdf
	@ for f in $(SUBDIRS); do $(MAKE) -C $$f clean ; done
