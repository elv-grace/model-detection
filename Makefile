IMAGE_NAME := general_detection

DUMMY := $(shell git submodule update --init 1>&2)
include buildscripts/Makefile.tagger-model
