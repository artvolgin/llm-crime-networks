
This repository contains Python and R scripts and the data used in the paper "Using Large Language Models for Extracting Criminal Network Data from Text Yields Comparable Results to Manual Coding"

## Files

/data

Contains the original PDF files, edgelists extracted by human coders, and the final edgelists obtained using ChatGPT.

exraction.py

- Loads and preprocesses the PDF files.
- Extracts entities and their relationships using ChatGPT.
- Postprocesses the extracted entities and relationships.

comparision.py

- Loads the networks extracted by ChatGPT.
- Compares these networks with those obtained by human coders.

network-plots.r

- Plots the networks extracted by ChatGPT and human coders for one case.

requirements.txt

This file lists the Python libraries and their versions required to run the scripts in this repository.

## Usage

Setup:

1. Configure the data folder.

2. Install the required Python libraries by running:

Execute the scripts in the following order:

1. query.py

2. comparision.py

3. network_plots.r
