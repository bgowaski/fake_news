#!/bin/bash
export KAGGLE_USERNAME="$1"
export KAGGLE_KEY="$2"
kaggle datasets download dataset snapcrack/all-the-news --unzip
