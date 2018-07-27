#!/bin/sh
# */AIPND/intropylab-classifying-images/run_models_batch.sh
#                                                                             
# PROGRAMMER: Partap S. Aujla.
# DATE CREATED: 07/25/2018                                  
# REVISED DATE: 02/27/2018  - reduce scope of program
# PURPOSE: Runs all three models to test which provides 'best' solution.
#          Please note output from each run has been piped into a text file.
#
# Usage: sh run_models_batch.sh    -- will run program from commandline
#  
python check_images.py -dir pet_images/ -arch resnet -dogfile dognames.txt > resnet.txt
python check_images.py -dir pet_images/ -arch alexnet -dogfile dognames.txt > alexnet.txt
python check_images.py -dir pet_images/ -arch vgg -dogfile dognames.txt > vgg.txt
