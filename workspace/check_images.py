#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#                                                                             
# Done: 0. Fill in your information in the programming header below
# PROGRAMMER: Partap S. Aujla
# DATE CREATED: Friday, July 20, 2018
# REVISED DATE:             <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print 
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir

# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# Imports print functions that check the lab
#from print_functions_for_lab_checks import *
from print_functions_for_lab_checks import check_calculating_results as check

# Main program function defined below
def main():
    # Done: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()
    
    # Done: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()
    
    # Done: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)

    # Done: 4. Define classify_images() function to create the classifier 
    # labels with the classifier function uisng in_arg.arch, comparing the 
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)
    
    # Done: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)

    # Done: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)

    # Done: 7. Define print_results() function to print summary results, 
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch, True, True)

    # Done: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # Done: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
   
    # Prints overall runtime in format hh:mm:ss
    print("\n** Total Elapsed Runtime:", str(int(tot_time / 3600)) 
          + ":" + str(int((tot_time % 3600) / 60)) 
          + ":" + str(int((tot_time % 3600) % 60)))
    
# TODO: 2.-to-7. Define all the function below. Notice that the input 
# paramaters and return values have been left in the function's docstrings. 
# This is to provide guidance for acheiving a solution similar to the 
# instructor provided solution. Feel free to ignore this guidance as long as 
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", default='pet_images/', help="Path to the pet image files(default- 'pet_images/')")
    parser.add_argument("-arch", choices=['vgg','alexnet','resnet'], default='vgg', help="CNN model architecture to use for image classificaiotn(default- vgg)")
    parser.add_argument("-dogfile", default='dognames.txt', help="Text file that contains all labels associated to dogs(default- 'dognames.txt')")
    return parser.parse_args()

def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image 
    files. Reads in pet filenames and extracts the pet image labels from the 
    filenames and returns these label as petlabel_dic. This is used to check 
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)  
    """    
    petlabel_dic = dict()
    key_list = listdir(path=image_dir)
    for i in range(0, len(key_list), 1):
        #Mac users
        if key_list[i][0] != ".":
            name = key_list[i].split("_")
            value = ""
            for element in name:
                if element.isalpha():
                    value += element + " "
            if key_list[i] not in petlabel_dic:
                petlabel_dic[key_list[i]] = value.strip().lower()    
    return petlabel_dic

def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifer labels and 0 = no match between labels
    """
    result_dic = dict()
    for key in iter(petlabel_dic):
        value_list = list()
        value_list.append(petlabel_dic.get(key))
        value_list.append(classifier(images_dir + key, model).strip().lower())
        found_idx = value_list[1].find(value_list[0])
        if found_idx == 0 and len(value_list[0]) == len(value_list[1]):
            value_list.append(int(1))
        elif (found_idx == 0 or value_list[1][found_idx - 1] == " ") and (found_idx + len(value_list[0]) == len(value_list[1])) or (value_list[1][found_idx + len(value_list[0]):found_idx + len(value_list[0]) + 1] in (" ", ",")):
                value_list.append(int(1))
        else:
            value_list.append(int(0))
        if key not in result_dic:
            result_dic[key] = value_list
    return result_dic

def adjust_results4_isadog(result_dic, dogfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    dognames_dic = dict()
    with open(dogfile, 'r') as file:
        for line in file:
            if line.rstrip() not in dognames_dic:
                dognames_dic[line.rstrip()] = 1
            else:
                print("Warning: '{}' is duplicate entry in file.".format(line.rstrip()))
    for key in iter(result_dic):
        # Possible refactoring
        pet_label_name = result_dic.get(key)[0]
        classifier_label_name = result_dic.get(key)[1]
        if dognames_dic.get(pet_label_name) == 1:
            result_dic[key].append(1)
        else:
            result_dic[key].append(0)
        if dognames_dic.get(classifier_label_name) == 1:
            result_dic[key].append(1)
        else:
            result_dic[key].append(0)
            
def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """
    result_stat_dic = dict()
    # Possible refactoring
    n_of_images, n_correct_dogs, n_dog_images, n_correct_nondogs, n_nondog_images, n_correct_breed, n_label_matches = len(results_dic), 0, 0, 0, 0, 0, 0
    for key in iter(results_dic):        
        if results_dic[key][3] == 1:
            n_dog_images += 1
            if results_dic[key][4] == 1:
                n_correct_dogs += 1
            if results_dic[key][2] == 1:
                n_correct_breed += 1
                n_label_matches +=1
        else:
            n_nondog_images += 1
            if results_dic[key][4] == 0:
                n_correct_nondogs += 1
            if results_dic[key][2] == 1:
                n_label_matches += 1    
    # Possible refactoring
    result_stat_dic["n_images"] = n_of_images
    result_stat_dic["n_dogs_img"] = n_dog_images
    result_stat_dic["n_notdogs_img"] = n_nondog_images
    result_stat_dic["n_correct_dogs"] = n_correct_dogs
    result_stat_dic["n_correct_nondogs"] = n_correct_nondogs
    result_stat_dic["n_correct_breed"] = n_correct_breed
    result_stat_dic["n_label_matces"] = n_label_matches
    if n_dog_images == 0:
        result_stat_dic["pct_correct_dogs"] = 0
        result_stat_dic["pct_correct_breed"] = 0
    else:
        result_stat_dic["pct_correct_dogs"] = (n_correct_dogs/n_dog_images)*100
        result_stat_dic["pct_correct_breed"] = (n_correct_breed/n_dog_images)*100
    if n_nondog_images == 0:
        result_stat_dic["pct_correct_notdogs"] = 0
    else:
        result_stat_dic["pct_correct_notdogs"] = (n_correct_nondogs/n_nondog_images)*100
    if n_of_images == 0:
        result_stat_dic["pct_label_matches"] = 0
        print("Warning: No images")
    else:
        result_stat_dic["pct_label_matches"] = (n_label_matches/n_of_images)*100
    return result_stat_dic

def print_results(results_dic, results_stats, model, print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """
    print("CNN Model Architecture: {}.".format(model))
    print("Number of Images: {}.".format(results_stats.get("n_images")))
    print("Number of Dog Images: {}.".format(results_stats.get("n_dogs_img")))
    print("Number of NOTDog Images: {}.".format(results_stats.get("n_notdogs_img")))
    print("% Correct Dogs: {}%.".format(results_stats.get("pct_correct_dogs")))
    print("% Correct Breed: {}%.".format(results_stats.get("pct_correct_breed")))
    print("% Correct NOTDogs: {}%.".format(results_stats.get("pct_correct_notdogs")))
    print("% Match (incl. dog & not dog): {}%.".format(results_stats.get("pct_label_matches")))
    
    # Printing misclassification 
    # use [] instead of get?
    if print_incorrect_dogs and (results_stats.get("n_correct_dogs") + results_stats.get("n_correct_nondogs") != results_stats.get("n_images")):
        for key in iter(results_dic):
            if sum(results_dic.get(key)[3:]) == 1:
                print("Dog misclassificaiton: pet image {} and classifier {} label.".format(results_dic.get(key)[0], results_dic.get(key)[1]))
    if print_incorrect_breed and (results_stats.get("n_correct_dogs") != results_stats.get("n_correct_breed")):
        for key in iter(results_dic):
            if sum(results_dic.get(key)[3:]) == 2 and results_dic.get(key)[2] == 0:
                print("Breed misclassificaiton: pet image {} and classifier {} label.".format(results_dic.get(key)[0], results_dic.get(key)[1]))
    pass                
                
# Call to main function to run the program
if __name__ == "__main__":
    main()