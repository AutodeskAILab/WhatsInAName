#!/bin/bash

mkdir -p data/abc

echo Downloading data...
curl "https://whats-in-a-name-dataset.s3.us-west-2.amazonaws.com/whats_in_a_name_dataset.zip" -o data/whats_in_a_name_dataset.zip || exit
cd data

echo Unzipping...
unzip whats_in_a_name_dataset.zip || exit
mv whats_in_a_name_dataset/* abc/

echo Cleaning up...
rm -rf whats_in_a_name_dataset
rm -f whats_in_a_name_dataset.zip

echo Completed. Please see license in ./data/abc/LICENSE.txt
