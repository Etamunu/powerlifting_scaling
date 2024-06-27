# Powerlifting Scaling

## Description

The aim of this project is to use the Open Powerlifting database in order to design an inclusive scoring system for powerlifting. 

The present code fits a logistic function to estimate the relationship between the expected total lifted by an athlete of a given bodyweight. The score of an athlete is then given by the total lifted by the said athlete divided by the predicted total. The logistic function has been selected over the model used for the IPF GL score as it can model the inflection point that appears when observing underweight athletes.

The distribution of bodyweights in the Open Powerlifting database is not uniform thus this database cannot be used to provide a reliable scoring method for athletes that have an extreme bodyweight. This problem is alleviated by performing a resampling based on a Kernel Density Estimate (KDE).

## References

Website of the I(nternational) P(owerlifting) F(ederation) : https://www.powerlifting.sport/

This project uses data that can be downloaded at : https://openpowerlifting.gitlab.io/opl-csv/bulk-csv.html

## Roadmap

  - improve the setup of the Gaussian noise
  - analysis of individual lifts
  - consider more general families of functions
  - add a functionality to download the dataset directly from the OpenPowerlifting Data Service instead of accessing it locally

## Changelog
