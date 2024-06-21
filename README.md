# Powerlifting Scaling

## Description

The aim of this project is to use the open powerlifting database in order to design an inclusive scoring system for powerlifting. 

The code fits a logistic function to estimate the relationship between the expected total lifted by an athlete of a given bodyweight. The score of an athlete is then given by the total lifted by the said athlete divided by the predicted total.

The distribution of bodyweights in the open powerlifting database is not uniform thus this database cannot be used to provide a reliable scoring method for athletes that have an extreme bodyweight. This problem is alleviated by performing a resampling based on a Kernel Density Estimate (KDE).

## Source

This project uses data that can be downloaded at
https://openpowerlifting.gitlab.io/opl-csv/bulk-csv.html

## Roadmap

## Changelog
