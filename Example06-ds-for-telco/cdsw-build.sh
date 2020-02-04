#!/bin/bash

if [ -d "models/telco/" ] 
then
  rm -r -f models/telco/*
else
  mkdir -p models/telco
fi

if [ -f "telco_rf.tar" ]
then 
  tar -xf telco_rf.tar 
fi

if [ -f "sklearn_rf.pkl" ]
then 
  mv sklearn_rf.pkl models
fi
