#!/bin/bash
kaggle datasets download -d cavidankarimli1/musicgenreclassification
unzip musicgenreclassification.zip -d 100_data
rm musicgenreclassification.zip