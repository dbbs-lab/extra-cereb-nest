#!/bin/bash
echo "Launching Tests for Extra-cereb Module"
rm *.csv
rm *.gdf

#### CHECK MODELS #####

python Check_Models.py  &>TestLog.txt
if [ $? = 0 ]; then
  echo "Check_Models.py SUCCESS"
else
  echo "Check_Models.py FAIL"
fi

python Remove_Empty.py &>>TestLog.txt

#### CLEAN-UP THE FOLDER #####
rm *.csv *.gdf
