#
#
#               DON'T FORGET TO ACTIVATE ENVIRONMENT
#
# Moving to source code directory
cd ../poseidon

# Preprocessing data
python preprocessing.py  --year_start 2010 --year_end 2014 --foldername training   --dawgz 1 --useWandb 1 --useCustomRegion 1 --saveMask 1
python preprocessing.py  --year_start 2015 --year_end 2017 --foldername validation --dawgz 1 --useWandb 1 --useCustomRegion 1 --saveMask 0
python preprocessing.py  --year_start 2018 --year_end 2019 --foldername test       --dawgz 1 --useWandb 1 --useCustomRegion 1 --saveMask 0