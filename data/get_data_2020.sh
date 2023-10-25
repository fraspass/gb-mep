## Get 2020 data
mkdir santander_summaries
cat file_names_2020.txt | while read line 
do
   curl https://cycling.data.tfl.gov.uk/usage-stats/$line --output santander_summaries/$line
done