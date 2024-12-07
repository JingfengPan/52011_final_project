# 52011_final_project
Proposal:
I would like to analyze the inside of csv datasets about what influences the compression efficiency (compression rate and time). I will do the followings:
1. Analyze the dataset with which data types it contains: numeric data, categorical data, string data, mixed data.
2. Compare the compression rate and time of 3 most popular compression algorithms: Gzip, Lz4. Zstandard for the same datasets.
3. Compare the compression rate and time of datasets with different data types in the same size.
4. Try to use some methods to help increase the compression rate based on the logic of compression algorithms , e.g. clustering or classification.
5. Try to use some methods to help accelerate the compression time.
I will do the things above and make a final report, which can help people improve compression efficiency.

Execution Plan: <br>
Week 4: prepare all the needed datasets, from some public resources like kaggle. <br>
Week 5: Implement 3 compression methods in Python and see their performance with different datasets. <br>
Week 6: Find some methods to help increase the compression rate, e.g. clustering or classification. <br>
Week 7: Find some methods to help accelerate the compression time, e.g. pypy. <br>
Week 8: Conclude the findings and prepare a final report.

</b>To run the experiment, just run the main function in the "main.py" file.</b>

Process of my experiment
1. Clean and filter data, select useful columns (numeric and category here).
2. Using K-prototypes/K-means to clustering, then compress each cluster and combine the total compressed size of each cluster, compared to compress the whole dataset directly using Gzip/LZ4/Zstd.
3. Explore the raltionship of cluster numbers and compression ratio.
4. Using a small part of dataset (maybe 10%) to cluster, then using the cluster labels to train a classifier model.
5. Using the trained model to predict the whole dataset (give all labels), then compress each cluster and combine the total compressed size of each cluster, compared to compress the whole dataset directly and compress by the labels of clustering all data.
6. Explore the raltionship of different classifiers and compression ratio.
7. Explore the raltionship of train size and compression ratio.

The link of downloading the datasets: shttps://drive.google.com/drive/folders/1UoQBW4KKNyCNKGbEKDz4TqpgysoUuI_A?usp=drive_link
