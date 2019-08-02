# Project-Angel
Contains code for collecting Instagram posts, classifying them into body shaming or not body shaming, and final analysis of the posts classified as body shaming.

-final_15K_dataset.json

-final_dataset_testing.py: running the models on the 15K dataset(third column in results table)

-2k_dataset_testing.py: running the models on the 2K dataset(first column in results table)

-sp+Ip+sn+In.csv: 2K labeled dataset

-scraper_posts.json: around 43K posts scraped from Instagram together with their elmo embeddings

-crawler.py: instagram crawler; collects and saves posts and images

-elmo_encoder.py: used to make elmo embeddings of the instagram posts

-posts_analysis.py: POS tagging, average length of posts for 15K dataset

-making_big_dataset.py: creating 15K dataset

#20% of 2K dataset used for testing

 -testing_20%.json: posts
 
 -testing_20%_labels.json: labels of the posts in testing_20%.json
 
 -testx_20%_emb.json: elmo emb of the above posts

#80% of 2K dataset used for training

 -training_80%.json: posts
 
 -training_80%_labels.json: labels
 
 -trainx_80%_emb.json: elmo emb


-folders: -->New dataset per model (contains all models trained on 2K used to predict and label by 0.9 confidence interval on 43K,
				labeled posts are added to 2K and the model is trained on the new augmented dataset and then tested on 20% of 2K to find accuracy,
				second column in results table)
	  -->Web demo (contains files to run the demo)



