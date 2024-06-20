# Build a recommendation system with TensorFlow and Keras
It is a step-by-step tutorial on developing a practical recommendation system (`retrieval` and `ranking` tasks) using [TensorFlow Recommenders](https://www.tensorflow.org/recommenders) and [Keras](https://keras.io/) and deploy it using [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving).

Here, you can find an introduction to the information retrieval and the recommendation systems, then you can explore [the Jupyter notebook](https://github.com/xei/recommender_system_tutorial/blob/main/recommender_system_tutorial.ipynb) and run it in [Google Colab](https://colab.research.google.com/github/xei/recommender_system_tutorial/blob/main/recommender_system_tutorial.ipynb) in order to study the code.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xei/recommender_system_tutorial/blob/main/recommender_system_tutorial.ipynb)
<a target="_blank" href="https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2Fxei%2Frecommender-system-tutorial%2Fblob%2Fmain%2Frecommender_system_tutorial.ipynb">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open in Studio" />
</a>

In the notebook, we load [MovieLens dataset](https://grouplens.org/datasets/movielens/) using [TesorFlow Datasets](https://www.tensorflow.org/datasets), preprocess its features using [Keras preprocessing layers](https://keras.io/guides/preprocessing_layers/), build the `retrieval` and `ranking` tasks using [TensorFlow Recommenders](https://www.tensorflow.org/recommenders) and index and search for similar items using [Spotify Annoy](https://github.com/spotify/annoy).

This tutorial is recommended to both academic and industry enthusiasts.

## Introduction

### What is a recommendation system?!
Online services usually provides thousands, millions or even billions of items like products, video clips, movies, musics, news, articles, blog posts, advertisements, etc. For example, the [Google Play Store](https://play.google.com/store) provides millions of apps and [YouTube](https://www.youtube.com/) provides billions of videos. [[1]](https://developers.google.com/machine-learning/recommendation/overview)

However, users prefer to see a handful shortlist of likely items instead of struggling with the full corpora. They usually can search or filter the list to find the best handful items, but sometimes they even don't know what they really want (e.g. a birthday gift). In a physical store an expert seller would help in this case by useful recommendations. So, why not in an online store?!

A recommedation system can retrieve, filter and recommend best personalized results for the user - results which the user is likely to buy. So it is one of the major requirements of modern businesses in order to increase their `conversion rate`. On September 21, 2009, Netflix gave a grand prize of $1,000,000 to a team which bested Netflix's own algorithm for predicting ratings by 10.06%. [[2]](https://web.archive.org/web/20090924184639/http://www.netflixprize.com/community/viewtopic.php?id=1537)

<p align="center">
  <img width="634" height="416" src="https://raw.githubusercontent.com/xei/recommender-system-tutorial/main/assets/netflix_prize.jpeg">
</p>

A recommendation system ia a system that gives a `query (context)` which is what we know about the liking list, and filter the corpus (full catalog of items) to a shortlist of `candidates` (items, documents). A query (context) can be a ***user id***, ***user's geographical location*** or ***user's history of previous purchases*** and the resulting candidates can be some new items that we guess are interesting for the user.

The query can also be an ***item id***, ***its image*** or ***its textual description*** and the candidates can be some similar or related items from the corpus.
<p>&nbsp;</p>
<p align="center">
  <img width="556" height="82" src="https://raw.githubusercontent.com/xei/recommender-system-tutorial/main/assets/recsys_io.png">
</p>
<p>&nbsp;</p>

### Recommendation stages (tasks)
In practice, dealing with a large corpus and filter it to a shortlist is an intractable and inefficient task. So practical recommender systems has two (or three) filterng phases:
1. Retrieval (Candidate Generation)
2. Ranking (Scoring)
3. Re-ranking or optimazation or ...

<p align="center">
  <img width="700" height="394" src="https://raw.githubusercontent.com/xei/recommender-system-tutorial/main/assets/retrieval_ranking.png">
</p>

<p align="center">
  <img width="700" height="394" src="https://raw.githubusercontent.com/xei/recommender-system-tutorial/main/assets/deepmind_forgoogle_recsys.png">
</p>

### Content-based Filtering vs Collaborative Filtering
Filtering items is based on similarities. we can filter the list based on similar candidates (`content-based filtering`) or based on the similarity between queries and candidates (`collaborative filtering`). Collaborative filtering algorithms usually perform better than content-based methods.

<p align="center">
  <img width="700" height="330" src="https://raw.githubusercontent.com/xei/recommender-system-tutorial/main/assets/cbf_cf.png">
</p>

### Representation of a query or a candidate
A query or a candidate has lots of different features. For example a query can be constructed by these features:
- user_id
- user_previous_history
- user_job
- etc.

And a candidate can have features like:
- item_description
- item_image
- item_price
- posted_time
- etc.

These obviouse features can be `numerical variables`, `categorical variables`, `bitmaps` or `raw texts`. However, these low-level features are not enough and we should extract some more abstract `latent features` from these obvious features to represent the query or the candidate as a numerical high-dimensional vector - known as `Embedding Vector`.

`Matrix Factorization` (MF) is a classic collaborative filtering method to learn some `latent factors` (latent features) from `user_id`, `item_id` and `rating` features and represent **users** and **items** by latent (embedding) vectors.

<p align="center">
  <img width="826" height="398" src="https://raw.githubusercontent.com/xei/recommender-system-tutorial/main/assets/mf.png">
</p>

Matrix Factorization method only uses `user_id` and `candidate_id` features collaboratively to learn the `latent features`. In fact it doesn't care about other side-features like `candidate_description`, `price`, `user_comment`, etc.

To involve side-features as well as ids while learning latent features (embeddings), we can use deep neural network (DNN) architectures like `softmax` or `two-tower` neural models.

<p align="center">
  <img width="540" height="340" src="https://raw.githubusercontent.com/xei/recommender-system-tutorial/main/assets/tensorflow_two_tower.gif">
</p>

YouTube two-tower neural model uses side-features to represent queries and candidates in an abstract high-dimentional embedding vector.

<p align="center">
  <img width="540" height="340" src="https://raw.githubusercontent.com/xei/recommender-system-tutorial/main/assets/youtube_retrieval.png">
</p>

### Movielens dataset

The `Movielens` dataset is a benchmark dataset in the field of recommender system research containing a set of *ratings* given to *movies* by a set of *users*, collected from the [MovieLens website](http://movielens.org/) - a movie recommendation service.

There are 5 different versions of Movielens available for different purposes: "25m", "latest-small", "100k", "1m" and "20m". In this tutorial we are going to work with "100k" version. For more information about different versions visit the [official website](https://grouplens.org/datasets/movielens/).

**movielens/100k-ratings**

The oldest version of the MovieLens dataset containing 100,000 ratings from 943 users on 1,682 movies. Each user has rated at least 20 movies. Ratings are in whole-star increments. This dataset contains demographic data of users in addition to data on movies and ratings.

**movielens/100k-movies**

This dataset contains data of 1,682 movies rated in the ***movielens/100k-ratings*** dataset.

<p>&nbsp;</p>

## Explore the Jupyter notebook

<table class="tfo-notebook-buttons" align="left">
    <td>
    <a target="_blank" href="https://github.com/xei/recommender_system_tutorial/blob/main/recommender_system_tutorial.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" /> View the code on GitHub</a>
  </td>
   
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/xei/recommender_system_tutorial/blob/main/recommender_system_tutorial.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" /> Run the code in Google Colab</a>
  </td>
</table>

<p>&nbsp;</p>
<p>&nbsp;</p>

<p>&nbsp;</p>
<p>&nbsp;</p>

## Donation
Give a ⭐ if this project helped you!
