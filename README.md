# Wordpress Blog Post Recommender in Spark and Scala

At the Advanced Data Analytics team at Barclays we solved the Kaggle competition https://www.kaggle.com/c/predict-wordpress-likes as proof-of-concept of how to use [Spark](https://github.com/apache/spark), [Scala](https://github.com/scala/scala) and the [Spark Notebook](https://github.com/andypetrella/spark-notebook) to solve a typical machine learning problem end-to-end.

## Problem

Recommending a sequence of blog posts that the users may like on WordPress based on their historical likes and blog/post/author characteristics.

## MVP Solution

### Features

Each data point consisted on the event like/not-like of a particular pair of user and blog post.

We computed the following set of features:
* categoriesLikelihood: probability of liking the post by the user only based on the categories term frquency of likes.
* tagsLikelihood: probability of liking the post by the user only based on the tags term frquency of likes.
* languageLikelihood: probability of liking the post by the user only based on the language frquency of likes.
* authorLikelihood: probability of liking the post by the user only based on the author frequency of likes.
* titleLengthMeanError: absolute difference between the title length and the average title length of liked posts.
* blogLikelihood: probability of liking the post by the user only based on the blog frequency of likes.
* averageLikesPerPost: average number of likes per post of that blog.

The raw data only contained the true class points. We generated the false class points according to the same distribution and by combining posts with random users who did not like that post.

Some features used some summary historical statistics about the user who were not available for all of them. We computed the global prior probabilities averaging all of the available information of other users instead of filling missing features using interpolation techniques.

Example of execution plan in Spark for multiple table joins generated with a few lines of Scala code:
![Alt text](/../master/execution-plan-feature-engineering.png?raw=true "Example of execution plan in Spark for multiple table joins")

### Modelling

We implemented two models:
* Simple Recommender simply based on the tags likelihood.
* Logistic Regression model to predict wheter the user is going to like or not a given post and then ranking the true-classified posts based on the logistic raw score.
 
### Evaluation
Final end-to-end evaluation was made using [Mean Average Precision @ 100](https://www.kaggle.com/wiki/MeanAveragePrecision).

For the logistic regression we also run a bunch of standard classifier evaluations (e.g. ROC curve) to evaluate how good it is on predicting the true/false like class and finding the best threshold for limiting the number of recommended blog posts.

## Methodology

For the ETL and Feature Engineering we used a mixed approach where data collections were trasnformed from DataFrame into RDD of case classes and viceversa based on the specific use case so that we could exploit both the functionalities of the two APIs.
We followed an investigate then develop methodology where we iterate through the following steps:

1. Run some interactive exploratory analysis in the SparkNotebook
2. Switch into a proper IDE (IntelliJ in our case) to develop the code and make the API available in the form of functions
3. Build the jar as a library and add it to the SparkContext classpath
4. Call the library functions within the notebook to trigger the execution
5. Analyse and present results
 
This workflow allowed us to first understand the data interactively and then develop the main logic into a more productive and robust environment like IntelliJ so that the Notebook stays clean and the library can be reused in different contexts.

The end-to-end workflow was implemented and performed according to the following external references:
* [Manifesto for Agile Data Science] (http://www.datasciencemanifesto.org)
* [The complete 18 steps to start a new Agile Data Science project](https://datasciencevademecum.wordpress.com/2015/11/12/the-complete-18-steps-to-start-a-new-agile-data-science-project/)

## Technical Specifications

We deployed Spark in local mode in a 8 cores 2.5 Ghz i7 Macbook Pro with 16GB of RAM.

Spark version 1.5.0

Notebook version 1.6.1

Scala version 2.10.4

## Limitations

We run our experiments in Spark local mode since that the size of data was small enough but the implementation is so that it can scale for any size and any cluster.

We did not leverage the DataFrame optimizations for the engineering part but we prefered the flexibility and functionalities of the classic RDDs.

Features independency was not verified, for example we expect tagsLikelihood and categoriesLikelihood features to be highly correlated. PCA or similar techniques could have been adopted to make the solution more reliable.

The whole analysis/modelling was performed statically without considering timestamps or sequence of events.

In order to reduce the search space in the evaluation we pre-filtered only the blog posts who were liked at least once in the test dataset. That is obviously cheating but we needed to reduce the search space for our experiments because we did not use a proper Big Data cluster.

Same for evaluated users, we only selected 100 random ones.

Bear in mind that ehe goal was not proving the correctness of the solution but more showing how you can easily implement and end-to-end scalable and production-quality solution for typical data science problem by leveraging Spark, Scala and the SparkNotebook.

## Conclusions

Key findings are:

* DataFrame is great for I/O and schema inference from the sources.
* DataFrame is good when you have flatten schemas, operations start to be more complicated with nested and array fields.
* RDD gives you the flexibility of doing your ETL using the richness of the scala framework, in the other hand you must be careful on optimizing your execution plans.
* Functional Programming allowed us to express complex logic and execution plans with a simple and clear code free of side effects.
* ETL and feature engineering is the most time-consuming part, once you obtained the data you want in vector format then you can convert back to DataFrame and use the ML APIs.
* SparkNotebook is good for EDA and as entry point for calling APIs and presenting results
* Developing in the notebook is non very productive, the more you write code the more become harder to track and refactor previously developed code. 
* Better writing code in IntelliJ and then either pack it into a fat jar and import it from the notebook or copy and paste everytime into a notebook dedicated cell.
* In order to keep normal Notebook cells clean, they should not contain more than 4/5 lines of code or complex logic, they should ideally just code queries in the form of functional processing and entry points of a logic API.
* Map joins with broadcast maps is very efficient but we need to make sure to reduce at minimum its size before to broadcast, e.g. applying some filtering to remove the unmatched keys before the join or capping the size of each record in case of size-variable structures (e.g. hash maps).
* ML unfortunately does not wrap everything available in MlLib, sometime you have to convert back to RDD[LabelledFeatures] or RDD[(Double, Vector)] in order to use the MlLib features (e.g. evaluation metrics).
* Plotting in the notebook with the built in visualization is handy but very rudimental, can only visualize 25 points, we created a Pimp to take any Array[(Double,Double)] and interpolate its values to only 25 points.
* Tip: when you visualize a Scala Map with Double keys in the range 0.0 to 1.0, the take(25) method will return already uniform samples in that range and since the x-axis is numerical, the built-in visualizatin will automatically sort it for you.
* Probably we should have investigated advanced libraries like Bokeh that are already supported in the Notebook.
* Even just 4/5GB of data in local mode is a bottleneck, Spark helps parallelizing but when the available data is too low is very slow and sometime runs out of memory.
* We could not find a way to increase memory in the notebook to more than 4GB even though we specified 12G in the spark configurations metadata.
