# LanguageDetection
Using two neural network builders to create language detection networks and analyzing the results

Topic: Natural Language Processing (Machine Learning Pipeline)



Introduction
The topic of our project is Natural Language Processing. This was decided as both of us in our team have had prior experiences in creating/studying machine learning. The assignment is to implement the Machine Learning pipeline so to make this a useful experience, we decided to look for a field or type of machine learning that we hadn’t previously tried our hands at and so NLP was decided on as both an interesting topic as well as one with many real life applications



About The Project
While looking for NLP projects, we decided to implement a network that would not be extremely difficult but at the same time wouldn’t be something basic that would warrant little to no effort. Looking online and at the process of creating NLP networks, Language Detection stood out the most. We can easily challenge ourselves without inventing new technology by creating a network by simply trying more languages, changing the way we filter the data, finding new ways to encode the data, etc. and still have a meaningful learning experience



The Dataset
As for datasets, we decided to go very basic and pull data from kaggle, where various datasets consisting of text and labels can be found in different stages of processing. We focused on Kaggle datasets in order to allow us to put the majority of our effort into the network and its functionality, rather than attempting to improve or expand performance  through implementing more data.
 
The dataset, visualized above with Pandas, has 10,000 + entries. The Text also includes noise and undesirable characters and symbols, which had to be addressed in the preprocessing stages.



Methodology 
Our process includes 3 main steps:

Preprocessing and Data Preparation
During this phase, we focused on gathering all the data we could, and making sure we implemented a universal method of processing data. This allowed us to streamline the preprocessing and increase the speed of new predictions. It also served to facilitate easier re-training with new data for in the future when we want to expand its functionality.
❖	We first removed all unnecessary characters, i.e. symbols. 
❖	Converted all characters to lowercase
❖	Tokenize the sentences, constructing numeric representation
❖	The next step here is up in the air for now, but depending on our final implementation, we first need to convert our data into vectors, with each dimension representing a word. Additionally, a couple of libraries could do that with the tokens inside the network themselves, i.e. using TensorFlow’s Embedding Layer as the input layer.
❖	Finally, the data will be split into train and test; for now we will aim for an 80-20 split

Model Selection and Implementation
This is a big part of the Machine Learning Pipeline process and so to properly reflect a thorough study and consideration we decided to test out two models and find out which is better suited or has the better process. 

We initially used the SKLearn MultinomialNB model. This was supported by a vectorised input, achieved using SKLearns CountVectorization function. 
 

Next, we employed the use of a fairly more transparent model, and decided to create a tensorflow model. To support this, our input format changed and we utilized arrays of tokenized words, using Tensorflow’s Tokenizer function. The structure of the model is as follows:
 

We decided to go with the second model as it afforded us more insight into what the algorithm actually looked like and we could more easily explore the features and different moving parts that powered this model, than with MultinomialNB, which we couldn’t really take a deeper dive into.


Evaluation, Prediction/Inference, and Results
In addition to outputting the final accuracy, loss, recall, etc. for both training and testing data, we can also analyze some visualizations, and included is a confusion matrix to visualize not only what the final predictions are but in general which languages the model gets wrong and if its answer makes sense in anyway (and if they don’t what that means or how we can fix that), as a way to further develop the model:
  
Above to the right we have the scores for our first model. It performed extremely well in most categories (avg. of 98%) and we can see that there is a generally impressive score of precision especially (99%). This also lends to the fact that the preprocessing done here was extremely useful as the vectorised input seemed to allow for a simpler model structure and yet produce such excellent results. On the left are the scores from the second model. It has an ever so slightly lower score than our first model (avg of 97%) but we still decided to go with this model for a couple of reasons:
❖	The preprocessing for this model is much simpler to follow and in a situation where the original functions that carried our preprocessing were not available, the structure could be replicated in plain Python even if need be.
❖	The second model offers more insight into what is happening behind the scenes, as mentioned earlier. Additionally, difference in accuracy is not very large so we went with one that would allow us to demonstrate the model and be able to explain in enough detail what is happening

Below, we also included a confusion matrix of our second model. You can immediately see how accurate it is. The numbers are a bit hard to read but the important part is the very low false positives in basically all categories. This proves that our method is an effective way to recognize languages without any mixup. You can especially note that even languages like English and Daish which are notoriously similar and use the same letters do not trip up our model as well. This could be attributed to the way tokenisation is done based on words not letters,this was a major influence in deciding that as languages like these could have some of the same letters but not the same words:
 



Tools
Our primary coding language is Python, in particular we are employing the use of IPython notebooks to facilitate easy code sharing and the convenience of easily importing most ML libraries. To create the project the necessary libraries needed include:
❖	Pandas (preprocessing)
❖	Numpy (preprocessing)
❖	re (preprocessing)
❖	Tensorflow (model building and preprocessing)
❖	SKLearn (model building and preprocessing)
❖	Seaborn (visualization)
❖	matplotlib (visualization)


References
Natural Language Processing - Tokenization (NLP Zero to Hero - Part 1)

Tensorflow documentation
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

SKLearn documentation
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html


Code

Initial attempt with SKLearn
https://colab.research.google.com/drive/1iBd6xlZbN93tik5AKJQcqUfo0tzaz6DW?usp=sharing

Attempt two with Tensorflow
https://colab.research.google.com/drive/1Udyze6FTHB1PRZXTmQLkc9VUCpo77F-J?usp=sharing
