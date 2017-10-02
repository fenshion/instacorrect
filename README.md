# InstaCorrect
End-to-end implementation of a **deep learning** (french) spell checker: www.instacorrect.com
* Data gathering and pre-processing
* Model definition
* Training and inference
* Model serving
* Front-end interaction

## Introduction
The ultimate goal of the project is to have a model that can effectively correct any french sentence for spell and grammatical mistakes. To this end, this repo implements the architecture from [*Sentence-Level Grammatical Error Identification as Sequence-to-Sequence Correction*](https://arxiv.org/abs/1604.04677)
In this paper the authors create a Sequence-to-Sequence model with a CNN over characters encoder. We will do the same!

## Data Gathering and Pre-Processing
The first step is to find a relevant dataset. In this case, there are no pre-made dataset available. We will have to make our own! For that, I downloaded all of the french translations from the [*European Parliament Proceedings Parallel Corpus 1996-2011*](http://www.statmt.org/europarl/), and news articles from the [*News Crawl: articles from 2014*](http://www.statmt.org/wmt15/translation-task.html). The resulting dataset contains around 13,000,000 sentences.

This gives us a set of correct sentences, at least in theory. We now have to come up with a set of erroneous sentences. We will generate them from correct one. In /Data/mistake.py, you will find a script that can automatically generate grammatical and spelling mistakes.
For instnace, *Rien ne sert de courir, il faut partir à point* could lead to *Rien ne sers de courir, il faut partir a point*. This mistake generator is still basic but at least we have somehting to start with.

Before creating our dataset, we will create two `vocabulary`. A vocabulary is a mapping of characters or words to integers. For example, the letter `b` will be assigned the number `1`, and the word `Network` the number `2504`. Every character in our dataset will be assigned a number, and the most common 50,000 words will be assigned a number, the less frequent words will be replace with a special id of `|UNK|`.

We can now create our dataset. It will consist of `tf.record` files filled with `tf.example`. Each entry will consist of:
1. The encoded input sentence, i.e., an array of integers. Each interger representing the associated character from the `dictionary`.
2. The input sentence length, i.e, the number of words in this sentence.
3. The correct output sentence, encoded word wise (not characters wise!).
4. The correct output sentence length.
This dataset will be split up in three parts: traning (99%), validation (1%) and testing (1%).

## Model Definition
The model definition follows the implementation of Schmaltz and Kim in [*Sentence-Level Grammatical Error Identification as Sequence-to-Sequence Correction*](https://arxiv.org/abs/1604.04677).
<p align="center" >
<img src="https://github.com/maximedb/instacorrect/blob/master/Misc/model_arch.PNG" alt="Model from Schamtz & Kim">
</p>

1. The inputs are the characters from erroneous sentences batched together: `[batch_size, max_sentence_length, max_word_length]`
   The inputs are zero padded.
2. The inputs are then embedded in a vector of dimension 30. The result is a tensor of shape `[batch_size, max_sentence_length, max_word_length, embedded_size]`
2. Many convolutional layer are then applied on these inputs, one word at a time. The result is a word embedding for each word in the dataset. Note that this embedding is not subject to out of vocabulary issues.
3. A two layer LSTM with 512 hidden units is applied on the word embeddings.
4. The final output of the LSTM is given to a decoder, a two layer LSTM with 512 hidden units. At each timestep a soft luong attention is performed on the encoder's ouput.
4. A dense linear layer with a softmax activation is applied on the final output of the LSTM to predict the next word id.
The loss is defined as the softmax cross entropy. It is minimized with the AdamOptimizer and gradient clipping.

## Training and Inference
The training and inference is done using a Tensorflow estimator. It is really useful and lets you focus on the essential part of the model and not the usual plumbing associated with running a tensorflow model. Furthermore, it is really easy to export a trained model using an estimator. The model reads examples using a tf.dataset. This functionality is great to read large files that would not fit into memory. Furthermore, there is a **dyanmic padding and bucketing** of the examples as to optimize the training time.

## Model Serving
Once a model is trained, it can be exported. It can be served to the "real world" with tensorflow serving. Tensorflow serving is not the easiest thing to grasp. It is not well documented, uses C++ code, the installation process is not clear, etc. Tensorflow serving is a program that lets you put your model on a server. This server accepts gRPC requests and returns the output of the model. In theory it sounds easy, in practice it not that easy.

Google released a binary that you can download using apt-get. This makes it much more easier. You just download this binary and execute it. It will launch your server. This server expects to find exported models in a directory you specified. In this directory you simply copy-paste your saved model from earlier. That's it. You can customize it more, but it does the job.

Now that's great. But it's not really user-friendly to ask your users to make gRPC request to your server. That's why you also need a simple application that can link the two together. Here I implemented a simple Flask application, that displays a simple website and exposes a single API function `is_correct`. When you make a POST request to this end-point with a text, it splits your text into sentences and calls the tensorflow-serving server with each sentences. It then multiplies the correct probabilities together to have a global correctnes probability and finally returns it to the front-end application.

All these applications are bundled together using Docker Compose and live on a DigitalOcean (mini) droplet.

## Front-end Application
A basic front-end application that runs angularjs to take the text written and send an AJAX request to the Flask app.

## Results and conclusion
The model achieves an accuracy of 95% on the validation set. This is already great! But in practice the model fails to detect basic errors. This is probably due to two things:
* The mistake generator is not really good. It generates unlikely mistakes more than likely mistakes. Your model is only as good as the quality of your data.
* The decoder vocabulary is limited to 50,000 words.

## Next Steps
The next model will need to have a larger decoder vocabulary as this is really limiting the current model. One way to do this would be to use convolutional embedding in the decoder as well. To improve the traning time, I will implement the transformer model from [Attention is all you need](https://arxiv.org/abs/1706.03762).

## Contributions
Feel free to contribute/comment/share this repo :-)
