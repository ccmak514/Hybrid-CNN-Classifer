# A New Hybrid CNN Classifier and Compare the Performance with 2 CNN Transfer Learning Models
This is a project of AASD 4015 - Advanced Applied Mathematical Concepts for Deep Learning <br>
Contributor: 1. Chun Cheong Mak (101409987) 2. Hei Yuet Lee (101409639)

| Name | Tasks    |
| :---:   | :---: |
| Chun Cheong Mak | Data preprocessing, modify the functions for feature vectors, design and implement the algorithm of the new hybrid CNN classifier |
| Hei Yuet Lee | Build the baseline models for comparison, debug and test the algorithm  |

<font size=3>The purpose of this project is to explore a way to build a <font color=red>new hybrid CNN classifier</font> by calculating the cosine similarity of feature vectors of 2 different images. The project is inspired by the concept of <font color=red>ensemble learning</font> discussed in the AASD 4000 - Machine Learning I and an article about image search using <font color=red>cosine similarity</font> of feature vectors of 2 different images. After building the new hybrid CNN classifier, it will be compared with other CNN transfer learning models. The link and literature review of the article is listed below:</font>

### <b>Article - Image based search engine with CNN and Transfer Learning by Dhruv Shrinet</b>
<font size=3>The link of the article:</font>
https://medium.com/swlh/image-based-search-engine-with-cnn-and-transfer-learning-153a1a3e58b4
<br><br>
<font size=3>This article demonstrate a way to make a simlpe image searcher by calculating the <font color=red>cosine similarity</font> of feature vectors of 2 different images. The author can find a new image and search images look similar from his database.</font>

1. Use the first to the second last layer of a <font color=red>pre-trained model (ResNet50)</font> to build a feature vector extractor of images. 

2. Use the extractor to calculate the feature vectors of all images in database.

3. Calculate the feature vector of a new image that cannot be found in database.

4. Calculate the cosine similarity between the feature vector of the new image and all the feature vectors of the database.

5. Print the most similar images by sorting.

### <b>Data - 70 Dog Breeds-Image Data Set</b>
<font size=3>The link of Kaggle:</font>
https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set

### <b>The Main Idea of the New Hybrid CNN classifier in AASD 4015 Project 2:</b>
<font size=3>A new hybrid CNN classifier for classifying the breeds of dog is built as follows:</font>

1. Use <font color=red>the first to the second last layer</font> of a <font color=red>ResNet50</font> model to build a first feature vector extractor.

2. Use <font color=red>the first to the second last layer</font> of a <font color=red>Xception</font> model to build a second feature vector extractor.

3. Use the 2 extractors to calculate the 2 feature vectors of each images in training set.

4. Calculate the 2 feature vectors of a new image that is being classified.

5. Calculate the <font color=red>ResNet50 cosine similarity</font> of the <font color=red>ResNet50 feature vectors</font> of the new image and the <font color=red>ResNet50 feature vectors</font> of each image in each breed in the training set.

6. Calculate the <font color=red>Xception cosine similarity</font> of the <font color=red>Xception feature vectors</font> of the new image and the <font color=red>Xception feature vectors</font> of each image in each breed in the training set.

7. Use the idea of ensemble learning, calculate the weighted sum of the above 2 cosine similarity of the new image and each image in each breed in the training set.

8. Save weighted sum for each breed if the weighted sum is larger than certain threshold.

9. Count the total number of saved weighted sum for each breed.

10. Classify the new image by comparing the the total number of saved weighted sum of each breed.

<font size=3>More details about the mechanism of the hybrid CNN classifier will be discussed in the notebook. The main takeaways from this notebook:
- Explore, preprocess the dataset 
- Build functions and models for building hybrid CNN classifier
- Produce feature vectors
- Prediction by the hybrid CNN classifier
- Validation and Testing of the hybrid CNN classifier
- Build 2 baseline CNN transfer learning models
- Compare the performance with the baseline models by test accuracy</font>

### <b>Table of Contents:</b><br>
#### 1. Data Preprocessing 
#### 2. Data Visualization
#### 3. Models and Functions for building the Hybrid Model
#### 4. Feature Vectors
#### 5. Prediction, Validation and Testing of the Hybrid Model
#### 6. Compare with 2 CNN Transfer Learning Models
#### 7. Summary and Discussion

## 1. Data Preprocessing
<font size=3>After unzipping the file from Kaggle, we got a folder called data. we found that the number of images of each breed in train, valid and test folders in the data folder are <b>NOT</b> the same. To avoid the problem of data imbalance, we make a new folder called breed and three folders called train, valid and test inside it for preprocessing data.</font>

<font size=3>The minimum number of training images in one of 70 breeds is 65. Therefore, we should choose number smaller than or equal to 65 as the number of training images from each breed for data preprocessing.</font>

<font size=3>We choose 64, 8 and 8 images from each breed from the original train, valid and test set respectively. The reason is to maintain the ratio of 0.8, 0.1, 0.1.</font>

<font size=3> Now, we have totally 4480, 560 and 560 images for training, validation and testing.</font>
## 2. Data Visualization
<font size=3>By observation, some breeds look similar. For example, the color and shape of Basset and Beagle are close.</font>

![Basset and Beagle](https://user-images.githubusercontent.com/101066418/230729461-e1016847-e159-40a5-8c86-8a8f1312c35c.png)

<font size=3>For Maltese and Shih-Tzu, it is difficult to distinguish them by the naked eye.</font>

![Maltese and Shih-Tzu](https://user-images.githubusercontent.com/101066418/230729469-87284d2f-7d52-458e-b8e1-01d27e419a63.png)

<font size=3>Above breeds could be the potential challenge of building the new hybrid model.</font>
## 3. Models and Functions for building the Hybrid Model
<font size=3>The main reference of the coding in Section 3 is the article mentioned above: <b>Image based search engine with CNN and Transfer Learning</b>.
<font size=3>The pre-trained model of ResNet50 and Xception are used to build feature vector extractors. Only <font color=red>the first to the second last layer</font> are considered.</font>

<font size=3>This is the summary of ResNet50 and avg_pool is the layer to get the ResNet50 feature vectors (1,2048)</font>

<img width="694" alt="Resnet50" src="https://user-images.githubusercontent.com/101066418/230730011-3c1f7eae-cc1d-4c62-acbd-085b2430adb0.png">

<font size=3>This is the summary of Xception and avg_pool is the layer to get the Xception feature vectors (1,2048)</font>

<img width="696" alt="Xception" src="https://user-images.githubusercontent.com/101066418/230730020-9b1b7b9a-798b-48ee-af7a-7ca89c56db40.png">

<font size=3>The `model_resnet.layers[-2]` and `model_xception.layers[-2]` are the avg_pool layers of ResNet50 and Xception respectively. They are the second last layer in the ResNet50 and Xception models.</font>

<img width="601" alt="extractor" src="https://user-images.githubusercontent.com/101066418/230730259-5d5bf4b7-19a8-41e1-b023-d3644bcb686b.png">

<font size=3>`preprocess_img_resnet` and `preprocess_img_xception` are made to preprocess the image into proper format and data type for their corresponding feature vector extractors. These two functions are used in the `encode_img_resnet` and `encode_img_xception` respectively.</font>

<img width="359" alt="preprocess_image" src="https://user-images.githubusercontent.com/101066418/230730422-32bd64c8-b1ab-40c1-bf7a-77079097c36d.png">

<font size=3>`encode_img_resnet` and `encode_img_xception` will return the <font color=red>ResNet50 feature vector</font> and <font color=red>Xception feature vector</font> respectively of a particular image. The vectors will be used to calculate the cosine similarity.</font>

<img width="339" alt="Encode" src="https://user-images.githubusercontent.com/101066418/230730427-898dc975-4311-43d1-8e8f-03854f4d6702.png">

## 4. Feature Vectors
<font size=3>`feature_resnet` and `feature_xception` are two dictionaries for storing the <font color=red>ResNet50 feature vectors</font> and <font color=red>Xception feature vectors</font> for each breed. As we have 70 breeds to consider, there are 70 keys indicating the name of each breed in `feature_resnet` and `feature_xception`.</font>

<img width="270" alt="dictionary" src="https://user-images.githubusercontent.com/101066418/230730763-bccb0498-5658-4ea6-ab5d-5fc888c47b47.png">

<font size=3>For each key, there is a corresponding list for storing the 64 feature vectors. The following nested `for` loop is adding the feature vectors into each list by the 2 functions `encode_img_resnet` and `encode_img_xception`.</font>

<img width="487" alt="encode the image" src="https://user-images.githubusercontent.com/101066418/230730767-679f72c0-57c4-4d0e-824c-2ca9cfda7974.png">

<font size=3>Before reading Section 5, let's see what we have. We have gotten 2 dictionaries: `feature_resnet` and `feature_xception`. For each dictionary, there are 70 keys that are the names of the 70 breeds of dog. For each key, there is a list storing 64 feature vectors produced by `encode_img_resnet` or `encode_img_xception` (depends on whether it is `feature_resnet` or `feature_xception`).</font>
## 5. Prediction, Validation and Testing of the Hybrid Model
<font size=3>To explain the algorithm of the hybrid model clearly, I will explain with an example and my handwritten notes.</font>

<img width="467" alt="1" src="https://user-images.githubusercontent.com/101066418/230728110-8af8c043-2918-45df-8183-4384d5a90fcd.png">

<font size=3>In step 1, I input a new image which is a <b>Golden Retriever</b>. `encode_img_resnet` and `encode_img_xception` will generate a <font color=red>ResNet50 feature vector</font> and a <font color=red>Xception feature vector</font> respectively.</font>
<br><br>
<font size=3>In step 2, `feature_resnet` and `feature_xception` are examined. In this example, we will focus on the <b>Golden Retriever</b> in `feature_resnet` and `feature_xception` to illustrate the algorithm. In `feature_resnet`, there are 64 <font color=red>ResNet50 feature vectors</font> stored in a list which is corresponding to <b>Golden Retriever</b>. In `feature_xception`, there are 64 <font color=red>Xception feature vectors</font> stored in a list which is corresponding to <b>Golden Retriever</b>.</font>
<br><br>
<font size=3>In step 3, calculate the weighted sum of <font color=red>ResNet50 cosine similarity</font> and <font color=red>Xception cosine similarity</font> and check whether the weighted sum is larger than 0.5. In my handwritten note, $w$ is used to indicate the weight and $s$ is the cosine similarity function.</font>
<br><br>
<img width="486" alt="2" src="https://user-images.githubusercontent.com/101066418/230728182-4024d000-de00-4254-a755-d177cac20a2a.png">

<font size=3>In step 4, if the total number of <font color=red>weighted sum which is larger than 0.5</font> is higher than 43, <b>Golden Retriever</b> is considered to be one of the possible correct breeds. If not, just examine other breeds.</font>
<br><br>
<font size=3>In step 5, if the total number of possible correct breeds is just 1, then the hybrid model will classify the image as <b>Golden Retriever</b>. If the total number of possible correct breeds is zero, then the hybrid model will print: No such class! Your image may not be a dog. <font color=red>If the total number of possible correct breeds is larger than 1, go to step 6.</font></font>
<br><br>
<img width="464" alt="3" src="https://user-images.githubusercontent.com/101066418/230728226-94cb9983-b435-42ff-88af-c80e21c75b72.png">

<font size=3>In step 6, let say there are 2 possible correct breeds: <b>Golden Retriever</b> and <b>Boxer</b>. The algorithmn will sum up the corresponding weighted sum of <font color=red>ResNet50 cosine similarity</font> and <font color=red>Xception cosine similarity</font> and check whose sum is larger. In this example, the total sum of <b>Golden Retriever</b> is 39.4, while the total sum of <b>Boxer</b> is 21.3. The algorithmn will classify the input image as <b>Golden Retriever</b>.</font>

<font size=3>The mechanism of `prediction` below follows above rationale:</font>

<img width="904" alt="prediction" src="https://user-images.githubusercontent.com/101066418/230728666-1f79775e-0f5e-40f3-9175-faa8a17b3f11.png">

<font size=3>The `prediction` will return and print the name of the breeds, if the `prediction` identifies the input is an image of a dog.</font>
<font size=3>The `prediction` will return and print 'X' and print: "No such class! Your image may not be a dog", if the `prediction` identifies the input is <b>NOT</b> an image of a dog.</font>


<font size=3>Let's use `prediction` to check whether it can predict a Golden Retriever in validation set correctly.</font>

![output](https://user-images.githubusercontent.com/101066418/230728599-5ed07d43-769d-43ce-b8f4-c74d968fde05.png)

<img width="383" alt="prediction of a dog" src="https://user-images.githubusercontent.com/101066418/230728997-f46fc56c-d19f-4630-b4c4-af1a4d049b93.png">

![output2](https://user-images.githubusercontent.com/101066418/230728604-fcb07b12-dc27-45b2-a960-6bc43ee2c1d0.png)

<img width="383" alt="prediction of a human" src="https://user-images.githubusercontent.com/101066418/230729004-b9373479-dc0b-42f2-90c8-d776ea3b75df.png">

<font size=3>The hybrid classifier will <b>NOT</b> classify the image of human as one of the 70 breeds but tell you that you may input some irrelevant images. This is what a traditional CNN classfication model <font color=red><b>CANNOT</b></font> do.</font>

<font size=3>`accuracy` can calculate the valid accuracy or test accuracy. The idea is to use the above `prediction` to predict the valid and test set and count the number of correct predictions. Finally, the number is divided by the total number of images in valid set or test set and return the accuracy of the hybrid CNN classifier under different settings.</font>

<img width="727" alt="accuracy" src="https://user-images.githubusercontent.com/101066418/230729332-055b4c43-b655-4c85-adc3-d83c0cc3f081.png">

<font size=3>We will try different weights and check the validation accuracy. The setting which produces the highest validation accuracy will be used in the remaining section and its test accuracy will be checked.</font>

| Weight ($w$)  | Validation Accuracy |
| ------------- | ------------------- |
| 0             | 0.916               |
| 0.3           | 0.930               |
| 0.5           | 0.925               |
| 0.8           | 0.916               |
| 1             | 0.877               |

<font size=3>From above evaulation, 0.3 will be the weight and the test accuracy is 0.939.</font>

| Weight ($w$)  | Test Accuracy |
| ------------- | ------------- |
| 0.3           | 0.939         |


## 6. Compare with 2 CNN Transfer Learning Models
<font size=3>Two CNN models will be built by transfer learning. The first use <font color=red>ResNet50</font> and the second one use <font color=red>Xception</font>. After building the models, their test accuracy will be calcualted for comparison. This section is done seperately on Colab and <font color=red>the code in this section basically follows the code in the AASD 4015 Project 1 for simplicity</font>.</font>

The link to AASD 4015 Project 1: https://github.com/Jclee967/AASD4015-project1

#### Baseline Model 1: Transfer learning of ResNet50

![baseline_model1_accuracy](https://user-images.githubusercontent.com/101066418/230731158-9458d961-4fc1-4d92-b924-fff2a8be575e.png)

![baseline_model1_loss](https://user-images.githubusercontent.com/101066418/230731161-5a00a915-b1b6-4139-a59d-0c65ec41c377.png)

| Test Accuracy  | Test Loss |
| ------------- | ------------- |
| 0.913           | 2.423         |

#### Baseline Model 2: Transfer learning of Xception

![baseline_model2_accuracy](https://user-images.githubusercontent.com/101066418/230731298-0d6550a1-a405-4dc5-848d-d302905c117e.png)

![baseline_model2_loss](https://user-images.githubusercontent.com/101066418/230731301-e452c9d3-4dde-4e4f-8c17-cc6bf6b284ed.png)

| Test Accuracy  | Test Loss |
| ------------- | ------------- |
| 0.962           | 0.785         |

<font size=3>Last but not least, let also perform simple fine-tuning to the above two baseline models and make comparison. <font color=red>The approach and code of fine-tuning basically follows the latter part of the AASD 4015 Project 1 for simplicity</font>.

#### Fine-tuning Baseline Model 1

![baseline1_fine_tune_accuracy](https://user-images.githubusercontent.com/101066418/230731546-47fe0a22-f64d-487e-aca7-c53cf1760b7d.png)

![baseline1_fine_tune_loss](https://user-images.githubusercontent.com/101066418/230731549-3a985550-a6b3-4c88-bd10-fc8a8eee13be.png)

| Test Accuracy  | Test Loss |
| ------------- | ------------- |
| 0.916           | 2.443         |

#### Fine-tuning Baseline Model 2

![baseline2_fine_tune_accuracy](https://user-images.githubusercontent.com/101066418/230731594-96f29bba-0799-4641-9e74-11b635f17cc6.png)

![baseline2_fine_tune_loss](https://user-images.githubusercontent.com/101066418/230731597-0113d6fd-0031-471c-9ec7-2a4de8db195f.png)

| Test Accuracy  | Test Loss |
| ------------- | ------------- |
| 0.962           | 0.644         |

## 7. Summary and Discussion

| Model                                            | Test Accuracy |
| ------------------------------------------------ | ------------- |
| Hybrid CNN Classifier                             | 0.939         |
| Baseline Model 1: Transfer learning of ResNet50  | 0.913         |
| Baseline Model 2: Transfer learning of Xception  | 0.962         |
| Baseline Model 1 after Fine-tuning               | 0.916         |
| Baseline Model 2 after Fine-tuning               | 0.962         |

<font size=3>The test accuracy of hybrid CNN classifier is 0.939 that is at the middle among all the 5 models' test accuracy. Moreover, the fine-tuning is not customized. The approach is from the AASD 4015 Project 1 for simplicity. Although this hybrid CNN classifier is not an outstanding model, there are still some advantages I would like to mention.</font>

<font size=3>First, this hybrid CNN classifier can identify images that do not belong to the 70 classes. In Section 5, the hybrid CNN classifier can successfully identify that the image of human does not belong to the 70 classes. For traditional classification models, they will classify the image of human as one of 70 classes.</font>

<font size=3>Second, this hybrid CNN classifier is built without using GPU but this model also use CNN. This may be an alternative for somebody who don't have a laptop with GPU or don't want to pay to use Colab.</font>

<font size=3>Next, I want to mention some possible improvement. We may do hyper-parameter tuning to find out the best setting of this hybrid CNN classifier and also introduce more pre-trained models. Only ResNet50 and Xception are used in this hybrid CNN classifier. Third model like InceptionV3 may be a good choice to introduced.</font>

