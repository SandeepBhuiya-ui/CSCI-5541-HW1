Rubric (15 points)
Code looks good, i.e., each cell in the Jupyter Notebook runs without error and outputs intended results. (+3)
Description of the task, dataset, models, and hardware used (+1)
Includes appropriate references (+1)
Explains how they checked their model was trained correctly using learning curve graphs or other
appropriate information (+2)
Specifies evaluation metrics used in the experiment (+1)
Discusses test set performance and comparison with score reported in original paper or leaderboard.
Includes justification if it differs from the reported scores. (+2)
Includes training and inference time (+1)
Includes hyperparameters used in the experiment (+1)
Hypothesis of model performance and/or some kind of discussion about what they found in their
incorrectly labeled samples (+1)
Minimum of ten incorrectly predicted test samples with their ground-truth labels (+1)
Discusses potential modeling or representation ideas to improve the errors (+1)
Annotation of error types and potential fixes (Step 5: +1 extra credit)
Error visualizations and qualitative Analysis (Step 6: +1 extra credit)
Task, Dataset, Models, and Hardware Description:
Task : We will be choosing the sentiment classification task.
Dataset :  The SST2 dataset which was used for the sentiment analysis. 
Model : We will be using the BERT base model (uncased).
Hardware Description : the code which was used for this model !nvidia-smi


References:
References to Hugging Face Libraries:

Citation:
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2019). HuggingFace's Transformers: State-of-the-art Natural Language Processing. ArXiv, abs/1910.03771.

Description:
The reference is taken from the library which is given to us by the HuggingFace Library known as ‘Transformers’. Within this Library, we can find a lot of pre-trained models which are designed for different NLP tasks. For the notebook, I have used the library for access to ‘distilbert-base-uncased’ model, this model is used for better optimization which ends up for faster production of the results while keeping most of the original model’s accuracy which can be used for sentiment analysis on the SST-2 dataset.
Reference to Hugging Face Tutorial on Fine-Tuning :
Citation:
Hugging Face. (n.d.). Fine-tuning a pretrained model. Hugging Face Course. Retrieved from https://huggingface.co/docs/transformers/training

	Description:
This was the online resource which gives me a detailed explanation and a guide to fine-tuning  the model. It broke down the steps which are needed for fine tuning the model by including most of the methods which are used for loading datasets,preprocessing the text data, configuring the training parameters and at the very end, implementing the training loops.

Overview of the Custom Training Loop:
The custom trainer which is present is a class which is an extension of the hugging face’s ‘Trainer’ class. This was customized to find a comprehensive review and tracking of the training which was done and the evaluation metrics which are the loss and the accuracy of the model per epoch. The customization which happens after this gives an overview or a model progression report over time thereby helping us in finding out potential cases of overfitting or underfitting scenarios.

Major Components of the Custom Trainer:
Training Loop:
This is used to iterate over the entire dataset for the amount of epochs which are mentioned, which is later used to calculate the loss and the accuracy for every batch of training which is done and later tested by comparing the model’s predictions which are taken against the ground truth labels. The optimizer then would update the model’s weights which are based on the values computed of the gradients and we then aim to minimize the loss over time.
Evaluation Loop:
The evaluation loop would run at the end of each training epoch to find out or get an assessment on the model’s performance on a different validation set. This step being the most crucial as it determines how the model would generalize the unseen data.
Metric Tracking:
This step of the model would be focusing on accumulating the training and evaluation loss and along with which the accuracy metrics as per each epoch providing the data for analysis and visualization. 
Time Measurement:
Taking a total measurement of the time it takes for the training time, thereby offering insights into the computational efficiency of the training process.

Analysis of Training Results:
Training and Evaluation metrics
First Epoch: 
The model started with a training loss of 0.201 and accuracy of 92.18%, which indicated a strong initial learning capability. The evaluation metrics, with a loss of 0.226 and accuracy of 89.89%, suggest that the model generalizes well to the validation set at this stage. 

Subsequent Sequences: 
We can see that as the training progresses, we observe a consistent decrease in the values of the training function or loss (after reaching 0.020 by the fourth epoch) and with an increase in training accuracy (up to 99.27%). But on further observation, I noticed that the evaluation loss increased from 0.226 to 0.412 and the evaluation accuracy would decrease slightly to 87.50%. This change in the values between training and evaluation would raise the suspicion to be a case of overfitting, this is where the model learns the training data very well but ends up being ineffective at generalizing new unseen data.

Total Time: 
The total time for the training process takes approximately 23.679 minutes.

Conclusions and Recommendations:
Model Performances: 
After assessing the ‘CustomTrainer’ successfully , we can enhance the model’s learning over epochs. After assessing the trainer, we notice that there is a decrease in the training loss and increasing training accuracy. But later notice that the increasing evaluation along with the relatively high accuracy would suggest that there is overfitting.

Suggestions:
To overcome the overfitting and improve the model generalization, we could introduce different regularization techniques such as dropout or experiment with different model architectures, or increase the size of the training dataset. We could also use early stopping and thus this could be implemented to stop the training when the evaluation loss starts to increase thereby preventing overfitting.
Evaluation Metrics:
The Evaluation metric, which is used for our model training is the ‘accuracy’ which is computed by Hugging Face’s ‘compute_metrics’ function. The applicability of this function is that accuracy is the most straightforward metric for classification tasks, thereby representing the proportion of correct predictions which are made. 

Interpretation: When analyzing the model’s performance, a high accuracy rate indicates that the model can correctly classify the sentiments of texts most of the time. However it is also valid to consider the dataset’s balance. In the case where there is imbalance, accuracy might not consider the model’s performance across all classes. 
Test Set Performance and Comparison:
My Model Performance: 
Final Training Accuracy: 99.27%
Final Evaluation: 87.50%
The following metrics indicate that even in the case where our model achieves a high accuracy on the training set, it could result in a noticeable drop in the case of the validation set indicating to be a case of overfitting.
Benchmark Comparison: 91.3% (as reported on papers with code for the SST-2 binary classification task).
The comparison would reveal that there is a gap of approximately 3.8% between my model’s validation and the benchmark. There could be several reasons for the gap such as : 
Model Configuration: Differences in the specific configuration
Data Preparation: There could be variations in how the dataset is pre-processed.
Regularization Techniques: The original benchmark might be using a more sophisticated benchmark.
Ways to Improve the model performance:
Hyperparameter Optimization: Trying out different settings for the learning rate and a different batch size and the number of training epochs.
Better Regularization: We could also integrate dropout, label smoothing etc to augment the training data.
Review Training Process: The training process closely aligns with the benchmark, including any specific training techniques.
This is the Training Loss Vs the Validation Loss. 

Training and Inference Time:
Reported Training Time: The Custom training loop implemented in the project shows a total time of about 23.679 minutes for all of the training process.
Hyperparaters:
Specified Hyperparameters : The Hyperparameters used for this training involves ‘TrainingArguements’, including the learning rate (2e-5), batch size (16), number of training epochs (5) and weight decay (0.01).
Hypothesis Formulation:
Based on the comparative with respect to the basic model which was made:
“Fine tuning the pre-trained DistilBERT model on a the SST-2 dataset for sentiment analysis significantly improves its performance on that task in comparison to the basic model which was pre-trained without a task-specific training”
The custom training loop which was implemented for fine-tuning the DistilBERT model on the SST-2 dataset demonstrates a systematic decrease in the training loss and also an increase in the training accuracy over all the epochs. This tells us that the model is effectively learning from the task-specific data. 
The final evaluation accuracy which was achieved after fine-tuning (87.50%) is evidence of the hypothesis 
Comparison with the base model:
The understanding gathered from evaluating both the base model and the fine-tuned model on specific input texts let us know that for the base model predictions are identified as positive(‘LABEL_1’) with some accuracy which ranges from 0.508 to 0.521. This type of uniformity suggests that there could be a general bias towards positive sentiment, likely stemming from the model’s pre-training data distribution.
For the Fine-tuning: We have noticed that it has a higher accuracy score in its classification s but also results in an improved sensitivity for the sentiments in the sentences. The fine-tuned model correctly identified the sentiments( the negatively classified sentences ) and gave it a high score which the base model failed to make. 
Impact of Fine Tuning: The comparative analysis helps in showing the benefits of fine-tuning showing the model’s accuracy in its predictions.
Implications for sentiment analysis: The observed performance improvements would suggest that for sentiment analysis tasks, it would be interpretations of the SST-2 dataset.
Minimum of Ten Incorrectly Predicted Samples:
Sentence: 'it 's a charming and often affecting journey . '
Predicted Label: Negative
Actual Label: Positive

Sentence: 'allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker . '
Predicted Label: Negative
Actual Label: Positive

Sentence: 'the acting , costumes , music , cinematography and sound are all astounding given the production 's austere locales . '
Predicted Label: Negative
Actual Label: Positive

Sentence: 'although laced with humor and a few fanciful touches , the film is a refreshingly serious look at young women . '
Predicted Label: Negative
Actual Label: Positive

Sentence: 'or doing last year 's taxes with your ex-wife . '
Predicted Label: Positive
Actual Label: Negative

Sentence: 'the mesmerizing performances of the leads keep the film grounded and keep the audience riveted . '
Predicted Label: Negative
Actual Label: Positive

Sentence: '... the film suffers from a lack of humor ( something needed to balance out the violence ) ... '
Predicted Label: Positive
Actual Label: Negative

Sentence: 'even horror fans will most likely not find what they 're seeking with trouble every day ; the movie lacks both thrills and humor . '
Predicted Label: Positive
Actual Label: Negative

Sentence: 'a gorgeous , high-spirited musical from india that exquisitely blends music , dance , song , and high drama . '
Predicted Label: Negative
Actual Label: Positive

Sentence: 'audrey tatou has a knack for picking roles that magnify her outrageous charm , and in this literate french comedy , she 's as morning-glory exuberant as she was in amélie . '
Predicted Label: Negative
Actual Label: Positive

Potential Modelling OR Representation Ideas to Improve the errors:
For better potential Modeling and representation Ideas to improve the errors would involve the following recommendations:


Better Contextual Understanding:
Having a pre-trained model such as ROBERTa, XLNet ,etc which offer a better grip on the contexts present or conceptual representation. These models have been proven to have a better contextual representation on these topics.
Domain Specific Fine-Tuning:
Conducting better fine tuning for the additional datasets that are closely related to or tend to overlap with their domain of interest.
Improved Relations Of Negation and Sentiment:
Negation Handling : Implement custom preprocessing or post processing steps that explicitly mention the marked negative contexts in the sentence. 
Sentiment Lexicons: Integrating sentiment lexicons as the additional inputs might help in identifying the negation scopes.
Model Architectural Adjustments and Hybrid Models:
Experimentation with custom attention mechanisms or modification to existing ones to enhance the model’s focus to the sentimental crucial part of the texts. Considering the Hybrid models would also be beneficial ,specially ones that consider transformer architectures with the rule based systems. 


