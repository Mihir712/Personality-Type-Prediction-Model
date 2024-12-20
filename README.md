I designed this prediction model as part of a project which examined the implementation of a machine learning algorithm to predict personality traits (introversion vs. extroversion) among UNC Chapel Hill students. The primary focus of the study was to ensure and investigate ethics and fairness in different data collection methods for a machine learning model. 

    [Collaborators in the original study: Akshara Kolipaka, Mikayla Tuttle, Zimu Guan]
  
To preserve both the anonymity and privacy of the students we sampled during our study, the file included in this repository contains data that was purely generated from ChatGPT.

This model takes in a dataset containing 50 responses to 5 questions (features) and returns a single prediction (Introverted/Extroverted) for each response.
Input features include an individual's:

    • Perception of their personality
  
    • College Major
  
    • Number of Social Events Attended Per Week
  
    • Preferred Study Location
  
    • Housing Situation (On/Off Campus)

Once predictions have been made for each row of the dataset, the overall ratio of correct responses is calculated using a Confusion Matrix and displayed as the model's "Overall Accuracy."

Features like “Major” were ultimately not implemented when designing the model due to the possibility that it could reinforce a bias or stereotype. The 3 features that the model took into account were “Whether the individual felt they were Introverted or Extroverted,” “Whether the individual lived On or Off Campus,” and “How many social events the individual attended per week.” These were the features that both required the least amount of assumptions when assessing their effects and were the most “algorithmic” of the 5. That being said, the 2 features that the model didn’t consider were used to subjectively assess its accuracy.

To numerically assess accuracy, the predictions based on the On/Off Campus and Social Events questions are compared with the Individual Feeling question. If both yield the same reponse, that prediction is considered a "success."
  
The model makes use of the Random Forest algorithm because its structure involves processing multiple categories or “groups” to reach a final value. Due to the social nature of this scenario, the Random Forest algorithm introduces a flow chart - like thought structure, as the model goes through multiple questions (features in this case) one by one and translates the result into a single boolean score. A crucial step in the Random Forest algorithm is ensuring that the sample sizes of all datasets being considered are equal, stressing the importance of balanced sampling.
  
The final results were heavily dependent on the On-Off Campus and Number of Social Events questions, but I did play around with the feature importance of each. Importing the pandas and sklearn libraries allowed me to find values for each of these features that would make the most sense for predictions, and these were the values that yielded the greatest overall accuracy for each dataset.
  
The training process involved generating new data from ChatGPT and feeding this new data into the model. The goal was to see how it performed and whether or not it reinforced any pre-existing biases. My team and I validated the model by comparing the results to the inputted data and considering it from multiple angles (i.e. if a result seems plausible based on the raw data or not).

Feel free to play around with this model and insert your own spreadsheet of the same format to view its predictions!

Sample Input:
<img width="1238" alt="Screen Shot 2024-12-20 at 2 53 57 PM" src="https://github.com/user-attachments/assets/e7cb4af1-fb6d-4d97-a566-817277bdb175" />
(Data shown was generated by ChatGPT, and extends to 50 rows)

Sample Output:

<img width="342" alt="Screen Shot 2024-12-20 at 2 59 14 PM" src="https://github.com/user-attachments/assets/aa8e1d81-0108-47a3-800a-7850159131f1" />
