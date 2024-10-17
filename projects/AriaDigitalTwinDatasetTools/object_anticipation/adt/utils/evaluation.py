class LLMEvaluation:
    def __init__(self, llm_predictions, ground_truth):
        self.llm_predictions = llm_predictions
        self.ground_truth = ground_truth
        self.correspondences = []

        # total ground truth
        self.total_ground_truth = len(ground_truth)
        self.total_llm_activations = len(llm_predictions)
        self.total_llm_predictions = 0
        self.total_actual_correspondences = 0

        # true & false positves amd false negatives
        self.Tp = 0    # True Positives
        self.Fp = 0    # False Positives
        self.Fn = 0    # False Negatives

    def calculate_FP_FN_GT_correspondances(self):
            
        """ 
        Match LLM predictions with GT values 
        
        Cases: 
            1. LLM didnt predict for an interaction 
                - Example: GT was less than LLM time. then we check for the next GT until one is bigger than current LLM. 
                - Result: FN

            2. LLM time and prediction does not correspond to any GT interaction. (FALSE POSITIVE)
                - Example: Two consecutive LLM times are less than current gt time which means that 1st llm time does not correspond to any GT value
                - Result: FP
                
            3. LLM time and prediction corresponds to two GT values or one but we have also on FN
                - Example: Current LLM time is less than GT time but next LLM time is bigger than next GT tine which means that current LLM time corresponds to two consecutive GT predictions or to one GT with the Next GT be FN
                - Result: TP/FP double or TP/FP and 1FN
                
            4. LLM time and prediction correspomnds to one GT value
                - Example: Current LLM time is less than GT time and next LLM time is less than next GT time       
        """
    
        # Initialize variables
        llm_times = sorted(map(float, self.llm_predictions.keys()))
        gt_times  = sorted(map(float, self.ground_truth.keys()))

        # Create the iterator for ground truth times
        gt_iter = iter(gt_times)
        current_gt_time = next(gt_iter, None)
        next_gt_time = next(gt_iter, None)

        # Iterate through the LLM predictions
        for i, current_llm_time in enumerate(llm_times):

            if current_gt_time is None: 
                self.Fp +=1
                self.total_llm_predictions +=1
                continue

            # Get the next LLM time
            next_llm_time = llm_times[i + 1] if i + 1 < len(llm_times) else None
            # Check if the ground truth time is before the first LLM prediction

            while current_gt_time is not None and current_gt_time < current_llm_time:
                self.Fn += 1  # No LLM activation before the ground truth
                current_gt_time = next_gt_time
                next_gt_time = next(gt_iter, None)
            
            # Check if two consecutive LLM times occur before the current ground truth time
            if current_gt_time is not None and next_llm_time is not None and next_llm_time < current_gt_time:
                self.Fp += 1  # False positive: as current LLM prediction does not correspond to any ground truth
                self.total_llm_predictions +=1
                continue
            
            # Check if the current LLM corresponds to the current or next two ground truths
            if current_gt_time is not None and current_llm_time <= current_gt_time:
                time_diff = next_gt_time - current_gt_time if next_gt_time is not None else float('inf')

                matched_ground_truths = []

                # Option 1: next LLM time is larger than next ground truth time so 1 LLM prediction for 2 ground truths 
                if next_llm_time is not None and next_gt_time is not None and next_llm_time > next_gt_time:
                    matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
                    self.total_llm_predictions +=1
                    current_gt_time = next_gt_time
                    next_gt_time = next(gt_iter, None)

                    # Either it's a missed prediction (false negative) --> for the next gt there is no LLM prediction
                    if time_diff > 1:
                        # for current gt there is a llm
                        self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
                        # for next gt there is not llm (as next llm > next gt)
                        self.Fn += 1

                    # Or it refers to two ground truth events
                    if time_diff <= 1:
                        matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
                        self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
                        self.total_llm_predictions +=1
                        current_gt_time = next_gt_time
                        next_gt_time = next(gt_iter, None)
                
                # Option 2: Next LLM time is smaller than next ground truth time
                elif next_llm_time is not None and next_gt_time is not None and next_llm_time <= next_gt_time:
                    matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
                    self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
                    self.total_llm_predictions +=1
                    current_gt_time = next_gt_time
                    next_gt_time = next(gt_iter, None)

                # Option 3: Next next ground truth time is None
                elif next_llm_time is not None and next_gt_time is None:
                    matched_ground_truths.append((current_gt_time, self.ground_truth[current_gt_time]))
                    self.correspondences.append((current_llm_time, self.llm_predictions[current_llm_time], matched_ground_truths))
                    self.total_llm_predictions +=1
                    current_gt_time = next_gt_time
                    next_gt_time = next(gt_iter, None)

        # After the LLM loop ends, handle remaining ground truth times
        while current_gt_time is not None:
            self.Fn += 1  # False negative: LLM did not predict while ground truth remains
            current_gt_time = next_gt_time
            next_gt_time = next(gt_iter, None)

    def calculate_final_TP_FP(self):
   
            """
            ********* Note *********
            
            --> in our case we predict 3 potential objects that the user may interact with.
            --> if the actual object that the user interacts with is one of the 3 potential objects it counts as TP 
                
            ********** TP **********

            --> TP : If at least 1 of the predicted objects is the GT object 
            --> FP : If none of the predicted objects is the GT objects or predictions does not correspond to any interaction
            --> TN : TN would be if the LLM predicts no interaction and no any interaction takes place. But LLM is activated for prediction so in our case there is no TN 
            --> FN : FN would be if the LLM precicts no interaction but there is intetacton. But LLM is activated for prediction so in our case there is not FN 
            """

            for llm_time, prediction, correspondence in self.correspondences:

                # Loop over the gt values that correspond to one LLM prediction
                for gt_time, gt_object in correspondence:
                    
                    self.total_actual_correspondences +=1 

                    # Flag to check if a TP is found in this correspondence
                    match_found = False  
                    
                    if gt_object in prediction:
                        self.Tp += 1
                        match_found = True
                        continue  # Break as soon as we find a match, considering it a TP
                    
                    # If no match was found, this is a false positive
                    elif not match_found:   
                        self.Fp +=1
            
    def calculate_metrics(self):
            
            """  
            Correspondances gives us information about the
                1. LLM total activations
                2. Actual interactions
                3. Correct predictions 
            
            Based on these three, we calculate metrics to evaluate the performance of our algorithm

            ******* Metrics *********

            1. Model_Overall_Accuracy: 
                    - Explanation: Measures the proportion of correct predictions out of total predictions
                    - Question:    Among all instances how many identied correcty as positives (relatives) and negatives (non-relatives)
                    - Intuition:   Accuracy is the most straightforward metric, measuring the overall correctness of the model. It represents the proportion of all correct predictions out of the total number of instances.
                    - Formula:     Model_Overall_Accuracy = TP + TN / TP + FP + TN + FN 
                    - Updated:     Model_Overall_Accuracy = TP / total llm activations (based on definition of Accuracy and the specifications of this problem)

            2. Precision (Positive_Prediction_Accuracy) : 
                    - Explanation: Measures the proportion of true positive predictions out of the total predicted positives.
                    - Question:    Of all the instances that were predicted as positive, how many were actually positive?   
                    - Intuition:   Precision is a metric that measures the proportion of correctly identified relevant instances out of all instances that were identified as relevant
                    - Formula:     Precision = TP / TP + FP 
            
            3. Recall (True_Positive_Rate): 
                    - Explanation: Measures the proportion of true positive predictions out of the total actual positives. 
                    - Question:    Of all the instances that were actually positive, how many were correctly predicted as positive?
                    - Intuition:   Measures the ability of a model to identify all relevant instances within a dataset. The natural intuition behind recall can be understood by considering its relationship to ”sensitivity” or ”true positive rate.
                    - Formula:     Recall = TP / TP + FN          
                    - Updated:     Recall = TP / total ground truths (since in this scenario we don't have FN or TN, we use the lenght of ground truthss
            
            4. LLM_Activation_Sensitivity 
                    - Explanation: Measures how easy the LLM is activated 
                    - Question:    From all LLM activtion how many corresponds to actual interaction
                    - Intuition:   Measures the sensitivity of the LLM to be activated 
                    - Formula:     LLM_Activation_Sensitivity = Actual_interaction / total llm activations
            
            5. LLM_Interaction_Accuracy:
                    - Explanation: Measures the proportion of correct predictions out of total actual interactions
                    - Intuition:   Measures the overall correctenss of the model. 
                    - Formula:     LLM_Interaction_Accuracy = TP / total_actual_interactions  
            """

            # Initialize counters
            self.total_ground_truth = len(self.ground_truth)            # Total number number of objects the user interacted with
            self.total_llm_activations = len(self.llm_predictions)      # Total number of LLM activated to make a predictions

            # Calculate Accuracy
            self.model_overall_accuracy = round((self.Tp / (self.total_llm_predictions + self.Fn)),3) if (self.total_llm_predictions + self.Fn) else 0

            # Calculate Precision
            self.precision = round((self.Tp / (self.Tp + self.Fp)),3) if (self.Tp + self.Fp) else 0

            # Recall is always 1 in this context because there are no FN cases
            self.recall = round((self.Tp / (self.total_actual_correspondences + self.Fn)),3) if (self.total_actual_correspondences + self.Fn) else 0

            # LLM sensitivity / How sensitive is the LLM to activation 
            self.llm_activation_sensitivity = round((self.total_actual_correspondences / self.total_llm_predictions),3) if self.total_llm_predictions else 0

            # LLM correctness is close to recall but does nto take into account the FN
            self.llm_interaction_accuracy = round((self.Tp / self.total_actual_correspondences),3) if self.total_actual_correspondences else 0
            
            return (self.model_overall_accuracy, self.precision, self.recall, self.llm_activation_sensitivity, self.llm_interaction_accuracy, 
                   self.Tp, self.Fp, self.Fn, 
                   self.total_ground_truth, self.total_llm_predictions, self.total_llm_activations, self.total_actual_correspondences, 
                   self.correspondences)
    
    def display_results(self):
        # Output the correspondences 
        for llm_time, prediction, correspondence in self.correspondences:
            if isinstance(correspondence, str):
                print(f"{llm_time}: {prediction} ----------------------> {correspondence}")
            else:
                gt_descriptions = ", ".join([f"{gt_time}: '{gt_object}'" for gt_time, gt_object in correspondence])
                print(f"{llm_time}: {prediction} ----------------------> {gt_descriptions}")

        # Output the Accuracy and Recall 
        print(f"\n Correct Predictions: {self.Tp}")
        print(f"\n Non correct Predictions: {self.Fp}")
        print(f"Total Ground Truth Instances: {self.total_ground_truth}")
        # print(f"Accuracy: {self.accuracy * 100:.2f}%")
        # print(f"Precision: {self.precision * 100:.2f}%")
        # print(f"Recall: {self.recall * 100:.2f}%")
    

# Usage example:
# llm_predictions = {...}  # dictionary with LLM predictions
# ground_truth = {...}     # dictionary with ground truth data

# evaluator = LLMEvaluation(llm_predictions, ground_truth)
# correct_predictions, total_ground_truth, accuracy = evaluator.evaluate()

# evaluator.display_results()