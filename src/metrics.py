"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def calculate_topk_accuracy(df, topk_values):
    """
    Calculates the Top-k accuracy for each value of k in topk_values.

    This function iterates through each row of the DataFrame to determine how often the true code 
    appears within the top k predictions, where k is each value in the topk_values list. The accuracy 
    for each k is the number of times the true code appears in the top k predictions divided by the 
    total number of rows in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the columns 'code' and 'codes', where 'code' is the true 
      code and 'codes' is a list of predicted codes.
    - topk_values (list of int): List of integer values of k to calculate the Top-k accuracy for.

    Returns:
    - dict: A dictionary with keys as the values of k and the accuracies as the values.
    """
    topk_accuracies = {k: 0 for k in topk_values}

    for index, row in df.iterrows():
        true_code = row['code']
        predicted_codes = row['codes']
        seen = set()
        unique_candidates = [x for x in predicted_codes if not (x in seen or seen.add(x))]

        for k in topk_values:
            if true_code in unique_candidates[:k]:
                topk_accuracies[k] += 1

    total_rows = len(df)
    for k in topk_values:
        topk_accuracies[k] = topk_accuracies[k] / total_rows

    return topk_accuracies