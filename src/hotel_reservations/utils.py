"""Utility class."""

import numpy as np
import pandas as pd


def serving_pred_function(client_ids: list, banned_client_list: pd.DataFrame, prediction: float) ->float:
    """
    Adjust predictions: if a client is on the banned list, set their prediction to 1.

    :param client_ids: Array of client IDs corresponding to the predictions.
    :param banned_client_list: DataFrame containing a column 'banned_clients_ids' with banned client IDs.
    :param predictions: Array of model predictions.
    :return: Adjusted predictions array, where banned clients have prediction set to 1.
    """
    banned_ids = set(banned_client_list["banned_clients_ids"].values)
    adjusted= [1 if client_id in banned_ids else prediction for client_id in client_ids]
    return adjusted
