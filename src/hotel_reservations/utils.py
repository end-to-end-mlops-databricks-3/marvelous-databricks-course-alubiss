"""Utility class."""

import numpy as np
import pandas as pd


def serving_pred_function(client_ids: np.ndarray, banned_client_list: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
    """
    Adjust predictions: if a client is on the banned list, set their prediction to 1.

    :param client_ids: Array of client IDs corresponding to the predictions.
    :param banned_client_list: DataFrame containing a column 'banned_clients_ids' with banned client IDs.
    :param predictions: Array of model predictions.
    :return: Adjusted predictions array, where banned clients have prediction set to 1.
    """
    banned_ids = set(banned_client_list["banned_clients_ids"].values)
    adjusted = np.array([
        1 if client_id in banned_ids else pred
        for client_id, pred in zip(client_ids, predictions)
    ])
    return adjusted
