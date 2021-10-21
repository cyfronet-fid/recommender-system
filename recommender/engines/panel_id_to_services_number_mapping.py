"""This module contains mapping between recommendation
 panel_id and the services number to display in related panel.
 """

PANEL_ID_TO_K = {"v1": 3, "v2": 2}
K_TO_PANEL_ID = {v: k for k, v in PANEL_ID_TO_K.items()}
