# Databricks notebook source
!pip install databricks-sdk==0.32.0

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

srcs = w.data_sources.list()


alert_query = """
SELECT
  (COUNT(CASE WHEN booking_status = 'Canceled' THEN 1 END) * 100.0 / COUNT(CASE WHEN booking_status IS NOT NULL THEN 1 END)) AS percentage_canceled
FROM mlops_dev.olalubic.test_set
"""


query = w.queries.create(query=sql.CreateQueryRequestQuery(display_name=f'hotel-reservations-alert-query-{time.time_ns()}',
                                                           warehouse_id=srcs[0].warehouse_id,
                                                           description="Alert on the number of predicted hotel reservation cancellations",
                                                           query_text=alert_query))

alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(condition=sql.AlertCondition(operand=sql.AlertConditionOperand(
        column=sql.AlertOperandColumn(name="percentage_canceled")),
            op=sql.AlertOperator.GREATER_THAN,
            threshold=sql.AlertConditionThreshold(
                value=sql.AlertOperandValue(
                    double_value=40))),
            display_name=f'hotel-reservation-cancellations-alert-{time.time_ns()}',
            query_id=query.id
        )
    )



# COMMAND ----------

# cleanup
w.queries.delete(id=query.id)
w.alerts.delete(id=alert.id)
