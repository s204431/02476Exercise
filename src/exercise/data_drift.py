import pandas as pd
from sklearn import datasets
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues, TestNumberOfColumns, TestConflictPrediction, TestNumberOfEmptyRows, TestHighlyCorrelatedColumns
reference_data = datasets.load_iris(as_frame=True).frame
reference_data = reference_data.rename(
    columns={
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)': 'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width',
        'target': 'prediction'
    }
)

current_data = pd.read_csv('prediction_database.csv')
current_data = current_data.drop(columns=['time'])

report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html('report.html')

data_test = TestSuite(tests=[TestNumberOfMissingValues(),TestNumberOfColumns(),TestConflictPrediction(),TestNumberOfEmptyRows(),TestHighlyCorrelatedColumns()])
data_test.run(reference_data=reference_data, current_data=current_data)
result = data_test.as_dict()
print(result)
print("All tests passed: ", result['summary']['all_passed'])