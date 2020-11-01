Основной идеей метода является построение фич и применение Random forest для предсказания оценки. В качестве итоговых характеристик используются:
* IoU
* Precision
* Recall
* F1-score
* False Positive Rate
* False Negative Rate
* Confusion Matrix
* Euclidian Distance Transform

Дополнительно в качестве характеристик использовались латентные векторы, полученные от автоэнкодера.

![](feature_importance.jpg)
![](mean_error.jpg)

*recurrent_autoencoder.ipynb*: содержит код для обучения и построения фич с помощью рекуррентного автоэнкодера. Автоэнкодер обучается восстанавливать данные о сегментации из файла DX_TEST_RESULT_FULL.csv, в качестве feature вектора используется выходное состояние энкодера.

*cnn_autoencoder.ipynb*: содержит аналогичный код для сверточного автоэнкодера. Модель в данном случае работает напрямую с исходными изображениями и масками сегментации, уменьшенными до размерешения 16х16.