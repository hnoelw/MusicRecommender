from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.recommendation import ALS
import pyspark.pandas as pd
from pyspark.pandas import DataFrame
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import *
from pyspark.sql import Window


spark = SparkSession.builder.appName("SongRecommender").getOrCreate()

#reading csv files
training = spark.read.options(header=True, inferSchema=True).csv("kkbox-music-recommendation-challenge/train.csv")
test = spark.read.options(header=True, inferSchema=True).csv("kkbox-music-recommendation-challenge/test.csv")
song = spark.read.options(header=True, inferSchema=True).csv("kkbox-music-recommendation-challenge/songs.csv")
song_extra = spark.read.options(header=True, inferSchema=True).csv("kkbox-music-recommendation-challenge/song_extra_info.csv")
members = spark.read.options(header=True, inferSchema=True).csv("kkbox-music-recommendation-challenge/members.csv")

w = Window.orderBy(lit('A'))

temp = members.withColumn('mem_id', row_number().over(w))


songs = song_extra.join(song, ["song_id"])
wi = Window.orderBy(lit('A'))
songTemp = songs.withColumn('s_id', row_number().over(wi))
print("songs")
songTemp.show()
mediumSongs = songTemp.join(training, ["song_id"])
print("medium")
mediumSongs.show()
BigSongs = mediumSongs.join(temp, ["msno"])
print("Big")
BigSongs.show()
BigSongs.na.drop(how="any")
last_row = songTemp.tail(5)
print(last_row)

(training_data, test_data) = BigSongs.randomSplit([0.7, 0.3], seed=42)

als= ALS(rank=10, seed=0, maxIter=5, regParam=0.1, ratingCol='target', userCol='mem_id', itemCol='s_id')
evaluator = RegressionEvaluator(metricName = "rmse", predictionCol = "prediction")

model = als.fit(training_data)
pred_rec = model.transform(test_data)
#works to here

rmse = evaluator.evaluate(pred_rec)
print("Root mean square error = " + str(rmse))


"""
als = ALS(maxIter=20, userCol="userId", itemCol="movieId", rank=5, ratingCol="rating",coldStartStrategy="drop", alpha=1.2, seed=42)

pipeline = Pipeline(stages=[als])

paramGrid = ParamGridBuilder().addGrid(als.maxIter, [5, 10, 20]).addGrid(als.rank, [5, 10, 15]).addGrid(als.alpha, [0.8, 1, 1.2]).build()

evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")

crossVal = CrossValidator(estimator = pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

cvModel = crossVal.fit(training)

predictions = cvModel.transform(test)

rmse = evaluator.evaluate(predictions)
print("Root mean square error = " + str(rmse))"""
