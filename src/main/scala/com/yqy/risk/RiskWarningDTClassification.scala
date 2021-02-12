package com.yqy.risk

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StructField}
import org.apache.spark.ml.PipelineModel

object RiskWarningDTClassification {
  //private[this] val logger = Logger(this.getClass)

  //val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local[*]")
      .appName("RiskWarningDTClassification").getOrCreate()


    //load train data
    val tmpDF = spark.read.format("csv").load("hdfs://" + args(0) + ":8020/user/hive/warehouse/labor_supervision_warning.db/ads_labor_key_risk_corp_rank_2019_di")

    val tmpRDD = tmpDF.rdd.map(line => line.toString().replaceAll("\\[", "").replaceAll("\\]", "").split(",")).map(i => concat(i))

    // save as libSVM style data
    val timestamps = System.currentTimeMillis()
    val fileName = "risk_warning_source_data_" + timestamps
    //logger.warn("df ......" + System.currentTimeMillis())

    tmpRDD.saveAsTextFile("hdfs://" + args(0) + ":8020/data/" + fileName)

    val trainDF = spark.read.format("libsvm").load("hdfs://" + args(0) + ":8020/data/" + fileName)

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(trainDF)

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).setHandleInvalid("skip").fit(trainDF)

    val Array(trainingData, validationData) = trainDF.randomSplit(Array(0.7, 0.3))

    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setImpurity("entropy").setMaxDepth(7)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    model.save("hdfs://" + args(0) + ":8020/model/DT_model_churn")

    // Make predictions.
    val predictions = model.transform(validationData)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")


    spark.stop()

  }

  //构造libSVM格式数据,第一列为label
  def concat(a: Array[String]): String = {
    var result = a(5) + " "

    for (i <- 8 to a.size.toInt - 1) {
      result = result + i + ":" + a(i)
      if (i < a.size.toInt - 1) {
        result = result + " "
      }

    }
    return result
  }






}
