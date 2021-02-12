package com.yqy.risk

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession


object Prediction {
  def main(args: Array[String]): Unit = {
    // load pre-predictions data

    val spark = SparkSession.builder().master("local[*]")
      .appName("Prediction").getOrCreate()

    val preDF = spark.read.format("csv").load("hdfs://" + args(0) + ":8020/user/hive/warehouse/labor_supervision_warning.db/ads_labor_key_risk_corp_rank_2020_3_di")

    val newData = preDF.rdd.map(line => line.toString().replaceAll("\\[", "").replaceAll("\\]", "").split(",")).map(i => concat(i))

    val timestamps = System.currentTimeMillis()
    val resultName = "risk_warning_result_data_" + timestamps
    //logger.warn("resultName ..." + resultName)

    newData.saveAsTextFile("hdfs://"+ args(0) + ":8020/data/" + resultName)

    val realData = spark.read.format("libsvm").load("hdfs://" + args(0) + ":8020/data/"+ resultName)
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(realData).setHandleInvalid("skip").handleInvalid
    val realFeatureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(realData).handleInvalid

    val model = PipelineModel.load("hdfs://" + args(0) +":8020/model/DT_model_churn")

    val realPredictions = model.transform(realData)

    //realPredictions.select("predictedLabel", "label", "features").show(5)
    //("src/main/resources/data/risk_result/")
    val df = realPredictions.select("predictedLabel", "label")
    df.write.csv("/data/result_1203")
    //val resultRDD=df.rdd.map(line => (line(0),line(1)))

    //resultRDD.saveAsTextFile("/data/result_"+timestamps)

    spark.stop()
  }
  //构造libSVM格式数据,第一列为label
  def concat(a: Array[String]): String = {
    var result = a(5) + " "
    for (i <- 8 to a.size.toInt - 1) {
      println("a(i) ---"+a(i))
      result = result + i + ":" + a(i)
      if (i < a.size.toInt - 1) {
        result = result + " "
      }

    }
    return result
  }
}