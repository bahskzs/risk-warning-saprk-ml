package com.yqy.risk

import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.SparkSession

/**
 * @author bahsk
 * @createTime 2021-01-14 9:01
 * @description
 */
object FPGrowthTrainApp extends App {

  val spark = SparkSession.builder().master("local[*]").appName("FPGrowthApp").getOrCreate()

  import spark.implicits._

  val dataset = spark.createDataset(Seq(
    "1 2 5",
    "1 2 3 5",
    "1 2")
  ).map(t => t.split(" ")).toDF("items")

  val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(0.5).setMinConfidence(0.6)

  val model = fpgrowth.fit(dataset)

  model.freqItemsets.show()
  model.associationRules.show()
  model.transform(dataset).show()

  spark.stop()

}
