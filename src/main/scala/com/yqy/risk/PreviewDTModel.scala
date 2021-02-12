package com.yqy.risk

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

object PreviewDTModel {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local[*]")
      .appName("PreviewDTModel").getOrCreate()

    val model = PipelineModel.load("hdfs://" + args(0) + ":8020/model/DT_model_churn")

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]

    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

  }
}
