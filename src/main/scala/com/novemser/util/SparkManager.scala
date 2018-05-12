package com.novemser.util

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object SparkManager {
  private val conf = new SparkConf()
    .setAppName("Transfer learning")
    .set("spark.executor.memory", "7g")
    .set("spark.driver.memory", "14g")
//    .setMaster("spark://192.168.1.8:7077")
    .setMaster("local[*]")

  private final val spark = SparkSession
    .builder()
    .config(conf)
    .getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  def getSpark: SparkSession = spark
}
