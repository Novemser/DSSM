package com.novemser

import org.apache.spark.{SparkConf, SparkContext}

object App {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Test App")
      .setMaster("local[2]")
    val spark = SparkContext.getOrCreate(conf)
    println(spark.jars)
  }
}
