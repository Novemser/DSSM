package com.novemser

import com.novemser.DSSM.doExperiment
import com.novemser.util.{SparkManager, Timer}

object HHAR {
  private val spark = SparkManager.getSpark

  /**
    * 1. 有很多nexus4手机的data,少量SamsungS+手机数据,预测迁移后的模型在所有三星手机中的预测效果
    *     - SamsungS+迁移用的数量从10-15-20-25-30-35-40-45-50
    *     - SamsungS3         ----
    *     - SamsungS3mini     ----
    * skyline:全部的三星手机
    * hunch:迁移数据越多越准确
    *
    */
  def test1(treeType: TreeType.Value): Unit = {
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", true)
      .csv("hdfs://novemser:9000/data/Phones_accelerometer.csv")
      .withColumnRenamed("gt", "class")
      //      .drop("model")
      .drop("device")
      .drop("Index")
      .drop("User")
      .filter("class != 'null'")
      .repartition(44)

    val nexus4Data = data.filter("model = 'nexus4'")
    val s3Data = data.filter("model = 's3'")
    val splusData = data.filter("model = 'samsungold'")
    val s3miniData = data.filter("model = 's3mini'")
    val expMap = Map(
      "s3" -> s3Data.drop("model"),
      "s+" -> splusData.drop("model"),
      "s3mini" -> s3miniData.drop("model")
    )
    expMap.values.foreach { _.cache }

    Range(10, 50, 5).foreach { transferPercent => {
      expMap.foreach(kv => {
        val timer = new Timer()
          .initTimer("src")
          .initTimer("transfer")
        val tgtData = kv._2
        println(s"Doing experiment:${kv._1}, percent:$transferPercent")
        val Array(transferData, _) = tgtData.randomSplit(Array(transferPercent, 100 - transferPercent))
        val (srcErr, transErr) =
          doExperiment(nexus4Data, transferData, tgtData, treeType = treeType, maxDepth = 20, timer = timer)
        println(s"in exp..srcErr, transErr=${(srcErr, transErr)}")
        timer.printTime()
        timer.reset()
        timer.initTimer("src")
        val (tgtErr, _) =
          doExperiment(tgtData, tgtData, tgtData, treeType = treeType, maxDepth = 20, srcOnly = true, timer = timer)
        timer.printTime()
        println(s"End experiment:${kv._1}, percent:$transferPercent whth SrcErr:$srcErr,TgtErr:$tgtErr,TransferErr:$transErr")
      })
    }
    }

//    import org.apache.spark.sql.functions._
//
//    filters.foreach(filter => {
//      val source = data.filter(filter._1).drop("model")
//        .withColumn("class", when(col("class") === "stairsdown", "stairs"))
//        .withColumn("class", when(col("class") === "stairsup", "stairs"))
//      val target = data.filter(filter._2).drop("model")
//        .withColumn("class", when(col("class") === "stairsdown", "stairs"))
//        .withColumn("class", when(col("class") === "stairsup", "stairs"))
//      println(s"Source(${filter._1}):${source.count()}, Tgt(${filter._2}):${target.count()}")
//      val Array(l, r) = target.randomSplit(Array(0.8, 0.2), 1)
//
//      val timer = new Timer()
//        .initTimer("src")
//        .initTimer("transfer")
//      doExperiment(source, l, r, timer = timer, treeType = treeType)
//      timer.printTime()
//    })
  }

  def main(args: Array[String]): Unit = {
    test1(
//      Array(
//        //        ("model = 'samsungold'", "model != 'samsungold'"),
//        //        ("model = 's3mini'", "model != 's3mini'"),
//        //        ("model = 'nexus4'", "model != 'nexus4'"),
//        //        ("model = 's3'", "model != 's3'")
//        //        ("class = 'stand'", "class != 'stand'"),
//        ("class = 'stairsdown'", "class = 'stairsup'"),
//        ("class = 'stairsup'", "class = 'stairsdown'")
//        //        ("class = 'walk'", "class != 'walk'")
//      ),
      TreeType.SER
    )

  }
}
