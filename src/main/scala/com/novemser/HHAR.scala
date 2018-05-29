package com.novemser

import com.novemser.DSSM.doExperiment
import com.novemser.util.{SparkManager, Timer}
//hello
//胡胡胡胡
//where r u
//taking shower?
//using small old huhu
//love you 4ever
object HHAR {
  private val spark = SparkManager.getSpark

  def test2(): Unit = {
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", true)
      .csv("hdfs://novemser:9000/data/Phones_accelerometer_shuffle_del_100w.csv")
      .withColumnRenamed("gt", "class")
//      .drop("Model")
      .filter("class != 'null'")

    val nexus4 = data.filter("model = 'nexus4'").drop("model")
    val s3Data = data.filter("model = 's3'").drop("model")
    val splusData = data.filter("model = 'samsungold'").drop("model")
    val s3miniData = data.filter("model = 's3mini'").drop("model")
    val src = nexus4
    val tgt = splusData

    val timer = new Timer()
      .initTimer("src")
      .initTimer("transfer")

    val depthList = Array(10, 11, 12, 13, 14, 15)
    depthList.foreach(depth => {
      val (srcErr, transErr) =
        doExperiment(
          src,
          tgt,
          tgt,
          treeType = TreeType.STRUT,
          timer = timer,
          maxDepth = depth + 10,
          numTrees = 50
        )
      println(s"depth=$depth,srcErr=$srcErr,transErr=$transErr")

      val (tgtErr, _) =
        doExperiment(
          tgt,
          tgt,
          tgt,
          treeType = TreeType.SER,
          timer = timer,
          maxDepth = depth,
          srcOnly = true,
          numTrees = 10
        )
      println(s"depth=$depth,tgtErr=$tgtErr")
    })
    timer.printTime()
  }

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
      .csv("hdfs://novemser:9000/data/Phones_accelerometer_shuffle_del_500w.csv")
      .withColumnRenamed("gt", "class")
      //      .drop("model")
      .drop("device")
      .drop("Index")
      .drop("User")
      .filter("class != 'null'")
//      .repartition(24 * 4)

    val nexus4 = data.filter("model = 'nexus4'").drop("model")
    val s3Data = data.filter("model = 's3'").drop("model")
    val splusData = data.filter("model = 'samsungold'").drop("model")
    val s3miniData = data.filter("model = 's3mini'").drop("model")
    val expMap = Map(
      "s3" -> s3Data,
      "s+" -> splusData,
//      "s3mini" -> s3miniData
      "nexus4" -> nexus4
    )
//    nexus4.cache()
//    expMap.values.foreach { _.cache }
    val transferPercentLst = Array(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5)
    val srcPercentList = Array(10, 20, 40, 50, 60, 0.1, 0.2, 1, 5)
    val tgtMap = scala.collection.mutable.HashMap[(String, Double), Double]()
//    val srcPercentList = Array(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)

    println(s"=============================$treeType===========================")
    srcPercentList.foreach { srcPercent =>
      {
        val sourceData = s3miniData.randomSplit(Array(srcPercent, 100 - srcPercent), 1)(0).repartition(24)
        sourceData.persist()
        println(s"=========src percent:$srcPercent=======")
        transferPercentLst.foreach { transferPercent =>
          {
            expMap.foreach(kv => {
              val timer = new Timer()
                .initTimer("src")
                .initTimer("transfer")
              val tgtData = kv._2.repartition(24)
              val tp = transferPercent //* 0.1
//            println(s"Doing experiment:${kv._1}, percent:$tp")
              val Array(td, _) = tgtData.randomSplit(Array(tp, 100 - tp), 1)
              val transferData = td //.repartition(64)
              transferData.persist()
//              println(s"transferData count:${transferData.count()}")
              val maxDep = treeType match {
                case TreeType.SER   => 5
                case TreeType.MIX   => 10
                case TreeType.STRUT => 15
              }
              val (srcErr, transErr) =
                doExperiment(sourceData, transferData, tgtData, treeType = treeType, timer = timer, maxDepth = maxDep)

              val tgtErr = if (tgtMap.contains((kv._1, tp))) {
                tgtMap((kv._1, tp))
              } else {
                tgtMap((kv._1, tp)) = doExperiment(
                  transferData,
                  transferData,
                  tgtData,
                  treeType = treeType,
                  srcOnly = true,
                  timer = timer
                )._1
                tgtMap((kv._1, tp))
              }
              transferData.unpersist()
              println(
                s"Experiment:${kv._1}, tgt percent:$tp whth SrcErr:$srcErr,TgtErr:$tgtErr,TransferErr:$transErr"
              )
              timer.printTime()
            })
          }
        }
        sourceData.unpersist()
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

  // test training time
  def test3(treeType: TreeType.Value): Unit = {
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", true)
      .csv("hdfs://novemser:9000/data/Phones_accelerometer_shuffle_del_500w.csv")
      .withColumnRenamed("gt", "class")
      //      .drop("model")
      .drop("device")
      .drop("Index")
      .drop("User")
      .filter("class != 'null'")
    //      .repartition(24 * 4)

    val nexus4 = data.filter("model = 'nexus4'").drop("model")
    val s3Data = data.filter("model = 's3'").drop("model")
    val splusData = data.filter("model = 'samsungold'").drop("model")
    val s3miniData = data.filter("model = 's3mini'").drop("model")
    val expMap = Map(
      "s3" -> s3Data,
      "s+" -> splusData,
      //      "s3mini" -> s3miniData
      "nexus4" -> nexus4
    )
    //    nexus4.cache()
    //    expMap.values.foreach { _.cache }
    val transferPercentLst = Array(0.5)
    //    val srcPercentList = Array(0.1, 0.2, 1, 5, 10, 20, 40, 50, 60)
    val srcPercentList = Array(0.1)

    println(s"=============================$treeType===========================")
    srcPercentList.foreach { srcPercent =>
      {
        val sourceData = s3miniData.randomSplit(Array(srcPercent, 100 - srcPercent), 1)(0)
        //        sourceData.cache()
        println(s"=========src percent:$srcPercent=======")
        transferPercentLst.foreach { transferPercent =>
          {
            expMap.foreach(kv => {
              val timer = new Timer()
                .initTimer("src")
                .initTimer("transfer")
              val tgtData = kv._2
              val tp = transferPercent //* 0.1
              //            println(s"Doing experiment:${kv._1}, percent:$tp")
              val Array(transferData, _) = tgtData.randomSplit(Array(tp, 100 - tp), 1)
              transferData.cache()
              //              println(s"transferData count:${transferData.count()}")
              val maxDep = treeType match {
                case TreeType.SER   => 5
                case TreeType.MIX   => 15
                case TreeType.STRUT => 20
              }
              val (srcErr, transErr) =
                doExperiment(sourceData, transferData, tgtData, treeType = treeType, timer = timer, maxDepth = maxDep)
//          val (tgtErr, _) =
//            doExperiment(transferData, transferData, tgtData, treeType = treeType, srcOnly = true, timer = timer)
              transferData.unpersist()
              println(
                s"Experiment:${kv._1}, tgt percent:$tp whth SrcErr:$srcErr,TransferErr:$transErr"
              )
              timer.printTime()
            })
          }
        }
        //        sourceData.unpersist()
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
//    println("=============================================STRUT============")
//    test1(
////      Array(
////        //        ("model = 'samsungold'", "model != 'samsungold'"),
////        //        ("model = 's3mini'", "model != 's3mini'"),
////        //        ("model = 'nexus4'", "model != 'nexus4'"),
////        //        ("model = 's3'", "model != 's3'")
////        //        ("class = 'stand'", "class != 'stand'"),
////        ("class = 'stairsdown'", "class = 'stairsup'"),
////        ("class = 'stairsup'", "class = 'stairsdown'")
////        //        ("class = 'walk'", "class != 'walk'")
////      ),
//      TreeType.STRUT
//    )
//    println("=============================================Mix============")
//    test1(TreeType.SER)
    test1(TreeType.STRUT)
    test1(TreeType.MIX)
//    test3(TreeType.SER)
//    test3(TreeType.STRUT)
//    test3(TreeType.MIX)
    //    test2()
  }
}
