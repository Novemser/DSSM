package com.novemser

import com.novemser.util.{SparkManager, Timer}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree.impl.Utils
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.mutable

object TreeType extends Enumeration {
  val SER, STRUT, MIX = Value
}

object DSSM {
  private val spark = SparkManager.getSpark

  def simple(): Unit = {
    // Prepare training data from a list of (label, features) tuples.
    val training = spark
      .createDataFrame(
        Seq(
          (1.0, Vectors.dense(0.0, 1.1, 0.1)),
          (0.0, Vectors.dense(2.0, 1.0, -1.0)),
          (0.0, Vectors.dense(2.0, 1.3, 1.0)),
          (1.0, Vectors.dense(0.0, 1.2, -0.5))
        )
      )
      .toDF("label", "features")

    val lr = new LogisticRegression()
    // Print out the parameters, documentation, and any default values.
    //    println(s"LogisticRegression parameters:\n ${lr.explainParams()}\n")
    lr.setMaxIter(10)
      .setRegParam(0.01d)

    val model1 = lr.fit(training)
    println(s"Model 1 was fit using parameters: ${model1.parent.extractParamMap}")

    // We may alternatively specify parameters using a ParamMap,
    // which supports several methods for specifying parameters.
    val paramMap = ParamMap(lr.maxIter -> 20)
      .put(lr.maxIter, 30) // Specify 1 Param. This overwrites the original maxIter.
      .put(lr.regParam -> 0.1, lr.threshold -> 0.55) // Specify multiple Params.

    // Combine ParamMaps
    val paramMap2 = ParamMap(lr.probabilityCol -> "My prob")
    val paramMapCombined = paramMap ++ paramMap2
    val model2 = lr.fit(training, paramMapCombined)

    // Prepare test data.
    val test = spark
      .createDataFrame(
        Seq(
          (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
          (0.0, Vectors.dense(3.0, 2.0, -0.1)),
          (1.0, Vectors.dense(0.0, 2.2, -1.5))
        )
      )
      .toDF("label", "features")

    // Using transformer.transform() to make predictions
    model2
      .transform(test)
      .select("features", "label", "My Prob", "prediction")
      .collect()
      .foreach {
        case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
          println(s"($features, $label) -> prob=$prob, prediction=$prediction")
      }
  }

  def pipeline(): Unit = {
    val training = spark
      .createDataFrame(
        Seq(
          (0L, "a b c d e spark", 1.0),
          (1L, "b d", 0.0),
          (2L, "spark c f g", 1.0),
          (3L, "hadoop mapreduce", 0.0)
        )
      )
      .toDF("id", "text", "label")

    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

    val pipline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    // Fit the pipeline to training docs.
    val model = pipline.fit(training)

    model.write.overwrite().save("/tmp/spark-lr-model")
    val sameModel = PipelineModel.load(
      "/tmp/spark-lr-model"
    )

    // Prepare test documents, which are unlabeled (id, text) tuples.
    val test = spark
      .createDataFrame(
        Seq(
          (4L, "spark i j k"),
          (5L, "l m n"),
          (6L, "spark hadoop spark"),
          (7L, "apache hadoop")
        )
      )
      .toDF("id", "text")

    model
      .transform(test)
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach {
        case Row(id, text, pb, pred) =>
          println(s"($id, $text) --> prob=$pb, prediction=$pred")
      }
  }

  def testWine(treeType: TreeType.Value): Unit = {
    val red = spark.read
      .option("header", "true")
      .option("inferSchema", value = true)
      .csv("/home/novemser/Documents/Code/DSSM/src/main/resources/wine/red.csv")

    val white = spark.read
      .option("header", "true")
      .option("inferSchema", value = true)
      .csv("/home/novemser/Documents/Code/DSSM/src/main/resources/wine/white.csv")

    doCrossValidateExperiment(white, red, treeType = treeType, expName = "WineSTRUT-White-Red")
//    doCrossValidateExperiment(white, red, treeType = TreeType.SER, expName = "WineSER-White-Red")
//    doCrossValidateExperiment(red, white, treeType = TreeType.SER, expName = "WineSER-Red-White")
    doCrossValidateExperiment(red, white, treeType = treeType, expName = "WineSTRUT-Red-White")
//    doCrossValidateExperiment(white, white, treeType = TreeType.STRUT, expName = "WineSTRUT-White-Red")
//    doCrossValidateExperiment(red, red, treeType = TreeType.SER, expName = "WineSER-Red-White")
  }

  def testNumeric(): Unit = {
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", value = true)
      .csv("src/main/resources/simple/numeric.csv")

    doCrossValidateExperiment(data, data, numTrees = 1, treeType = TreeType.STRUT)
  }

  def testMushroom(treeType: TreeType.Value): Unit = {
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", value = true)
      .csv("/home/novemser/Documents/Code/DSSM/src/main/resources/mushroom/mushroom.csv")

    val timerSER = new Timer()
      .initTimer("srcTrain")
      .initTimer("transferTrain")

    val timerSTRUT = new Timer()
      .initTimer("srcTrain")
      .initTimer("transferTrain")

    val shapeE = data.filter("`stalk-shape` = 'e'") // 3516
    val shapeT = data.filter("`stalk-shape` = 't'") // 4608
    val source = shapeE
    val target = shapeT
    println(s"----------------------------------Mushroom shapeE -> shapeT--------------------------------------")
    val expData = MLUtils.kFold(shapeT.rdd, 20, 1)
    val count = expData.length
    val result = expData
      .map { data =>
        {
          val train = spark.createDataFrame(data._1, target.schema)
          val test = spark.createDataFrame(data._2, target.schema)
//          doExperimentMush(source, train, test, treeType = TreeType.SER, timer = timerSER)
          doExperimentMush(source, train, test, treeType = treeType, timer = timerSTRUT)
        }
      }
      .reduce { (l, r) => // average
        (l._1 + r._1, l._2 + r._2)
      }
    println(s"CV src result:${result._1 / count}, transfer result:${result._2 / count}")
//    timerSER.printTime()
    println(s"--------------------------------------------------------------------------------")

//    val indexers = mutable.ArrayBuffer[StringIndexerModel]()
//    data.schema
//      .map(_.name)
//      .filter(_ != "id")
//      .filter(_ != "class")
//      .foreach((name: String) => {
//        val stringIndexer = new StringIndexer()
//          .setInputCol(name)
//          .setHandleInvalid("keep")
//          .setOutputCol(s"indexed_$name")
//          .setStringOrderType("alphabetAsc")
//          .fit(shapeE)
//        indexers += stringIndexer
//      })
//
//    val trainLabelIndexer = new StringIndexer()
//      .setHandleInvalid("skip")
//      .setInputCol("class")
//      .setStringOrderType("alphabetAsc")
//      .setOutputCol("label")
//      .fit(shapeE)
//
//    val transferLabelIndexer = new StringIndexer()
//      .setHandleInvalid("skip")
//      .setInputCol("class")
//      .setStringOrderType("alphabetAsc")
//      .setOutputCol("label")
//      .fit(shapeT)
//
//    val trainAssembler = new VectorAssembler()
//      .setInputCols(indexers.map(_.getOutputCol).toArray)
//      .setOutputCol("features")
//
//    val transferAssembler = trainAssembler
//
//    val rf = new SourceRandomForestClassifier()
//    rf.setFeaturesCol { trainAssembler.getOutputCol }
//      .setLabelCol { trainLabelIndexer.getOutputCol }
//      .setNumTrees(50)
//      .setImpurity("gini")
//
//    val labelConverter = new IndexToString()
//      .setInputCol("prediction")
//      .setOutputCol("predictedLabel")
//      .setLabels(trainLabelIndexer.labels)
//
//    val pipeline = new Pipeline()
//      .setStages(indexers.toArray ++ Array(trainLabelIndexer, trainAssembler, rf, labelConverter))
//    // Train model. This also runs the indexers.
//    val srcAcc = Utils.trainAndTest(pipeline, shapeE, shapeT, withBErr = false, timerSER, "srcTrain")
//
//    val ser = new SERClassifier(rf.model)
//      .setFeaturesCol { trainAssembler.getOutputCol }
//      .setLabelCol { trainLabelIndexer.getOutputCol }
//
//    val transferPipeline = new Pipeline()
//      .setStages(
//        indexers.toArray ++ Array(transferLabelIndexer, transferAssembler, ser, labelConverter)
//      )
//
//    val transferAcc = Utils.trainAndTest(transferPipeline, shapeT, shapeT, withBErr = false, timerSER, "transferTrain")
//    println(s"Mushroom SER :SrcOnly err:$srcAcc, strut err:$transferAcc")
  }

  def testMushroom2(): Unit = {
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", value = true)
      .csv("/home/novemser/Documents/Code/DSSM/src/main/resources/mushroom/mushroom.csv")

    val shapeE = data.filter("`stalk-shape` = 'e'") // 3516
    val shapeT = data.filter("`stalk-shape` = 't'") // 4608

    val indexers = mutable.ArrayBuffer[StringIndexerModel]()
    data.schema
      .map(_.name)
      .filter(_ != "id")
      .filter(_ != "class")
      .foreach((name: String) => {
        val stringIndexer = new StringIndexer()
          .setInputCol(name)
          .setHandleInvalid("keep")
          .setStringOrderType("alphabetAsc")
          .setOutputCol(s"indexed_$name")
          .fit(shapeE)
        indexers += stringIndexer
      })

    val trainLabelIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("class")
      .setStringOrderType("alphabetAsc")
      .setOutputCol("label")
      .fit(shapeE)

    val transferLabelIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("class")
      .setStringOrderType("alphabetAsc")
      .setOutputCol("label")
      .fit(shapeT)

    val trainAssembler = new VectorAssembler()
      .setInputCols(indexers.map(_.getOutputCol).toArray)
      .setOutputCol("features")

    val transferAssembler = trainAssembler

    val rf = new SourceRandomForestClassifier()
    rf.setFeaturesCol { trainAssembler.getOutputCol }
      .setLabelCol { trainLabelIndexer.getOutputCol }
      .setNumTrees(50)
      .setImpurity("entropy")

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(trainLabelIndexer.labels)

    val timer = new Timer()
      .initTimer("srcTrain")
      .initTimer("transferTrain")

    val pipeline = new Pipeline()
      .setStages(indexers.toArray ++ Array(trainLabelIndexer, trainAssembler, rf, labelConverter))
    // Train model. This also runs the indexers.
    val srcAcc = Utils.trainAndTest(pipeline, shapeE, shapeT, withBErr = false, timer, "srcTrain")

    val ser = new STRUTClassifier(rf.model)
      .setFeaturesCol { trainAssembler.getOutputCol }
      .setLabelCol { transferLabelIndexer.getOutputCol }
      .setImpurity { "entropy" }

    val transferPipeline = new Pipeline()
      .setStages(
        indexers.toArray ++ Array(transferLabelIndexer, transferAssembler, ser, labelConverter)
      )

    val transferAcc = Utils.trainAndTest(transferPipeline, shapeT, shapeT, withBErr = false, timer, "transferTrain")
    println(s"Mushroom STRUT:SrcOnly err:$srcAcc, strut err:$transferAcc")
  }

  def testDigits(treeType: TreeType.Value, depth: Int): Unit = {
    val d6 = spark.read
      .option("header", "true")
      .option("inferSchema", value = true)
      .csv("/home/novemser/Documents/Code/DSSM/src/main/resources/digits/optdigits_6.csv")

    val d9 = spark.read
      .option("header", "true")
      .option("inferSchema", value = true)
      .csv("/home/novemser/Documents/Code/DSSM/src/main/resources/digits/optdigits_9.csv")

//    doCrossValidateExperiment(d6, d9, treeType = TreeType.SER, expName = "DigitsSER-6-9")
    doCrossValidateExperiment(d6, d9, treeType = treeType, expName = "DigitsSER-6-9", maxDepth = depth)
//    doCrossValidateExperiment(d9, d6, treeType = TreeType.SER, expName = "DigitsSER-9-6")
    doCrossValidateExperiment(d9, d6, treeType = treeType, expName = "DigitsSER-9-6", maxDepth = depth)
  }

  def testLandMine(treeType: TreeType.Value): Unit = {
    val mine = mutable.ArrayBuffer[DataFrame]()
    val timer = new Timer()
      .initTimer("src")
      .initTimer("transfer")
    Range(1, 30)
      .map(i => s"/home/novemser/Documents/Code/DSSM/src/main/resources/landMine/minefield$i.csv")
      .foreach(path => {
        val data = spark.read
          .option("header", "true")
          .option("inferSchema", value = true)
          .csv(path)
        mine += data
      })

    val source = mine.take(15).reduce { _ union _ }
    val data = mine.takeRight(14)

    val res = Range(0, 14)
      .map { _ =>
        {
          val target = data.remove(0)
          val test = data.reduce { _ union _ }
          val (srcAcc, serAcc) =
            doExperiment(source, target, test, berr = true, treeType = treeType, timer = timer)
          data += target
          (srcAcc, serAcc)
        }
      }
      .reduce { (l, r) =>
        (l._1 + r._1, l._2 + r._2)
      }
    timer.printTime()
    println(s"src err:${res._1 / 14}, $treeType err:${res._2 / 14}")
  }

  def testLetter(treeType: TreeType.Value, depth: Int): Unit = {
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", true)
      .csv("/home/novemser/Documents/Code/DSSM/src/main/resources/letter/letter-recognition.csv")

    val x2barmean = data.groupBy("class").agg("x2bar" -> "mean").collect().sortBy(_.getString(0))

    val filterFunc: Row => Boolean = row => {
      val x2bar = row.getInt(7)
      val mean = x2barmean
        .filter(keyMean => keyMean.getString(0).equalsIgnoreCase(row.getString(16)))
        .head
        .getDouble(1)
      x2bar <= mean
    }

    val x2barGMean = data.filter(r => !filterFunc(r))
    val x2barLEMean = data.filter(filterFunc)

//    doCrossValidateExperiment(x2barLEMean, x2barGMean, expName = "LetterSER-x2bar<=mean-x2bar>mean", treeType = TreeType.SER)
    doCrossValidateExperiment(
      x2barLEMean,
      x2barGMean,
      expName = "LetterSTRUT-x2bar<=mean-x2bar>mean",
      treeType = treeType,
      maxDepth = depth
    )
//    doCrossValidateExperiment(x2barGMean, x2barLEMean, expName = "LetterSER-x2bar>mean-x2bar<=mean", treeType = TreeType.SER)
    doCrossValidateExperiment(
      x2barGMean,
      x2barLEMean,
      expName = "LetterSTRUT-x2bar>mean-x2bar<=mean",
      treeType = treeType,
      maxDepth = depth
    )
  }

  def doCrossValidateExperimentWithTest(source: DataFrame,
                                        target: DataFrame,
                                        test: DataFrame,
                                        berr: Boolean = false,
                                        numTrees: Int = 50,
                                        treeType: TreeType.Value = TreeType.SER,
                                        maxDepth: Int = 10,
                                        expName: String = "",
                                        doTransfer: Boolean = false): (Double, Double) = {
    println(s"----------------------------------$expName--------------------------------------")
    val expData = MLUtils.kFold(target.rdd, 20, 1)
    val timer = new Timer()
      .initTimer("src")
      .initTimer("transfer")
    source.persist()
    target.persist()
    val count = expData.length
    val result = expData
      .map { data =>
        {
          val train = spark.createDataFrame(data._1, target.schema)
          doExperimentLibSVM(source, train, test, berr, numTrees, treeType, maxDepth, timer)
        }
      }
      .reduce { (l, r) => // average
        (l._1 + r._1, l._2 + r._2)
      }
    println(s"CV src result:${result._1 / count}, transfer result:${result._2 / count}")
    timer.printTime()
    println(s"--------------------------------------------------------------------------------")
    result
  }

  def doCrossValidateExperiment(source: DataFrame,
                                target: DataFrame,
                                berr: Boolean = false,
                                numTrees: Int = 50,
                                treeType: TreeType.Value = TreeType.SER,
                                maxDepth: Int = 10,
                                expName: String = "",
                                doTransfer: Boolean = false): (Double, Double) = {
    println(s"----------------------------------$expName--------------------------------------")
    val expData = MLUtils.kFold(target.rdd, 20, 1)
    val timer = new Timer()
      .initTimer("src")
      .initTimer("transfer")
    source.persist()
    target.persist()
    val count = expData.length
    val result = expData
      .map { data =>
        {
          val train = spark.createDataFrame(data._1, target.schema)
          val test = spark.createDataFrame(data._2, target.schema)
          doExperiment(source, train, test, berr, numTrees, treeType, maxDepth, timer)
        }
      }
      .reduce { (l, r) => // average
        (l._1 + r._1, l._2 + r._2)
      }
    println(s"CV src result:${result._1 / count}, transfer result:${result._2 / count}")
    timer.printTime()
    println(s"--------------------------------------------------------------------------------")
    result
  }

  def doExperimentMush(source: DataFrame,
                       target: DataFrame,
                       test: DataFrame,
                       berr: Boolean = false,
                       numTrees: Int = 50,
                       treeType: TreeType.Value = TreeType.SER,
                       maxDepth: Int = 10,
                       timer: Timer = new Timer): (Double, Double) = {
    val indexers = mutable.ArrayBuffer[StringIndexerModel]()
    source.schema
      .map(_.name)
      .filter(_ != "id")
      .filter(_ != "class")
      .foreach((name: String) => {
        val stringIndexer = new StringIndexer()
          .setInputCol(name)
          .setHandleInvalid("keep")
          .setStringOrderType("alphabetAsc")
          .setOutputCol(s"indexed_$name")
          .fit(source)
        indexers += stringIndexer
      })

    val trainLabelIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("class")
      .setOutputCol("label")
      .setStringOrderType("alphabetAsc")
      .fit(source)

    val transferLabelIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("class")
      .setOutputCol("label")
      .setStringOrderType("alphabetAsc")
      .fit(target)

    val trainAssembler = new VectorAssembler()
      .setInputCols(indexers.map(_.getOutputCol).toArray)
      .setOutputCol("features")

    val transferAssembler = trainAssembler

    val rf = new SourceRandomForestClassifier()
    rf.setFeaturesCol { trainAssembler.getOutputCol }
      .setLabelCol { trainLabelIndexer.getOutputCol }
      .setNumTrees(50)

    treeType match {
      case TreeType.SER   => rf.setImpurity("gini")
      case TreeType.STRUT => rf.setImpurity("entropy")
      case TreeType.MIX   => rf.setImpurity("entropy")
    }

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(trainLabelIndexer.labels)

    val timer = new Timer()
      .initTimer("srcTrain")
      .initTimer("transferTrain")

    val pipeline = new Pipeline()
      .setStages(indexers.toArray ++ Array(trainLabelIndexer, trainAssembler, rf, labelConverter))
    // Train model. This also runs the indexers.
    val srcErr = Utils.trainAndTest(pipeline, source, test, withBErr = false, timer, "srcTrain")

    val classifier = treeType match {
      case TreeType.SER   => new SERClassifier(rf.model)
      case TreeType.STRUT => new STRUTClassifier(rf.model)
      case TreeType.MIX   => new MixClassifier(rf.model)
    }

    treeType match {
      case TreeType.SER   => classifier.setImpurity("gini")
      case TreeType.STRUT => classifier.setImpurity("entropy")
      case TreeType.MIX   => classifier.setImpurity("entropy")
    }

    val transferPipeline = new Pipeline()
      .setStages(
        indexers.toArray ++ Array(transferLabelIndexer, transferAssembler, classifier, labelConverter)
      )

    val transferErr = Utils.trainAndTest(transferPipeline, target, test, withBErr = false, timer, "transferTrain")
    println(s"Mushroom :SrcOnly err:$srcErr, $treeType err:$transferErr")
    (srcErr._1, transferErr._1)
  }

  def doExperimentLibSVM(source: DataFrame,
                         target: DataFrame,
                         test: DataFrame,
                         berr: Boolean = false,
                         numTrees: Int = 50,
                         treeType: TreeType.Value = TreeType.SER,
                         maxDepth: Int = 10,
                         timer: Timer = new Timer,
                         srcOnly: Boolean = false,
                         seed: Int = 1): (Double, Double) = {
    val rf = new SourceRandomForestClassifier()
    rf.setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxDepth(maxDepth)
      .setSeed(seed)
      .setNumTrees(numTrees)

    treeType match {
      case TreeType.SER   => rf.setImpurity("entropy")
      case TreeType.STRUT => rf.setImpurity("entropy") //.setMinInfoGain(0.03) // prevent over fitting
      case TreeType.MIX   => rf.setImpurity("entropy")
    }

    // Chain indexers and tree in a Pipeline.
    val trainPipeline = new Pipeline()
      .setStages(Array(rf))

    val srcAcc = Utils.trainAndTest(trainPipeline, source, test, berr, timer, "src")
    if (srcOnly) {
      println(s"Src err:$srcAcc")
      return srcAcc
    }

    val classifier = treeType match {
      case TreeType.SER   => new SERClassifier(rf.model)
      case TreeType.STRUT => new STRUTClassifier(rf.model)
      case TreeType.MIX   => new MixClassifier(rf.model)
    }

    treeType match {
      case TreeType.SER   => classifier.setImpurity("entropy")
      case TreeType.STRUT => classifier.setImpurity("entropy")
      case TreeType.MIX   => classifier.setImpurity("entropy")
    }

    classifier
      .setFeaturesCol { "features" }
      .setLabelCol { "label" }
      .setMaxDepth { maxDepth }
      .setSeed { seed }

    val transferPipeline = new Pipeline()
      .setStages(Array(classifier))

    val transferAcc = Utils.trainAndTest(transferPipeline, target, test, berr, timer, "transfer")
    println(s"SrcOnly err:$srcAcc, $treeType err:$transferAcc")
    // Using b error mentioned in paper
    if (!berr) {
      (srcAcc._1, transferAcc._1)
    } else {
      (srcAcc._2, transferAcc._2)
    }
  }

  def doExperiment(source: DataFrame,
                   target: DataFrame,
                   test: DataFrame,
                   berr: Boolean = false,
                   numTrees: Int = 50,
                   treeType: TreeType.Value = TreeType.SER,
                   maxDepth: Int = 5,
                   timer: Timer = new Timer,
                   srcOnly: Boolean = false,
                   seed: Int = 1): (Double, Double) = {
//    printInfo(source, target, test)

    val trainLabelIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("class")
      .setOutputCol("label")
      .setStringOrderType("alphabetAsc")
      .fit(source)

    val transferLabelIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("class")
      .setOutputCol("label")
      .setStringOrderType("alphabetAsc")
      .fit(target)

    val trainAssembler = new VectorAssembler()
      .setInputCols(source.schema.map(_.name).filter(s => s != "class").toArray)
      .setOutputCol("features")

    val transferAssembler = trainAssembler

    val rf = new SourceRandomForestClassifier()
    rf.setFeaturesCol { trainAssembler.getOutputCol }
      .setLabelCol { trainLabelIndexer.getOutputCol }
      .setMaxDepth(maxDepth)
      .setSeed(seed)
      .setNumTrees(numTrees)

    treeType match {
      case TreeType.SER   => rf.setImpurity("entropy")
      case TreeType.STRUT => rf.setImpurity("entropy").setMinInfoGain(0.03) // prevent over fitting
      case TreeType.MIX   => rf.setImpurity("entropy")
    }

    val trainPipeline = new Pipeline()
      .setStages(Array(trainLabelIndexer, trainAssembler, rf))

    val srcAcc = Utils.trainAndTest(trainPipeline, source, test, berr, timer, "src")
    if (srcOnly) {
      println(s"Src err:$srcAcc")
      return srcAcc
    }

    val classifier = treeType match {
      case TreeType.SER   => new SERClassifier(rf.model)
      case TreeType.STRUT => new STRUTClassifier(rf.model)
      case TreeType.MIX   => new MixClassifier(rf.model)
    }

    treeType match {
      case TreeType.SER   => classifier.setImpurity("entropy")
      case TreeType.STRUT => classifier.setImpurity("entropy")
      case TreeType.MIX   => classifier.setImpurity("entropy")
    }

    classifier
      .setFeaturesCol { trainAssembler.getOutputCol }
      .setLabelCol { trainLabelIndexer.getOutputCol }
      .setMaxDepth { maxDepth }
      .setSeed { seed }

    val transferPipeline = new Pipeline()
      .setStages(Array(transferLabelIndexer, transferAssembler, classifier))

    val transferAcc = Utils.trainAndTest(transferPipeline, target, test, berr, timer, "transfer")
    println(s"SrcOnly err:$srcAcc, $treeType err:$transferAcc")
    // Using b error mentioned in paper
    if (!berr) {
      (srcAcc._1, transferAcc._1)
    } else {
      (srcAcc._2, transferAcc._2)
    }
  }

  def testMIX(): Unit = {
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", true)
      .csv("src/main/resources/letter/letter-recognition.csv")

    val x2barmean = data.groupBy("class").agg("x2bar" -> "mean").collect().sortBy(_.getString(0))

    val filterFunc: Row => Boolean = row => {
      val x2bar = row.getInt(7)
      val mean = x2barmean
        .filter(keyMean => keyMean.getString(0).equalsIgnoreCase(row.getString(16)))
        .head
        .getDouble(1)
      x2bar <= mean
    }

    val x2barGMean = data.filter(r => !filterFunc(r))
    val x2barLEMean = data.filter(filterFunc)
    val timer = new Timer()
    timer
      .initTimer("transfer")
      .initTimer("src")
    doExperiment(
      x2barLEMean,
      x2barGMean,
      x2barGMean,
      treeType = TreeType.MIX,
      maxDepth = 10,
      numTrees = 50,
      timer = timer
    )
    doExperiment(
      x2barGMean,
      x2barLEMean,
      x2barLEMean,
      treeType = TreeType.MIX,
      maxDepth = 10,
      numTrees = 50,
      timer = timer
    )
  }

  def testStrut(): Unit = {
    def testBug(): Unit = {
      val data = spark.read
        .option("header", "true")
        .option("inferSchema", true)
        .csv("src/main/resources/letter/letter-recognition.csv")

      val x2barmean = data.groupBy("class").agg("x2bar" -> "mean").collect().sortBy(_.getString(0))

      val filterFunc: Row => Boolean = row => {
        val x2bar = row.getInt(7)
        val mean = x2barmean
          .filter(keyMean => keyMean.getString(0).equalsIgnoreCase(row.getString(16)))
          .head
          .getDouble(1)
        x2bar <= mean
      }

      val x2barGMean = data.filter(r => !filterFunc(r))
      val x2barLEMean = data.filter(filterFunc)
      val timer = new Timer()
      timer
        .initTimer("transfer")
        .initTimer("src")
      doExperiment(
        x2barLEMean,
        x2barGMean,
        x2barGMean,
        treeType = TreeType.STRUT,
        maxDepth = 10,
        numTrees = 50,
        timer = timer
      )
      doExperiment(
        x2barGMean,
        x2barLEMean,
        x2barLEMean,
        treeType = TreeType.STRUT,
        maxDepth = 10,
        numTrees = 50,
        timer = timer
      )
    }

    def testStrutLetter(): Unit = {
      val data = spark.read
        .option("header", "true")
        .option("inferSchema", true)
        .csv("src/main/resources/letter/letter-recognition.csv")

      val x2barmean = data.groupBy("class").agg("x2bar" -> "mean").collect().sortBy(_.getString(0))

      val filterFunc: Row => Boolean = row => {
        val x2bar = row.getInt(7)
        val mean = x2barmean
          .filter(keyMean => keyMean.getString(0).equalsIgnoreCase(row.getString(16)))
          .head
          .getDouble(1)
        x2bar <= mean
      }

      val x2barGMean = data.filter(r => !filterFunc(r))
      val x2barLEMean = data.filter(filterFunc)

      doCrossValidateExperiment(
        x2barLEMean,
        x2barGMean,
        expName = "LetterSTRUT-x2bar<=mean-x2bar>mean",
        treeType = TreeType.STRUT,
        numTrees = 50
      )
      doCrossValidateExperiment(
        x2barGMean,
        x2barLEMean,
        expName = "LetterSTRUT-x2bar>mean-x2bar<=mean",
        treeType = TreeType.STRUT,
        numTrees = 50
      )
    }

    def testStrutDigits(): Unit = {
      val d6 = spark.read
        .option("header", "true")
        .option("inferSchema", value = true)
        .csv("src/main/resources/digits/optdigits_6.csv")

      val d9 = spark.read
        .option("header", "true")
        .option("inferSchema", value = true)
        .csv("src/main/resources/digits/optdigits_9.csv")

      doCrossValidateExperiment(d9, d6, treeType = TreeType.STRUT, maxDepth = 10, numTrees = 1)
    }

    def testStrutSimple(): Unit = {
      val a = spark.read
        .option("header", "true")
        .option("inferSchema", value = true)
        .csv("src/main/resources/simple/load.csv")
        .drop("id")

      val b = spark.read
        .option("header", "true")
        .option("inferSchema", value = true)
        .csv("src/main/resources/simple/load.csv")
        .drop("id")

//      spark.sparkContext.setLogLevel("INFO")
      val timer = new Timer()
      timer
        .initTimer("transfer")
        .initTimer("src")
      doExperiment(a, b, a, treeType = TreeType.STRUT, maxDepth = 5, numTrees = 1, timer = timer)
    }

//    testStrutSimple()
//    testStrutDigits()
//    testStrutLetter()
    testBug()
  }

  def printInfo(sourceData: DataFrame, targetData: DataFrame, testData: DataFrame): Unit = {
    println(
      s"Source data.count:${sourceData.count()}\n " +
        s"Target data.count:${targetData.count()}\n " +
        s"Test data.count:${testData.count()}"
    )
  }

  def testUsps(treeType: TreeType.Value): Unit = {
    val mnist = spark.read
      .format("libsvm")
      .option("numFeatures", 784)
      .load("/home/novemser/mnist.libsvm")

    val usps = spark.read
      .format("libsvm")
      .option("numFeatures", 784)
      .load("/home/novemser/Documents/Code/DSSM/src/main/resources/usps/usps.libsvm")

    val test = spark.read
      .format("libsvm")
      .option("numFeatures", 784)
      .load("/home/novemser/Documents/Code/DSSM/src/main/resources/usps/usps_test.libsvm")
    doCrossValidateExperimentWithTest(
      mnist,
      usps,
      test,
      expName = "Usps SER",
      treeType = TreeType.SER
    )
    doCrossValidateExperimentWithTest(
      mnist,
      usps,
      test,
      expName = "Usps STRUT",
      treeType = TreeType.STRUT
    )
    doCrossValidateExperimentWithTest(
      mnist,
      usps,
      test,
      expName = "Usps MIX",
      treeType = TreeType.MIX
    )

    // Automatically identify categorical features, and index them.
//    val featureIndexer = new VectorIndexer()
//      .setInputCol("features")
//      .setOutputCol("indexedFeatures")
//      .setMaxCategories(10)
//      .fit(mnist)

//    val Array(trainingData, testData) = mnist.randomSplit(Array(0.7, 0.3))
//    val trainingData = usps
//    val testData = test
//    val dt = new DecisionTreeClassifier()
//      .setLabelCol("label")
//      .setFeaturesCol("features")
//
//    // Chain indexers and tree in a Pipeline.
//    val pipeline = new Pipeline()
//      .setStages(Array(dt))
//
//    val model = pipeline.fit(trainingData)
//
//    // Make predictions.
//    val predictions = model.transform(testData)
//
//    // Select example rows to display.
//    predictions.select("prediction", "label", "features").show(5)
//
//    // Select (prediction, true label) and compute test error.
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("prediction")
//      .setMetricName("accuracy")
//    val accuracy = evaluator.evaluate(predictions)
//    println(s"Test Error = ${(1.0 - accuracy)}")
//
//    val treeModel = model.stages(0).asInstanceOf[DecisionTreeClassificationModel]
//    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
  }

  def testInversion(treeType: TreeType.Value): Unit = {
    val src = spark.read
      .format("libsvm")
      .option("numFeatures", 784)
      .load("/home/novemser/Documents/Code/DSSM/src/main/resources/inversion/inversion_source.libsvm")

    val tgt = spark.read
      .format("libsvm")
      .option("numFeatures", 784)
      .load("/home/novemser/Documents/Code/DSSM/src/main/resources/inversion/inversion_target.libsvm")
//      .randomSplit(Array(0.1, 0.9), 123)(0)

    val test = spark.read
      .format("libsvm")
      .option("numFeatures", 784)
      .load("/home/novemser/Documents/Code/DSSM/src/main/resources/inversion/inversion_test.libsvm")

    val timer = new Timer()
      .initTimer("src")
      .initTimer("transfer")

    doExperimentLibSVM(src, tgt, test, treeType = treeType, timer = timer, maxDepth = 5)
    doExperimentLibSVM(src, tgt, test, treeType = treeType, timer = timer, maxDepth = 10)
    doExperimentLibSVM(src, tgt, test, treeType = treeType, timer = timer, maxDepth = 15)
    //    doExperimentLibSVM(tgt, tgt, test, treeType = treeType, timer = timer, maxDepth = 10, srcOnly = true)
  }

  def testLowRes(treeType: TreeType.Value, depth: Int = 5): Unit = {
    val test = spark.read
      .format("libsvm")
      .option("numFeatures", 784)
      .load("/home/novemser/Documents/Code/DSSM/src/main/resources/low_res_10/lowres_test.libsvm")
    val timer = new Timer()
      .initTimer("src")
      .initTimer("transfer")

    val errList = mutable.ArrayBuffer[Double]()
    for (i <- 0 to 99) {
      val src = spark.read
        .format("libsvm")
        .option("numFeatures", 784)
        .load(s"/home/novemser/Documents/Code/DSSM/src/main/resources/low_res_10/lowres_source_$i.libsvm")

      val tgt = spark.read
        .format("libsvm")
        .option("numFeatures", 784)
        .load(s"/home/novemser/Documents/Code/DSSM/src/main/resources/low_res_10/lowres_target_$i.libsvm")

      val (err, _) = doExperimentLibSVM(src, tgt, test, treeType = treeType, timer = timer, maxDepth = depth)
      errList += err
    }

    timer.printTime()
    println(
      s"""
         |TreeType:$treeType
         |Depth:$depth
         |Each:${errList.mkString(",")}
         |Average:${errList.sum / errList.size}
       """.stripMargin)
  }

  def testOffice(treeType: TreeType.Value): Unit = {
    val domains: Array[String] = Array("amazon", "Caltech10", "dslr", "webcam")
    val dataList = mutable.ArrayBuffer[DataFrame]()
//    val dataMap = mutable.Map[String, DataFrame]()
    Range(0, 4).foreach(i => {
      val data = spark.read
        .option("header", "true")
        .option("inferSchema", value = true)
        .csv(s"src/main/resources/office_caltech/office_caltech_${i + 1}.csv")
      dataList += data
    })

    val timer = new Timer()
      .initTimer("src")
      .initTimer("transfer")

    for (i <- 0 to 1) {
      for (j <- 0 to 3) {
        if (i != j) {
          doCrossValidateExperiment(
            dataList(i),
            dataList(j),
            treeType = treeType,
            berr =  false,
            expName = s"$treeType ${domains(i)}->${domains(j)} depth 5",
            maxDepth = 5
          )
          doCrossValidateExperiment(
            dataList(i),
            dataList(j),
            treeType = treeType,
            berr =  false,
            expName = s"$treeType ${domains(i)}->${domains(j)} depth 10",
            maxDepth = 10
          )
          doCrossValidateExperiment(
            dataList(i),
            dataList(j),
            treeType = treeType,
            berr =  false,
            expName = s"$treeType ${domains(i)}->${domains(j)} depth 15",
            maxDepth = 15
          )
        }

      }
    }
  }


  //  def testHighRes()

  def main(args: Array[String]): Unit = {
//    testOffice(TreeType.SER)
//    testOffice(TreeType.STRUT)
//    testOffice(TreeType.MIX)
//    testUsps(null)
//    testLowRes(TreeType.SER, 3)
    testLowRes(TreeType.STRUT, 15)
//    testInversion(TreeType.SER)
//    testInversion(TreeType.STRUT)
//    testInversion(TreeType.MIX)
//    testMIX()
//    testNumeric()
//    testStrut()
//    testLetter(TreeType.MIX, 5)
//    testLetter(TreeType.MIX, 10)
//    testWine(TreeType.SER)
//    testWine(TreeType.STRUT)
//    testWine(TreeType.MIX)
//    testUsps(TreeType.SER)
//    testDigits(TreeType.SER, 10)
//    testDigits(TreeType.STRUT, 10)
//    testDigits(TreeType.MIX, 10)
//    testLandMine(TreeType.SER)
//    testLandMine(TreeType.STRUT)
//    testLandMine(TreeType.MIX)
//    testMushroom(TreeType.STRUT)
//    testMushroom2()
    //    pipeline()
    //    testDT()
    //    testLoadToDT("/home/novemser/Documents/Code/DSSM/src/main/resources/mushroom/mushroom.csv")
    //    testMyTree("/home/novemser/Documents/Code/DSSM/src/main/resources/mushroom/mushroom.csv")
    //    testMyTree("/home/novemser/Documents/Code/DSSM/src/main/resources/simple/load.csv")
  }
}
