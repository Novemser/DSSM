package com.novemser

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{CustomDecisionTreeClassifier, DecisionTreeClassificationModel, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.mutable

object LearnMLlib {
  private val conf = new SparkConf()
    .setAppName("Test App")
    .setMaster("local[2]")
  private val spark = SparkSession.builder()
    .config(conf)
    .getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  def simple(): Unit = {
    // Prepare training data from a list of (label, features) tuples.
    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")

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
    val test = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")

    // Using transformer.transform() to make predictions
    model2.transform(test)
      .select("features", "label", "My Prob", "prediction")
      .collect()
      .foreach {
        case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
          println(s"($features, $label) -> prob=$prob, prediction=$prediction")
      }
  }

  def pipeline(): Unit = {
    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark c f g", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label")

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
    val test = spark.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "spark hadoop spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")

    model.transform(test)
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach {
        case Row(id, text, pb, pred) =>
          println(s"($id, $text) --> prob=$pb, prediction=$pred")
      }
  }

  def testDT(): Unit = {
    val data = spark
      .read
      .format("libsvm")
      .load("/home/novemser/Documents/Code/spark/data/mllib/sample_libsvm_data.txt")

    //    val training = spark.createDataFrame(Seq(
    //      (0L, "a b c d e spark", 1.0),
    //      (1L, "b d", 0.0),
    //      (2L, "spark c f g", 1.0),
    //      (3L, "hadoop mapreduce", 0.0)
    //    )).toDF("id", "text", "label")
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    // Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
      .fit(data)
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    //    val tokenizer = new Tokenizer()
    //      .setInputCol("text")
    //      .setOutputCol("words")
    //    val hashingTF = new HashingTF()
    //      .setNumFeatures(1000)
    //      .setInputCol(tokenizer.getOutputCol)
    //      .setOutputCol("features")
    val dt = new CustomDecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    //    val pipeline = new Pipeline()
    //      .setStages(Array(tokenizer, hashingTF, dt))
    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))


    //    val model = pipeline.fit(data)
    //    val test = spark.createDataFrame(Seq(
    //      (4L, "spark i j k"),
    //      (5L, "l m n"),
    //      (6L, "spark hadoop spark"),
    //      (7L, "apache hadoop")
    //    )).toDF("id", "text")
    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1.0 - accuracy}")

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

  }

  def testLoadToDT(path: String): Unit = {
    val data = spark
      .read
      .option("header", "true")
      .csv(path)

    val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2))
    //    val trainData = data.filter("`stalk-shape` = 'e'")
    //    val testData = data.filter("`stalk-shape` = 't'")
    trainData.cache()
    testData.cache()
    println(
      s"trainData.count():${trainData.count()}\ntestData.count():${testData.count()}"
    )

    val indexers = mutable.ArrayBuffer[StringIndexerModel]()
    data.schema.map(_.name).filter(_ != "id").filter(_ != "class").foreach((name: String) => {
      val stringIndexer = new StringIndexer()
        .setInputCol(name)
        .setHandleInvalid("keep")
        .setOutputCol(s"indexed_$name")
        .fit(trainData)
      indexers += stringIndexer
    })

    val labelIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("class")
      .setOutputCol("label")
      .fit(trainData)

    val assembler = new VectorAssembler()
      .setInputCols(indexers.map(_.getOutputCol).toArray)
      .setOutputCol("features")

    val rf = new RandomForestClassifier()
      .setNumTrees(50)
      .setFeaturesCol(assembler.getOutputCol)
      .setLabelCol(labelIndexer.getOutputCol)
      .setMaxBins(100)
      .setMaxDepth(9)
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(indexers.toArray ++ Array(labelIndexer, assembler, rf, labelConverter))
    // Train model. This also runs the indexers.
    var s = System.currentTimeMillis()
    val model = pipeline.fit(trainData)
    var e = System.currentTimeMillis()
    println(s"Train time ${e - s}")
    // Make predictions.
    s = System.currentTimeMillis()
    val predictions = model.transform(testData)
    e = System.currentTimeMillis()
    println(s"Predict time ${e - s}")
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5, truncate = false)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelIndexer.getOutputCol)
      .setPredictionCol(labelConverter.getInputCol)
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy: $accuracy")

    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(100, 40, 50))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)
      .setNumFolds(20)
      .setParallelism(4)
    // Run cross-validation, and choose the best set of parameters.
    //    val cvModel = cv.fit(trainData)
    //    val transformed = cvModel.transform(testData)
    //      .select("predictedLabel", "label", "features")
    //      .show(10, truncate = false)
    //    val acc2 = evaluator.evaluate(transformed)
    //    println(s"Acc2: $acc2")
    //    data.show(10, truncate = false)
  }

  def testMyTree(path: String): Unit = {
    val data = spark
      .read
      .option("header", "true")
      .csv(path)

    val Array(trainData, testData) = data.randomSplit(Array(1.0, 0.0))
    //    val trainData = data.filter("`stalk-shape` = 'e'")
    //    val testData = data.filter("`stalk-shape` = 't'")
    trainData.cache()
    testData.cache()
    println(
      s"trainData.count():${trainData.count()}\ntestData.count():${testData.count()}"
    )

    val indexers = mutable.ArrayBuffer[StringIndexerModel]()
    data.schema.map(_.name).filter(_ != "id").filter(_ != "class").foreach((name: String) => {
      val stringIndexer = new StringIndexer()
        .setInputCol(name)
        .setHandleInvalid("skip")
        .setOutputCol(s"indexed_$name")
        .fit(trainData)
      indexers += stringIndexer
    })

    val labelIndexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("label")
      .fit(trainData)

    val assembler = new VectorAssembler()
      .setInputCols(indexers.map(_.getOutputCol).toArray)
      .setOutputCol("features")

    val rf = new CustomDecisionTreeClassifier()
      .setFeaturesCol(assembler.getOutputCol)
      .setLabelCol(labelIndexer.getOutputCol)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(indexers.toArray ++ Array(labelIndexer, assembler, rf, labelConverter))
    // Train model. This also runs the indexers.
    var s = System.currentTimeMillis()
    val model = pipeline.fit(trainData)
    var e = System.currentTimeMillis()
    println(s"Train time ${e - s}")
    // Make predictions.
    s = System.currentTimeMillis()
    val predictions = model.transform(testData)
    e = System.currentTimeMillis()
    println(s"Predict time ${e - s}")
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5, truncate = false)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelIndexer.getOutputCol)
      .setPredictionCol(labelConverter.getInputCol)
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy: $accuracy")
  }

  def main(args: Array[String]): Unit = {
    //    pipeline()
    //    testDT()
    //    testLoadToDT("/home/novemser/Documents/Code/DSSM/src/main/resources/mushroom/mushroom.csv")
    //    testMyTree("/home/novemser/Documents/Code/DSSM/src/main/resources/mushroom/mushroom.csv")
    testMyTree("/home/novemser/Documents/Code/DSSM/src/main/resources/simple/load.csv")
  }
}
