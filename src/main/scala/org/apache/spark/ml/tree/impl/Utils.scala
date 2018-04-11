package org.apache.spark.ml.tree.impl

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.model.ClassificationError
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object Utils {
  def predictImpl(rootNode: LearningNode, binnedFeatures: Array[Int], splits: Array[Array[Split]]): LearningNode = {
    if (rootNode.isLeaf || rootNode.split.isEmpty) {
      rootNode
    } else {
      val split = rootNode.split.get
      val featureIndex = split.featureIndex
      val splitLeft = split.shouldGoLeft(binnedFeatures(featureIndex), splits(featureIndex))
      if (rootNode.leftChild.isEmpty) {
        throw new RuntimeException("Unexpected error")
      } else {
        if (splitLeft) {
          predictImpl(rootNode.leftChild.get, binnedFeatures, splits)
        } else {
          predictImpl(rootNode.rightChild.get, binnedFeatures, splits)
        }
      }
    }
  }

  def calcClassificationError(stats: Array[Double], predict: Double): ClassificationError = {
    val predictCount = stats(predict.toInt)
    ClassificationError(predictCount, stats.sum)
  }

  def leafError(node: TransferLearningNode): Double = {
    node.error.errorRate()
  }

  def subTreeError(node: TransferLearningNode): Double = {
    val (errorCount, totalCount) = subTreeStats(node)
    errorCount / totalCount
  }

  private def subTreeStats(node: TransferLearningNode): (Double, Double) = {
    if (node.leftChild.isEmpty) {
      require(node.rightChild.isEmpty)
      (node.error.errorSampleCount(), node.error.totalSampleCount())
    } else {
      val left = subTreeStats(node.leftChild.get.asInstanceOf[TransferLearningNode])
      val right = subTreeStats(node.leftChild.get.asInstanceOf[TransferLearningNode])
      (left._1 + right._1, left._2 + right._2)
    }
  }

  def trainAndTest(pipeline: Pipeline, trainData: Dataset[Row], testData: Dataset[Row], withBErr: Boolean = false): (Double, Double)= {
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
//    predictions.select("prediction", "predictedLabel", "label", "class").show(50, truncate = false)
    val bErr = berr(predictions.select("prediction", "label"), 2)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"accuracy: $accuracy, berr: $bErr")
    (accuracy, bErr)
  }

  def calcConfMtrx(prediction: DataFrame, numClasses: Int): Array[Array[Int]] = {
    val retVal = new Array[Array[Int]](numClasses)
    Range(0, numClasses).foreach(i => retVal(i) = new Array[Int](numClasses))
    prediction.collect().foreach {
      case Row(prediction: Double, label: Double) =>
        retVal(label.toInt)(prediction.toInt) += 1
      case _ =>
    }
    retVal
  }

  def berr(prediction: DataFrame, numClasses: Int): Double = {
    val confMtrx = calcConfMtrx(prediction, numClasses)
    val localAccuracy = new Array[Double](confMtrx.length)
    localAccuracy.indices.foreach(i => {
      var total = 0
      val correct = confMtrx(i)(i)
      confMtrx(i).foreach(total += _)
      localAccuracy(i) = 1.0 * correct / total
    })
    localAccuracy.sum / localAccuracy.length
  }
}
