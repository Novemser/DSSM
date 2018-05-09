package org.apache.spark.ml.tree.impl

import com.novemser.util.Timer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.model.ClassificationError
import org.apache.spark.sql.{DataFrame, Dataset, Row}

object Utils {
  private val SMALL = 1.0E-6D
  val log2: Double = Math.log(2.0D)

  def gr(a: Double, b: Double): Boolean = (a - b) > SMALL

  def eq(a: Double, b: Double): Boolean = a - b < SMALL && b - a < SMALL

  def log2(a: Double): Double = Math.log(a) / log2

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

  /**
   * Calculate the error rate of one prediction.
   *
   * @param stats   label data array
   * @param predict prediction
   */
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
      val right = subTreeStats(node.rightChild.get.asInstanceOf[TransferLearningNode])
      (left._1 + right._1, left._2 + right._2)
    }
  }

  def trainAndTest(pipeline: Pipeline,
                   trainData: Dataset[Row],
                   testData: Dataset[Row],
                   withBErr: Boolean = false,
                   timer: Timer,
                   timerName: String): (Double, Double) = {
    val model = timer.time({ pipeline.fit(trainData) }, "Train", timerName)
    // Make predictions.
    val predictions = model.transform(testData)
    // Select example rows to display.
//    predictions.select("prediction", "predictedLabel", "label", "class").show(50, truncate = false)
    val bErr = if (withBErr) {
      berr(predictions.select("prediction", "label"), 2)
    } else {
      0.0d
    }
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
//    println(s"accuracy: $accuracy, berr: $bErr")
    (1 - accuracy, 1 - bErr)
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
