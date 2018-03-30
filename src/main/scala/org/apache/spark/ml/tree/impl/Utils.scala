package org.apache.spark.ml.tree.impl

import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.model.ClassificationError

object Utils {
  def printTree(rootNode: Node): Unit = {
    println(rootNode.subtreeToString())
  }

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
    require(stats.length == 2, "Stats length must be 2[false predict count, true predict count]")
    val falseCount = stats(0)
    val trueCount = stats(1)
    val correctPredictionCount = if (predict < 1) {
      falseCount
    } else {
      trueCount
    }
    ClassificationError(correctPredictionCount, falseCount + trueCount)
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
      val left = subTreeStats(node.leftChild.asInstanceOf[TransferLearningNode])
      val right = subTreeStats(node.leftChild.asInstanceOf[TransferLearningNode])
      (left._1 + right._1, left._2 + right._2)
    }
  }
}
