package org.apache.spark.ml.tree

import org.apache.spark.ml.tree.model.ErrorStats
import org.apache.spark.mllib.tree.model.ImpurityStats

class TransferLearningNode(id: Int,
                           leftChild: Option[TransferLearningNode],
                           rightChild: Option[TransferLearningNode],
                           split: Option[Split],
                           isLeaf: Boolean,
                           stats: ImpurityStats,
                           var error: ErrorStats
                          ) extends LearningNode(id, leftChild, rightChild, split, isLeaf, stats) {
  def predictPath(binnedFeatures: Array[Int], splits: Array[Array[Split]]): Array[Int] = {
    // do this explicit cast
    val learningNode = this.asInstanceOf[LearningNode]
    if (learningNode.isLeaf || learningNode.split.isEmpty) {
      Array(this.id)
    } else {
      val split = learningNode.split.get
      val featureIndex = split.featureIndex
      val splitLeft = split.shouldGoLeft(binnedFeatures(featureIndex), splits(featureIndex))
      if (learningNode.leftChild.isEmpty) {
        // Not yet split. Return next layer of nodes to train
        if (splitLeft) {
          Array(LearningNode.leftChildIndex(this.id))
        } else {
          Array(LearningNode.rightChildIndex(this.id))
        }
      } else {
        if (splitLeft) {
          Array(this.id) ++ learningNode.leftChild.get.asInstanceOf[TransferLearningNode]
            .predictPath(binnedFeatures, splits)
        } else {
          Array(this.id) ++ learningNode.rightChild.get.asInstanceOf[TransferLearningNode]
            .predictPath(binnedFeatures, splits)
        }
      }
    }
  }
}

object TransferLearningNode {
  /** Create a node with some of its fields set. */
  def apply(id: Int,
            isLeaf: Boolean,
            stats: ImpurityStats,
            error: ErrorStats): TransferLearningNode = {
    new TransferLearningNode(id, None, None, None, false, stats, error)
  }

  /** Create an empty node with the given node index.  Values must be set later on. */
  def emptyNode(nodeIndex: Int): TransferLearningNode = {
    new TransferLearningNode(nodeIndex, None, None, None, false, null, null)
  }
}