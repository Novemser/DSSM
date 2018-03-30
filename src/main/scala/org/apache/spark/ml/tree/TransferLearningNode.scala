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
                          ) extends LearningNode(id, leftChild, rightChild, split, isLeaf, stats) {}

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