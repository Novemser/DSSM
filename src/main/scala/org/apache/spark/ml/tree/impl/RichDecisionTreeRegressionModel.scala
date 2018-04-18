package org.apache.spark.ml.tree.impl

import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.tree.{LearningNode, Node}
import org.apache.spark.ml.util.Identifiable

class RichDecisionTreeRegressionModel(
  override val uid: String,
  override val rootNode: Node,
  override val numFeatures: Int,
  val rootLearningNode: LearningNode
) extends DecisionTreeRegressionModel(uid, rootNode, numFeatures) {
  private[ml] def this(rootNode: Node, numFeatures: Int, rootLearningNode: LearningNode) =
    this(Identifiable.randomUID("dtr"), rootNode, numFeatures, rootLearningNode)
}
