package org.apache.spark.ml.tree.impl

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.tree.{LearningNode, Node, TransferLearningNode}
import org.apache.spark.ml.util.Identifiable

class RichDecisionTreeClassificationModel
(override val uid: String,
 override val rootNode: Node,
 override val numFeatures: Int,
 override val numClasses: Int,
 val rootLearningNode: TransferLearningNode
) extends DecisionTreeClassificationModel(uid, rootNode, numFeatures, numClasses) {
  private[ml] def this(rootNode: Node, numFeatures: Int, numClasses: Int, rootLearningNode: TransferLearningNode) =
    this(Identifiable.randomUID("dtc"), rootNode, numFeatures, numClasses, rootLearningNode)
}

