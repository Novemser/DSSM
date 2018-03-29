package org.apache.spark.ml.tree.impl

import org.apache.spark.ml.tree.{InternalNode, LeafNode, Node}

object Utils {
  def printTree(rootNode: Node): Unit = {
    println(rootNode.subtreeToString())
  }
}
