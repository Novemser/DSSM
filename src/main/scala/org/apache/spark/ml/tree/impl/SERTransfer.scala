package org.apache.spark.ml.tree.impl

import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.TransferRandomForest.{findSplits, logDebug, logInfo, logWarning}
import org.apache.spark.ml.tree.{LearningNode, TransferLearningNode}
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable
import scala.util.Random

object SERTransfer extends ModelTransfer {
  def transferModels(
    trainedModels: Array[RichDecisionTreeClassificationModel],
    target: RDD[LabeledPoint],
    strategy: Strategy,
    numTrees: Int,
    featureSubsetStrategy: String,
    seed: Long,
    instr: Option[Instrumentation[_]],
    parentUID: Option[String] = None
  ): Array[RichDecisionTreeClassificationModel] = {
    val timer = new TimeTracker()

    timer.start("total")

    timer.start("init")
    val retaggedInput = target.retag(classOf[LabeledPoint])
    val metadata =
      DecisionTreeMetadata.buildMetadata(retaggedInput, strategy, numTrees, featureSubsetStrategy)
    instr match {
      case Some(instrumentation) =>
        instrumentation.logNumFeatures(metadata.numFeatures)
        instrumentation.logNumClasses(metadata.numClasses)
      case None =>
        logInfo("numFeatures: " + metadata.numFeatures)
        logInfo("numClasses: " + metadata.numClasses)
    }

    // Find the splits and the corresponding bins (interval between the splits) using a sample
    // of the input data.
    timer.start("findSplits")
    val splits = findSplits(retaggedInput, metadata, seed)
    timer.stop("findSplits")
    logDebug("numBins: feature: number of bins")
    logDebug(
      Range(0, metadata.numFeatures)
        .map { featureIndex =>
          s"\t$featureIndex\t${metadata.numBins(featureIndex)}"
        }
        .mkString("\n")
    )

    // Bin feature values (TreePoint representation).
    // Cache input RDD for speedup during multiple passes.
    val treeInput = TreePoint.convertToTreeRDD(retaggedInput, splits, metadata)

    val withReplacement = numTrees > 1

    val baggedInput = BaggedPoint
      .convertToBaggedRDD(treeInput, strategy.subsamplingRate, numTrees, withReplacement, seed)
      .persist(StorageLevel.MEMORY_AND_DISK)

    // depth of the decision tree
    val maxDepth = strategy.maxDepth
    require(
      maxDepth <= 30,
      s"DecisionTree currently only supports maxDepth <= 30, but was given maxDepth = $maxDepth."
    )
    // Max memory usage for aggregates
    // TODO: Calculate memory usage more precisely.
    val maxMemoryUsage: Long = strategy.maxMemoryInMB * 1024L * 1024L
    logDebug("max memory usage for aggregates = " + maxMemoryUsage + " bytes.")

    val nodeIdCache = if (strategy.useNodeIdCache) {
      Some(
        NodeIdCache.init(
          data = baggedInput,
          numTrees = numTrees,
          checkpointInterval = strategy.checkpointInterval,
          initVal = 1
        )
      )
    } else {
      None
    }

    val nodeStack = new mutable.ArrayStack[(Int, LearningNode)]
    val rng = new Random()
    rng.setSeed(seed)

    // Allocate and queue root nodes.
    val topNodes = trainedModels.map(_.rootLearningNode)
    val leafToExpand = mutable.ArrayBuffer[TransferLearningNode]()
    topNodes.zipWithIndex.foreach(t => {
      extractLeafNodes(t._1).foreach(leaf => {
        nodeStack.push((t._2, leaf))
        leafToExpand.append(leaf)
      })
    })

    timer.stop("init")
    while (nodeStack.nonEmpty) {
      // Collect some nodes to split, and choose features for each node (if subsampling).
      // Each group of nodes may come from one or multiple trees, and at multiple levels.
      val (nodesForGroup, treeToNodeToIndexInfo) =
        RandomForest.selectNodesToSplit(nodeStack, maxMemoryUsage, metadata, rng)
      //      val indexInfo = treeToNodeToIndexInfo.values.flatMap(_.values).mkString(",")
      //      println(s"indexInfo:$indexInfo")
      // Sanity check (should never occur):
      assert(
        nodesForGroup.nonEmpty,
        s"RandomForest selected empty nodesForGroup.  Error for unknown reason."
      )

      // Only send trees to worker if they contain nodes being split this iteration.
      val topNodesForGroup: Map[Int, LearningNode] =
        nodesForGroup.keys.map(treeIdx => treeIdx -> topNodes(treeIdx)).toMap

      // Choose node splits, and enqueue new nodes as needed.
      timer.start("findBestSplits")
      TransferRandomForest.findBestSplits(
        baggedInput,
        metadata,
        topNodesForGroup,
        nodesForGroup,
        treeToNodeToIndexInfo,
        splits,
        nodeStack,
        timer,
        nodeIdCache
      )
      timer.stop("findBestSplits")
    }

    baggedInput.unpersist()

    timer.stop("total")

    logInfo("Internal timing for SER expansion:")
    logInfo(s"$timer")

    //    topNodes.foreach(pruneNode)
    //    println(topNodes.head.toNode.subtreeToString())
    // Pruning
    leafToExpand.foreach(expandedLeaf => {
      // If the leaf error of current node is smaller than the expanded sub-tree,
      // that means that if we do not expand current node, will get better result
      // in the target dataset.
      if (expandedLeaf.leftChild.nonEmpty) {
        val leafError = Utils.leafError(expandedLeaf)
        val subTreeError = Utils.subTreeError(expandedLeaf)
        //println(s"LeafErr:$leafError, SubTreeError:$subTreeError")
        if (leafError < subTreeError) {
          // Pruning this node
          logWarning(s"Pruning node [${expandedLeaf.id}]")
          expandedLeaf.leftChild = None
          expandedLeaf.rightChild = None
          expandedLeaf.isLeaf = true
        }
      }
    })

    val numFeatures = metadata.numFeatures

    parentUID match {
      case Some(uid) =>
        topNodes.map { rootNode =>
          new RichDecisionTreeClassificationModel(
            uid,
            rootNode.toNode,
            numFeatures,
            strategy.getNumClasses,
            rootNode
          )
        }
      case None =>
        topNodes.map { rootNode =>
          new RichDecisionTreeClassificationModel(
            rootNode.toNode,
            numFeatures,
            strategy.getNumClasses,
            rootNode
          )
        }
    }
  }

  def transfer(trainedModel: RichDecisionTreeClassificationModel,
               target: RDD[LabeledPoint],
               strategy: Strategy,
               numTrees: Int,
               featureSubsetStrategy: String,
               seed: Long,
               instr: Option[Instrumentation[_]],
               parentUID: Option[String] = None): RichDecisionTreeClassificationModel = {
    // 1. Find all leaf nodes
    // 2. Expand those lead nodes
    //    关键是统计信息计算的分布式
    val timer = new TimeTracker()

    timer.start("total")

    timer.start("init")
    val retaggedInput = target.retag(classOf[LabeledPoint])
    val metadata =
      DecisionTreeMetadata.buildMetadata(retaggedInput, strategy, numTrees, featureSubsetStrategy)
    instr match {
      case Some(instrumentation) =>
        instrumentation.logNumFeatures(metadata.numFeatures)
        instrumentation.logNumClasses(metadata.numClasses)
      case None =>
        logInfo("numFeatures: " + metadata.numFeatures)
        logInfo("numClasses: " + metadata.numClasses)
    }

    // Find the splits and the corresponding bins (interval between the splits) using a sample
    // of the input data.
    timer.start("findSplits")
    val splits = findSplits(retaggedInput, metadata, seed)
    timer.stop("findSplits")
    logDebug("numBins: feature: number of bins")
    logDebug(
      Range(0, metadata.numFeatures)
        .map { featureIndex =>
          s"\t$featureIndex\t${metadata.numBins(featureIndex)}"
        }
        .mkString("\n")
    )

    // Bin feature values (TreePoint representation).
    // Cache input RDD for speedup during multiple passes.
    val treeInput = TreePoint.convertToTreeRDD(retaggedInput, splits, metadata)

    val withReplacement = numTrees > 1

    val baggedInput = BaggedPoint
      .convertToBaggedRDD(treeInput, strategy.subsamplingRate, numTrees, withReplacement, seed)
      .persist(StorageLevel.MEMORY_AND_DISK)

    // depth of the decision tree
    val maxDepth = strategy.maxDepth
    require(
      maxDepth <= 30,
      s"DecisionTree currently only supports maxDepth <= 30, but was given maxDepth = $maxDepth."
    )
    // Max memory usage for aggregates
    // TODO: Calculate memory usage more precisely.
    val maxMemoryUsage: Long = strategy.maxMemoryInMB * 1024L * 1024L
    logDebug("max memory usage for aggregates = " + maxMemoryUsage + " bytes.")

    val nodeIdCache = if (strategy.useNodeIdCache) {
      Some(
        NodeIdCache.init(
          data = baggedInput,
          numTrees = numTrees,
          checkpointInterval = strategy.checkpointInterval,
          initVal = 1
        )
      )
    } else {
      None
    }

    val nodeStack = new mutable.ArrayStack[(Int, LearningNode)]
    val rootNode = trainedModel.rootLearningNode

    val rng = new Random()
    rng.setSeed(seed)

    println("trainedModel:\n" + rootNode.toNode.subtreeToString())
    // Add top node to stack
    val topNodes = Array(rootNode)
    // push the leaf node into stack
    val leafToExpand = topNodes.flatMap(extractLeafNodes)
    leafToExpand.foreach(node => nodeStack.push((0, node)))
    timer.stop("init")

    while (nodeStack.nonEmpty) {
      // Collect some nodes to split, and choose features for each node (if subsampling).
      // Each group of nodes may come from one or multiple trees, and at multiple levels.
      val (nodesForGroup, treeToNodeToIndexInfo) =
        RandomForest.selectNodesToSplit(nodeStack, maxMemoryUsage, metadata, rng)
      //      val indexInfo = treeToNodeToIndexInfo.values.flatMap(_.values).mkString(",")
      //      println(s"indexInfo:$indexInfo")
      // Sanity check (should never occur):
      assert(
        nodesForGroup.nonEmpty,
        s"RandomForest selected empty nodesForGroup.  Error for unknown reason."
      )

      // Only send trees to worker if they contain nodes being split this iteration.
      val topNodesForGroup: Map[Int, LearningNode] =
        nodesForGroup.keys.map(treeIdx => treeIdx -> topNodes(treeIdx)).toMap

      // Choose node splits, and enqueue new nodes as needed.
      timer.start("findBestSplits")
      TransferRandomForest.findBestSplits(
        baggedInput,
        metadata,
        topNodesForGroup,
        nodesForGroup,
        treeToNodeToIndexInfo,
        splits,
        nodeStack,
        timer,
        nodeIdCache
      )
      timer.stop("findBestSplits")
    }

    baggedInput.unpersist()

    timer.stop("total")

    logInfo("Internal timing for SER expansion:")
    logInfo(s"$timer")
    println("After trainedModel\n" + trainedModel.rootLearningNode.toNode.subtreeToString())
    // Pruning
    leafToExpand.foreach(expandedLeaf => {
      // If the leaf error of current node is smaller than the expanded sub-tree,
      // that means that if we do not expand current node, will get better result
      // in the target dataset.
      if (expandedLeaf.leftChild.nonEmpty) {
        val leafError = Utils.leafError(expandedLeaf)
        val subTreeError = Utils.subTreeError(expandedLeaf)
        println(s"LeafErr:$leafError, SubTreeError:$subTreeError")
        if (leafError < subTreeError) {
          // Pruning this node
          logWarning(s"Pruning node [${expandedLeaf.id}]")
          expandedLeaf.leftChild = None
          expandedLeaf.rightChild = None
          expandedLeaf.isLeaf = true
        }
      }
    })
    println("Pruned tree:\n" + trainedModel.rootLearningNode.toNode.subtreeToString())
    //    trainedModel
    val numFeatures = metadata.numFeatures

    parentUID match {
      case Some(uid) =>
        new RichDecisionTreeClassificationModel(
          uid,
          rootNode.toNode,
          numFeatures,
          strategy.getNumClasses,
          rootNode
        )
      case None =>
        new RichDecisionTreeClassificationModel(
          rootNode.toNode,
          numFeatures,
          strategy.getNumClasses,
          rootNode
        )
    }
  }

  private def extractLeafNodes(node: TransferLearningNode): Array[TransferLearningNode] = {
    if (node.leftChild.isEmpty && node.rightChild.isEmpty) {
      // Reset node statistic info to null for the following calculation
      node.stats = null
      //      println(s"Capture node [${node.id}], error {${node.error}")
      return Array(node)
    }
    extractLeafNodes(node.leftChild.get.asInstanceOf[TransferLearningNode]) ++
      extractLeafNodes(node.rightChild.get.asInstanceOf[TransferLearningNode])
  }

  private def pruneNode(node: TransferLearningNode): Unit = {
    if (node.leftChild.nonEmpty) {
      pruneNode(node.leftChild.get.asInstanceOf[TransferLearningNode])
      pruneNode(node.rightChild.get.asInstanceOf[TransferLearningNode])
      val leafError = Utils.leafError(node)
      val subTreeError = Utils.subTreeError(node)
      //      println(s"LeafErr:$leafError, SubTreeError:$subTreeError")
      if (leafError < subTreeError) {
        // Pruning this node
        logWarning(s"Pruning node [${node.id}]")
        node.leftChild = None
        node.rightChild = None
      }
    }
  }
}
