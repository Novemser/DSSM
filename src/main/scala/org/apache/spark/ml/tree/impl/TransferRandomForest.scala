package org.apache.spark.ml.tree.impl

import java.io.IOException

import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.RandomForest._
import org.apache.spark.ml.tree.model.ErrorStats
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo, Strategy => OldStrategy}
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.XORShiftRandom

import scala.collection.mutable
import scala.util.Random

object TransferRandomForest extends Logging {
  /**
    * Train a random forest.
    *
    * @param input Training data: RDD of `LabeledPoint`
    * @return an unweighted set of trees
    */
  def run(input: RDD[LabeledPoint],
          strategy: OldStrategy,
          numTrees: Int,
          featureSubsetStrategy: String,
          seed: Long,
          instr: Option[Instrumentation[_]],
          parentUID: Option[String] = None): Array[DecisionTreeModel] = {

    val timer = new TimeTracker()

    timer.start("total")

    timer.start("init")

    val retaggedInput = input.retag(classOf[LabeledPoint])

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
    logDebug(Range(0, metadata.numFeatures).map { featureIndex =>
      s"\t$featureIndex\t${metadata.numBins(featureIndex)}"
    }.mkString("\n"))

    // Bin feature values (TreePoint representation).
    // Cache input RDD for speedup during multiple passes.
    val treeInput = TreePoint.convertToTreeRDD(retaggedInput, splits, metadata)

    val withReplacement = numTrees > 1

    val baggedInput = BaggedPoint
      .convertToBaggedRDD(treeInput, strategy.subsamplingRate, numTrees, withReplacement, seed)
      .persist(StorageLevel.MEMORY_AND_DISK)

    // depth of the decision tree
    val maxDepth = strategy.maxDepth
    require(maxDepth <= 30,
      s"DecisionTree currently only supports maxDepth <= 30, but was given maxDepth = $maxDepth.")

    // Max memory usage for aggregates
    // TODO: Calculate memory usage more precisely.
    val maxMemoryUsage: Long = strategy.maxMemoryInMB * 1024L * 1024L
    logDebug("max memory usage for aggregates = " + maxMemoryUsage + " bytes.")

    /*
     * The main idea here is to perform group-wise training of the decision tree nodes thus
     * reducing the passes over the data from (# nodes) to (# nodes / maxNumberOfNodesPerGroup).
     * Each data sample is handled by a particular node (or it reaches a leaf and is not used
     * in lower levels).
     */

    // Create an RDD of node Id cache.
    // At first, all the rows belong to the root nodes (node Id == 1).
    val nodeIdCache = if (strategy.useNodeIdCache) {
      Some(NodeIdCache.init(
        data = baggedInput,
        numTrees = numTrees,
        checkpointInterval = strategy.checkpointInterval,
        initVal = 1))
    } else {
      None
    }

    /*
      Stack of nodes to train: (treeIndex, node)
      The reason this is a stack is that we train many trees at once, but we want to focus on
      completing trees, rather than training all simultaneously.  If we are splitting nodes from
      1 tree, then the new nodes to split will be put at the top of this stack, so we will continue
      training the same tree in the next iteration.  This focus allows us to send fewer trees to
      workers on each iteration; see topNodesForGroup below.
     */
    val nodeStack = new mutable.ArrayStack[(Int, LearningNode)]

    val rng = new Random()
    rng.setSeed(seed)

    // Allocate and queue root nodes.
    val topNodes = Array.fill[TransferLearningNode](numTrees)(TransferLearningNode.emptyNode(nodeIndex = 1))
    Range(0, numTrees).foreach(treeIndex => nodeStack.push((treeIndex, topNodes(treeIndex))))

    timer.stop("init")

    while (nodeStack.nonEmpty) {
      // Collect some nodes to split, and choose features for each node (if subsampling).
      // Each group of nodes may come from one or multiple trees, and at multiple levels.
      val (nodesForGroup, treeToNodeToIndexInfo) =
      selectNodesToSplit(nodeStack, maxMemoryUsage, metadata, rng)
      //      val indexInfo = treeToNodeToIndexInfo.values.flatMap(_.values).mkString(",")
      //      println(s"indexInfo:$indexInfo")
      // Sanity check (should never occur):
      assert(nodesForGroup.nonEmpty,
        s"RandomForest selected empty nodesForGroup.  Error for unknown reason.")

      // Only send trees to worker if they contain nodes being split this iteration.
      val topNodesForGroup: Map[Int, LearningNode] =
        nodesForGroup.keys.map(treeIdx => treeIdx -> topNodes(treeIdx)).toMap

      // Choose node splits, and enqueue new nodes as needed.
      timer.start("findBestSplits")
      TransferRandomForest.findBestSplits(baggedInput, metadata, topNodesForGroup, nodesForGroup,
        treeToNodeToIndexInfo, splits, nodeStack, timer, nodeIdCache)
      timer.stop("findBestSplits")
    }

    baggedInput.unpersist()

    timer.stop("total")

    logInfo("Internal timing for DecisionTree:")
    logInfo(s"$timer")

    // Delete any remaining checkpoints used for node Id cache.
    if (nodeIdCache.nonEmpty) {
      try {
        nodeIdCache.get.deleteAllCheckpoints()
      } catch {
        case e: IOException =>
          logWarning(s"delete all checkpoints failed. Error reason: ${e.getMessage}")
      }
    }

    val numFeatures = metadata.numFeatures

    parentUID match {
      case Some(uid) =>
        if (strategy.algo == OldAlgo.Classification) {
          topNodes.map { rootNode =>
            new RichDecisionTreeClassificationModel(uid, rootNode.toNode, numFeatures,
              strategy.getNumClasses, rootNode)
          }
        } else {
          topNodes.map { rootNode =>
            new RichDecisionTreeRegressionModel(uid, rootNode.toNode, numFeatures, rootNode)
          }
        }
      case None =>
        if (strategy.algo == OldAlgo.Classification) {
          topNodes.map { rootNode =>
            new RichDecisionTreeClassificationModel(rootNode.toNode, numFeatures,
              strategy.getNumClasses, rootNode)
          }
        } else {
          topNodes.map(rootNode => new RichDecisionTreeRegressionModel(rootNode.toNode, numFeatures, rootNode))
        }
    }
  }

  def transfer(trainedModel: RichDecisionTreeClassificationModel,
               target: RDD[LabeledPoint],
               strategy: OldStrategy,
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
    logDebug(Range(0, metadata.numFeatures).map { featureIndex =>
      s"\t$featureIndex\t${metadata.numBins(featureIndex)}"
    }.mkString("\n"))

    // Bin feature values (TreePoint representation).
    // Cache input RDD for speedup during multiple passes.
    val treeInput = TreePoint.convertToTreeRDD(retaggedInput, splits, metadata)

    val withReplacement = numTrees > 1

    val baggedInput = BaggedPoint
      .convertToBaggedRDD(treeInput, strategy.subsamplingRate, numTrees, withReplacement, seed)
      .persist(StorageLevel.MEMORY_AND_DISK)

    // depth of the decision tree
    val maxDepth = strategy.maxDepth
    require(maxDepth <= 30,
      s"DecisionTree currently only supports maxDepth <= 30, but was given maxDepth = $maxDepth.")
    // Max memory usage for aggregates
    // TODO: Calculate memory usage more precisely.
    val maxMemoryUsage: Long = strategy.maxMemoryInMB * 1024L * 1024L
    logDebug("max memory usage for aggregates = " + maxMemoryUsage + " bytes.")

    val nodeIdCache = if (strategy.useNodeIdCache) {
      Some(NodeIdCache.init(
        data = baggedInput,
        numTrees = numTrees,
        checkpointInterval = strategy.checkpointInterval,
        initVal = 1))
    } else {
      None
    }

    val nodeStack = new mutable.ArrayStack[(Int, LearningNode)]

    val rng = new Random()
    rng.setSeed(seed)

    println("trainedModel:" + trainedModel.rootLearningNode.toNode.subtreeToString(4))
    // Add top node to stack
    val topNodes = Array(trainedModel.rootLearningNode)
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
      assert(nodesForGroup.nonEmpty,
        s"RandomForest selected empty nodesForGroup.  Error for unknown reason.")

      // Only send trees to worker if they contain nodes being split this iteration.
      val topNodesForGroup: Map[Int, LearningNode] =
        nodesForGroup.keys.map(treeIdx => treeIdx -> topNodes(treeIdx)).toMap

      // Choose node splits, and enqueue new nodes as needed.
      timer.start("findBestSplits")
      TransferRandomForest.findBestSplits(baggedInput, metadata, topNodesForGroup, nodesForGroup,
        treeToNodeToIndexInfo, splits, nodeStack, timer, nodeIdCache)
      timer.stop("findBestSplits")
    }

    baggedInput.unpersist()

    timer.stop("total")

    logInfo("Internal timing for SER expansion:")
    logInfo(s"$timer")
    println("After trainedModel" + trainedModel.rootLearningNode.toNode.subtreeToString(4))
    // Pruning
    leafToExpand.foreach( expandedLeaf => {
      // If the leaf error of current node is smaller than the expanded sub-tree,
      // that means that if we do not expand current node, will get better result
      // in the target dataset.
      if (Utils.leafError(expandedLeaf) < Utils.subTreeError(expandedLeaf)) {
        // Pruning this node
        logWarning(s"Pruning node [${expandedLeaf.id}]")
        expandedLeaf.leftChild = None
        expandedLeaf.rightChild = None
        expandedLeaf.isLeaf = true
      }
    })
    trainedModel
  }

  private def extractLeafNodes(node: TransferLearningNode): Array[TransferLearningNode] = {
    if (node.leftChild.isEmpty && node.rightChild.isEmpty) {
      // Reset node statistic info to null for the following calculation
      node.stats = null
      println(s"Capture node [${node.id}], error {${node.error}")
      return Array(node)
    }
    extractLeafNodes(node.leftChild.get.asInstanceOf[TransferLearningNode]) ++
      extractLeafNodes(node.rightChild.get.asInstanceOf[TransferLearningNode])
  }

  private[tree] def findBestSplits(input: RDD[BaggedPoint[TreePoint]],
                                   metadata: DecisionTreeMetadata,
                                   topNodesForGroup: Map[Int, LearningNode],
                                   nodesForGroup: Map[Int, Array[LearningNode]],
                                   treeToNodeToIndexInfo: Map[Int, Map[Int, NodeIndexInfo]],
                                   splits: Array[Array[Split]],
                                   nodeStack: mutable.ArrayStack[(Int, LearningNode)],
                                   timer: TimeTracker = new TimeTracker,
                                   nodeIdCache: Option[NodeIdCache] = None): Unit = {
    // numNodes:  Number of nodes in this group
    val numNodes = nodesForGroup.values.map(_.length).sum
    logDebug("numNodes = " + numNodes)
    logDebug("numFeatures = " + metadata.numFeatures)
    logDebug("numClasses = " + metadata.numClasses)
    logDebug("isMulticlass = " + metadata.isMulticlass)
    logDebug("isMulticlassWithCategoricalFeatures = " +
      metadata.isMulticlassWithCategoricalFeatures)
    logDebug("using nodeIdCache = " + nodeIdCache.nonEmpty.toString)

    /**
      * Performs a sequential aggregation over a partition for a particular tree and node.
      *
      * For each feature, the aggregate sufficient statistics are updated for the relevant
      * bins.
      *
      * @param treeIndex   Index of the tree that we want to perform aggregation for.
      * @param nodeInfo    The node info for the tree node.
      * @param agg         Array storing aggregate calculation, with a set of sufficient statistics
      *                    for each (node, feature, bin).
      * @param baggedPoint Data point being aggregated.
      */
    def nodeBinSeqOp(treeIndex: Int,
                     nodeInfo: NodeIndexInfo,
                     agg: Array[DTStatsAggregator],
                     baggedPoint: BaggedPoint[TreePoint]): Unit = {
      if (nodeInfo != null) {
        val aggNodeIndex = nodeInfo.nodeIndexInGroup
        val featuresForNode = nodeInfo.featureSubset
        val instanceWeight = baggedPoint.subsampleWeights(treeIndex)
        if (metadata.unorderedFeatures.isEmpty) {
          //          println("orderedBinSeqOp")
          orderedBinSeqOp(agg(aggNodeIndex), baggedPoint.datum, instanceWeight, featuresForNode)
        } else {
          //          println("mixedBinSeqOp")
          mixedBinSeqOp(agg(aggNodeIndex), baggedPoint.datum, splits,
            metadata.unorderedFeatures, instanceWeight, featuresForNode)
        }
        agg(aggNodeIndex).updateParent(baggedPoint.datum.label, instanceWeight)
      }
    }

    /**
      * Performs a sequential aggregation over a partition.
      *
      * Each data point contributes to one node. For each feature,
      * the aggregate sufficient statistics are updated for the relevant bins.
      *
      * @param agg         Array storing aggregate calculation, with a set of sufficient statistics for
      *                    each (node, feature, bin).
      * @param baggedPoint Data point being aggregated.
      * @return agg
      */
    def binSeqOp(agg: Array[DTStatsAggregator],
                 baggedPoint: BaggedPoint[TreePoint]): Array[DTStatsAggregator] = {
      // Iterate over all nodes in this data pass
      treeToNodeToIndexInfo.foreach { case (treeIndex, nodeIndexToInfo) =>
        val nodeIndex =
          topNodesForGroup(treeIndex).predictImpl(baggedPoint.datum.binnedFeatures, splits)
        val nodeIndexInfo = nodeIndexToInfo.getOrElse(nodeIndex, null)
        //        if (nodeIndexInfo == null) {
        //          println("nodeInfo is null....")
        //          println(s"current tree:\n${topNodesForGroup(treeIndex).toNode.subtreeToString(4)}")
        //          println(s"binnedFeatures:${baggedPoint.datum.binnedFeatures.mkString(",")}, label:${baggedPoint.datum.label}")
        //        }
        nodeBinSeqOp(treeIndex, nodeIndexInfo, agg, baggedPoint)
      }
      agg
    }

    /**
      * Do the same thing as binSeqOp, but with nodeIdCache.
      */
    def binSeqOpWithNodeIdCache(agg: Array[DTStatsAggregator],
                                dataPoint: (BaggedPoint[TreePoint], Array[Int])): Array[DTStatsAggregator] = {
      treeToNodeToIndexInfo.foreach { case (treeIndex, nodeIndexToInfo) =>
        val baggedPoint = dataPoint._1
        val nodeIdCache = dataPoint._2
        val nodeIndex = nodeIdCache(treeIndex)
        nodeBinSeqOp(treeIndex, nodeIndexToInfo.getOrElse(nodeIndex, null), agg, baggedPoint)
      }

      agg
    }

    /**
      * Get node index in group --> features indices map,
      * which is a short cut to find feature indices for a node given node index in group.
      */
    def getNodeToFeatures(treeToNodeToIndexInfo: Map[Int, Map[Int, NodeIndexInfo]]): Option[Map[Int, Array[Int]]] = {
      if (!metadata.subsamplingFeatures) {
        None
      } else {
        val mutableNodeToFeatures = new mutable.HashMap[Int, Array[Int]]()
        treeToNodeToIndexInfo.values.foreach { nodeIdToNodeInfo =>
          nodeIdToNodeInfo.values.foreach { nodeIndexInfo =>
            assert(nodeIndexInfo.featureSubset.isDefined)
            mutableNodeToFeatures(nodeIndexInfo.nodeIndexInGroup) = nodeIndexInfo.featureSubset.get
          }
        }
        Some(mutableNodeToFeatures.toMap)
      }
    }

    // array of nodes to train indexed by node index in group
    val nodes = new Array[LearningNode](numNodes)
    nodesForGroup.foreach { case (treeIndex, nodesForTree) =>
      nodesForTree.foreach { node =>
        nodes(treeToNodeToIndexInfo(treeIndex)(node.id).nodeIndexInGroup) = node
      }
    }

    // Calculate best splits for all nodes in the group
    timer.start("chooseSplits")

    // In each partition, iterate all instances and compute aggregate stats for each node,
    // yield a (nodeIndex, nodeAggregateStats) pair for each node.
    // After a `reduceByKey` operation,
    // stats of a node will be shuffled to a particular partition and be combined together,
    // then best splits for nodes are found there.
    // Finally, only best Splits for nodes are collected to driver to construct decision tree.
    val nodeToFeatures = getNodeToFeatures(treeToNodeToIndexInfo)
    val nodeToFeaturesBc = input.sparkContext.broadcast(nodeToFeatures)

    val partitionAggregates: RDD[(Int, DTStatsAggregator)] = if (nodeIdCache.nonEmpty) {
      input.zip(nodeIdCache.get.nodeIdsForInstances).mapPartitions { points =>
        // Construct a nodeStatsAggregators array to hold node aggregate stats,
        // each node will have a nodeStatsAggregator
        val nodeStatsAggregators = Array.tabulate(numNodes) { nodeIndex =>
          val featuresForNode = nodeToFeaturesBc.value.map { nodeToFeatures =>
            nodeToFeatures(nodeIndex)
          }
          new DTStatsAggregator(metadata, featuresForNode)
        }

        // iterator all instances in current partition and update aggregate stats
        points.foreach(binSeqOpWithNodeIdCache(nodeStatsAggregators, _))

        // transform nodeStatsAggregators array to (nodeIndex, nodeAggregateStats) pairs,
        // which can be combined with other partition using `reduceByKey`
        nodeStatsAggregators.view.zipWithIndex.map(_.swap).iterator
      }
    } else {
      input.mapPartitions { points =>
        // Construct a nodeStatsAggregators array to hold node aggregate stats,
        // each node will have a nodeStatsAggregator
        val nodeStatsAggregators = Array.tabulate(numNodes) { nodeIndex =>
          val featuresForNode = nodeToFeaturesBc.value.flatMap { nodeToFeatures =>
            Some(nodeToFeatures(nodeIndex))
          }
          new DTStatsAggregator(metadata, featuresForNode)
        }
        // iterator all instances in current partition and update aggregate stats
        println(s"Evaluating input data to bins")
        // 两次的points数据一模一样啊
        points.foreach(point => {
          //          println(s"Points:${point.datum.binnedFeatures.mkString(",")},Feature:${point.datum.label}")
          binSeqOp(nodeStatsAggregators, point)
        })
        //        println("2333333")

        // transform nodeStatsAggregators array to (nodeIndex, nodeAggregateStats) pairs,
        // which can be combined with other partition using `reduceByKey`
        nodeStatsAggregators.zipWithIndex.map(_.swap).iterator
      }
    }

    val nodeToBestSplits = partitionAggregates.reduceByKey((a, b) => a.merge(b)).map {
      case (nodeIndex, aggStats) =>
        val featuresForNode = nodeToFeaturesBc.value.flatMap { nodeToFeatures =>
          Some(nodeToFeatures(nodeIndex))
        }

        // find best split for each node
        val (split: Split, stats: ImpurityStats, (parentError, leftError, rightError)) =
          binsToBestSplit(aggStats, splits, featuresForNode, nodes(nodeIndex))
        //        println(s"nodeIndex:$nodeIndex, featureIndex:${split.featureIndex}, stats.gain:${stats.gain}")
        (nodeIndex, (split, stats, (parentError, leftError, rightError)))
    }.collectAsMap()

    timer.stop("chooseSplits")

    val nodeIdUpdaters = if (nodeIdCache.nonEmpty) {
      Array.fill[mutable.Map[Int, NodeIndexUpdater]](
        metadata.numTrees)(mutable.Map[Int, NodeIndexUpdater]())
    } else {
      null
    }
    // Iterate over all nodes in this group.
    nodesForGroup.foreach { case (treeIndex, nodesForTree) =>
      nodesForTree.foreach { node =>
        val nodeIndex = node.id
        val nodeInfo = treeToNodeToIndexInfo(treeIndex)(nodeIndex)
        val aggNodeIndex = nodeInfo.nodeIndexInGroup
        val (split: Split, stats: ImpurityStats, (parentError, leftError, rightError)) =
          nodeToBestSplits(aggNodeIndex)
        logDebug("best split = " + split)

        // Extract info for this node.  Create children if not leaf.
        val isLeaf =
          (stats.gain <= 0) || (LearningNode.indexToLevel(nodeIndex) == metadata.maxDepth)
        node.isLeaf = isLeaf
        node.stats = stats
        // set this node's error
        node match {
          case n: TransferLearningNode if n.error == null => n.error = parentError
          case _ =>
        }
        logDebug("Node = " + node)
//        logWarning(s"Node [${node.id}] predict[${node.stats.impurityCalculator.predict}] error stats:$parentError")

        if (!isLeaf) {
          node.split = Some(split)
          val childIsLeaf = (LearningNode.indexToLevel(nodeIndex) + 1) == metadata.maxDepth
          val leftChildIsLeaf = childIsLeaf || (stats.leftImpurity == 0.0)
          val rightChildIsLeaf = childIsLeaf || (stats.rightImpurity == 0.0)
          node.leftChild = Some(TransferLearningNode(LearningNode.leftChildIndex(nodeIndex),
            leftChildIsLeaf, ImpurityStats.getEmptyImpurityStats(stats.leftImpurityCalculator), leftError))
          node.rightChild = Some(TransferLearningNode(LearningNode.rightChildIndex(nodeIndex),
            rightChildIsLeaf, ImpurityStats.getEmptyImpurityStats(stats.rightImpurityCalculator), rightError))

          if (nodeIdCache.nonEmpty) {
            val nodeIndexUpdater = NodeIndexUpdater(
              split = split,
              nodeIndex = nodeIndex)
            nodeIdUpdaters(treeIndex).put(nodeIndex, nodeIndexUpdater)
          }

          // enqueue left child and right child if they are not leaves
          if (!leftChildIsLeaf) {
            nodeStack.push((treeIndex, node.leftChild.get))
          }
          if (!rightChildIsLeaf) {
            nodeStack.push((treeIndex, node.rightChild.get))
          }

          logDebug("leftChildIndex = " + node.leftChild.get.id +
            ", impurity = " + stats.leftImpurity)
          logDebug("rightChildIndex = " + node.rightChild.get.id +
            ", impurity = " + stats.rightImpurity)
        }
      }
    }

    if (nodeIdCache.nonEmpty) {
      // Update the cache if needed.
      nodeIdCache.get.updateNodeIndices(input, nodeIdUpdaters, splits)
    }
  }

  private def binsToBestSplit(binAggregates: DTStatsAggregator,
                              splits: Array[Array[Split]],
                              featuresForNode: Option[Array[Int]],
                              node: LearningNode): (Split, ImpurityStats, (ErrorStats, ErrorStats, ErrorStats)) = {

    // Calculate InformationGain and ImpurityStats if current node is top node
    val level = LearningNode.indexToLevel(node.id)
    var gainAndImpurityStats: ImpurityStats = if (level == 0) {
      null
    } else {
      node.stats
    }

    val validFeatureSplits =
      Range(0, binAggregates.metadata.numFeaturesPerNode).view.map { featureIndexIdx =>
        featuresForNode.map(features => (featureIndexIdx, features(featureIndexIdx)))
          .getOrElse((featureIndexIdx, featureIndexIdx))
      }.withFilter { case (_, featureIndex) =>
        binAggregates.metadata.numSplits(featureIndex) != 0
      }

    // For each (feature, split), calculate the gain, and select the best (feature, split).
    val splitsAndImpurityInfo =
      validFeatureSplits.map { case (featureIndexIdx, featureIndex) =>
        val numSplits = binAggregates.metadata.numSplits(featureIndex)
        if (binAggregates.metadata.isContinuous(featureIndex)) {
          // Cumulative sum (scanLeft) of bin statistics.
          // Afterwards, binAggregates for a bin is the sum of aggregates for
          // that bin + all preceding bins.
          val nodeFeatureOffset = binAggregates.getFeatureOffset(featureIndexIdx)
          var splitIndex = 0
          while (splitIndex < numSplits) {
            binAggregates.mergeForFeature(nodeFeatureOffset, splitIndex + 1, splitIndex)
            splitIndex += 1
          }
          // Find best split.
          val (bestFeatureSplitIndex, bestFeatureGainStats) =
            Range(0, numSplits).map { splitIdx =>
              val leftChildStats = binAggregates.getImpurityCalculator(nodeFeatureOffset, splitIdx)
              val rightChildStats =
                binAggregates.getImpurityCalculator(nodeFeatureOffset, numSplits)
              rightChildStats.subtract(leftChildStats)
              gainAndImpurityStats = calculateImpurityStats(gainAndImpurityStats,
                leftChildStats, rightChildStats, binAggregates.metadata)
              (splitIdx, gainAndImpurityStats)
            }.maxBy(_._2.gain)
          (splits(featureIndex)(bestFeatureSplitIndex), bestFeatureGainStats, null)
        } else if (binAggregates.metadata.isUnordered(featureIndex)) {
          // Unordered categorical feature
          val leftChildOffset = binAggregates.getFeatureOffset(featureIndexIdx)
          val (bestFeatureSplitIndex, bestFeatureGainStats) =
            Range(0, numSplits).map { splitIndex =>
              val leftChildStats = binAggregates.getImpurityCalculator(leftChildOffset, splitIndex)
              val rightChildStats = binAggregates.getParentImpurityCalculator()
                .subtract(leftChildStats)
              gainAndImpurityStats = calculateImpurityStats(gainAndImpurityStats,
                leftChildStats, rightChildStats, binAggregates.metadata)
              (splitIndex, gainAndImpurityStats)
            }.maxBy(_._2.gain)
          (splits(featureIndex)(bestFeatureSplitIndex), bestFeatureGainStats, null)
        } else {
          // Ordered categorical feature
          val nodeFeatureOffset = binAggregates.getFeatureOffset(featureIndexIdx)
          val numCategories = binAggregates.metadata.numBins(featureIndex)

          /* Each bin is one category (feature value).
           * The bins are ordered based on centroidForCategories, and this ordering determines which
           * splits are considered.  (With K categories, we consider K - 1 possible splits.)
           *
           * centroidForCategories is a list: (category, centroid)
           */
          val centroidForCategories = Range(0, numCategories).map { featureValue =>
            val categoryStats =
              binAggregates.getImpurityCalculator(nodeFeatureOffset, featureValue)
            val centroid = if (categoryStats.count != 0) {
              if (binAggregates.metadata.isMulticlass) {
                // multiclass classification
                // For categorical variables in multiclass classification,
                // the bins are ordered by the impurity of their corresponding labels.
                categoryStats.calculate()
              } else if (binAggregates.metadata.isClassification) {
                // binary classification
                // For categorical variables in binary classification,
                // the bins are ordered by the count of class 1.
                categoryStats.stats(1)
              } else {
                // regression
                // For categorical variables in regression and binary classification,
                // the bins are ordered by the prediction.
                categoryStats.predict
              }
            } else {
              Double.MaxValue
            }
            (featureValue, centroid)
          }

          logDebug("Centroids for categorical variable: " + centroidForCategories.mkString(","))

          // bins sorted by centroids
          val categoriesSortedByCentroid = centroidForCategories.toList.sortBy(_._2)

          logDebug("Sorted centroids for categorical variable = " +
            categoriesSortedByCentroid.mkString(","))

          // Cumulative sum (scanLeft) of bin statistics.
          // Afterwards, binAggregates for a bin is the sum of aggregates for
          // that bin + all preceding bins.
          var splitIndex = 0
          //          println(s"mergeForFeature for nodeFeatureOffset:$nodeFeatureOffset, numCategories:$numCategories")
          //          val allStats = binAggregates.getClass.getDeclaredField("allStats")
          //          allStats.setAccessible(true)
          //          val res = allStats.get(binAggregates)
          //          println(s"All stats b4 merge:${res.asInstanceOf[Array[Double]].mkString(",")}")
          while (splitIndex < numSplits) {
            val currentCategory = categoriesSortedByCentroid(splitIndex)._1
            val nextCategory = categoriesSortedByCentroid(splitIndex + 1)._1
            binAggregates.mergeForFeature(nodeFeatureOffset, nextCategory, currentCategory)
            splitIndex += 1
          }
          // lastCategory = index of bin with total aggregates for this (node, feature)
          val lastCategory = categoriesSortedByCentroid.last._1
          // Find best split.
          val (bestFeatureSplitIndex, bestFeatureGainStats, bestSplitError) =
            Range(0, numSplits).map { splitIndex =>
              val featureValue = categoriesSortedByCentroid(splitIndex)._1
              val leftChildStats =
                binAggregates.getImpurityCalculator(nodeFeatureOffset, featureValue)
              val rightChildStats =
                binAggregates.getImpurityCalculator(nodeFeatureOffset, lastCategory)
              rightChildStats.subtract(leftChildStats)
              // For binary classification, we count error using these bin stats we collected
              val lStats = leftChildStats.stats
              val rStats = rightChildStats.stats
              val Array(falseCount, trueCount) = Array
                .tabulate(2) { idx => lStats(idx) + rStats(idx) }
              //              println(s"FalseCount:$falseCount, TrueCount:$trueCount")
              //              println(s"leftChildStats.stats:${lStats.mkString(",")}")
              //              println(s"rightChildStats.stats:${rStats.mkString(",")}")
              gainAndImpurityStats = calculateImpurityStats(gainAndImpurityStats,
                leftChildStats, rightChildStats, binAggregates.metadata)
              val parentError = Utils.calcClassificationError(Array(falseCount, trueCount), gainAndImpurityStats.impurityCalculator.predict)
              val lError = Utils.calcClassificationError(lStats, leftChildStats.predict)
              val rError = Utils.calcClassificationError(rStats, rightChildStats.predict)
//              println(s"Parent [$parentError], l [$lError], r [$rError]")
              (splitIndex, gainAndImpurityStats, (parentError, lError, rError))
            }.maxBy(_._2.gain)
          // 在这算该节点的error吧
          //          val leftCount = categoriesSortedByCentroid(bestFeatureSplitIndex)._2
          //          println(s"Selected Node bestFeatureSplitIndex $bestFeatureSplitIndex " +
          //            s"prediction is:${bestFeatureGainStats.impurityCalculator.predict}")

          val categoriesForSplit =
            categoriesSortedByCentroid.map(_._1.toDouble).slice(0, bestFeatureSplitIndex + 1)
          // 放在categoriesForSplit是走左边 true
          val bestFeatureSplit =
            new CategoricalSplit(featureIndex, categoriesForSplit.toArray, numCategories)
          (bestFeatureSplit, bestFeatureGainStats, bestSplitError)
        }
      }

    val (bestSplit, bestSplitStats, bestSplitError) =
      if (splitsAndImpurityInfo.isEmpty) {
        // If no valid splits for features, then this split is invalid,
        // return invalid information gain stats.  Take any split and continue.
        // Splits is empty, so arbitrarily choose to split on any threshold
        val dummyFeatureIndex = featuresForNode.map(_.head).getOrElse(0)
        val parentImpurityCalculator = binAggregates.getParentImpurityCalculator()
        if (binAggregates.metadata.isContinuous(dummyFeatureIndex)) {
          (new ContinuousSplit(dummyFeatureIndex, 0),
            ImpurityStats.getInvalidImpurityStats(parentImpurityCalculator), null)
        } else {
          val numCategories = binAggregates.metadata.featureArity(dummyFeatureIndex)
          (new CategoricalSplit(dummyFeatureIndex, Array(), numCategories),
            ImpurityStats.getInvalidImpurityStats(parentImpurityCalculator), null)
        }
      } else {
        splitsAndImpurityInfo.maxBy(_._2.gain)
      }
    (bestSplit, bestSplitStats, bestSplitError)
  }

  private def calculateImpurityStats(stats: ImpurityStats,
                                     leftImpurityCalculator: ImpurityCalculator,
                                     rightImpurityCalculator: ImpurityCalculator,
                                     metadata: DecisionTreeMetadata): ImpurityStats = {

    val parentImpurityCalculator: ImpurityCalculator = if (stats == null) {
      leftImpurityCalculator.copy.add(rightImpurityCalculator)
    } else {
      stats.impurityCalculator
    }

    val impurity: Double = if (stats == null) {
      parentImpurityCalculator.calculate()
    } else {
      stats.impurity
    }

    val leftCount = leftImpurityCalculator.count
    val rightCount = rightImpurityCalculator.count

    val totalCount = leftCount + rightCount

    // If left child or right child doesn't satisfy minimum instances per node,
    // then this split is invalid, return invalid information gain stats.
    if ((leftCount < metadata.minInstancesPerNode) ||
      (rightCount < metadata.minInstancesPerNode)) {
      return ImpurityStats.getInvalidImpurityStats(parentImpurityCalculator)
    }

    val leftImpurity = leftImpurityCalculator.calculate() // Note: This equals 0 if count = 0
    val rightImpurity = rightImpurityCalculator.calculate()

    val leftWeight = leftCount / totalCount.toDouble
    val rightWeight = rightCount / totalCount.toDouble

    val gain = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity

    // if information gain doesn't satisfy minimum information gain,
    // then this split is invalid, return invalid information gain stats.
    if (gain < metadata.minInfoGain) {
      return ImpurityStats.getInvalidImpurityStats(parentImpurityCalculator)
    }

    new ImpurityStats(gain, impurity, parentImpurityCalculator,
      leftImpurityCalculator, rightImpurityCalculator)
  }

  /**
    * Returns splits for decision tree calculation.
    * Continuous and categorical features are handled differently.
    *
    * Continuous features:
    * For each feature, there are numBins - 1 possible splits representing the possible binary
    * decisions at each node in the tree.
    * This finds locations (feature values) for splits using a subsample of the data.
    *
    * Categorical features:
    * For each feature, there is 1 bin per split.
    * Splits and bins are handled in 2 ways:
    * (a) "unordered features"
    * For multiclass classification with a low-arity feature
    * (i.e., if isMulticlass && isSpaceSufficientForAllCategoricalSplits),
    * the feature is split based on subsets of categories.
    * (b) "ordered features"
    * For regression and binary classification,
    * and for multiclass classification with a high-arity feature,
    * there is one bin per category.
    *
    * @param input    Training data: RDD of [[LabeledPoint]]
    * @param metadata Learning and dataset metadata
    * @param seed     random seed
    * @return Splits, an Array of [[Split]]
    *         of size (numFeatures, numSplits)
    */
  protected[tree] def findSplits(input: RDD[LabeledPoint],
                                 metadata: DecisionTreeMetadata,
                                 seed: Long): Array[Array[Split]] = {

    logDebug("isMulticlass = " + metadata.isMulticlass)

    val numFeatures = metadata.numFeatures

    // Sample the input only if there are continuous features.
    val continuousFeatures = Range(0, numFeatures).filter(metadata.isContinuous)
    val sampledInput = if (continuousFeatures.nonEmpty) {
      // Calculate the number of samples for approximate quantile calculation.
      val requiredSamples = math.max(metadata.maxBins * metadata.maxBins, 10000)
      val fraction = if (requiredSamples < metadata.numExamples) {
        requiredSamples.toDouble / metadata.numExamples
      } else {
        1.0
      }
      logDebug("fraction of data used for calculating quantiles = " + fraction)
      input.sample(withReplacement = false, fraction, new XORShiftRandom(seed).nextInt())
    } else {
      input.sparkContext.emptyRDD[LabeledPoint]
    }

    findSplitsBySorting(sampledInput, metadata, continuousFeatures)
  }

  private def findSplitsBySorting(input: RDD[LabeledPoint],
                                  metadata: DecisionTreeMetadata,
                                  continuousFeatures: IndexedSeq[Int]): Array[Array[Split]] = {

    val continuousSplits: scala.collection.Map[Int, Array[Split]] = {
      // reduce the parallelism for split computations when there are less
      // continuous features than input partitions. this prevents tasks from
      // being spun up that will definitely do no work.
      val numPartitions = math.min(continuousFeatures.length, input.partitions.length)

      input
        .flatMap(point => continuousFeatures.map(idx => (idx, point.features(idx))))
        .groupByKey(numPartitions)
        .map { case (idx, samples) =>
          val thresholds = findSplitsForContinuousFeature(samples, metadata, idx)
          val splits: Array[Split] = thresholds.map(thresh => new ContinuousSplit(idx, thresh))
          logDebug(s"featureIndex = $idx, numSplits = ${splits.length}")
          (idx, splits)
        }.collectAsMap()
    }

    val numFeatures = metadata.numFeatures
    val splits: Array[Array[Split]] = Array.tabulate(numFeatures) {
      case i if metadata.isContinuous(i) =>
        val split = continuousSplits(i)
        metadata.setNumSplits(i, split.length)
        split

      case i if metadata.isCategorical(i) && metadata.isUnordered(i) =>
        // Unordered features
        // 2^(maxFeatureValue - 1) - 1 combinations
        val featureArity = metadata.featureArity(i)
        Array.tabulate[Split](metadata.numSplits(i)) { splitIndex =>
          val categories = extractMultiClassCategories(splitIndex + 1, featureArity)
          new CategoricalSplit(i, categories.toArray, featureArity)
        }

      case i if metadata.isCategorical(i) =>
        // Ordered features
        //   Splits are constructed as needed during training.
        Array.empty[Split]
    }
    splits
  }

  private def mixedBinSeqOp(agg: DTStatsAggregator,
                            treePoint: TreePoint,
                            splits: Array[Array[Split]],
                            unorderedFeatures: Set[Int],
                            instanceWeight: Double,
                            featuresForNode: Option[Array[Int]]): Unit = {
    val numFeaturesPerNode = if (featuresForNode.nonEmpty) {
      // Use sub-sampled features
      featuresForNode.get.length
    } else {
      // Use all features
      agg.metadata.numFeatures
    }
    // Iterate over features.
    var featureIndexIdx = 0
    while (featureIndexIdx < numFeaturesPerNode) {
      val featureIndex = if (featuresForNode.nonEmpty) {
        featuresForNode.get.apply(featureIndexIdx)
      } else {
        featureIndexIdx
      }
      if (unorderedFeatures.contains(featureIndex)) {
        // Unordered feature
        val featureValue = treePoint.binnedFeatures(featureIndex)
        val leftNodeFeatureOffset = agg.getFeatureOffset(featureIndexIdx)
        // Update the left or right bin for each split.
        val numSplits = agg.metadata.numSplits(featureIndex)
        val featureSplits = splits(featureIndex)
        var splitIndex = 0
        while (splitIndex < numSplits) {
          if (featureSplits(splitIndex).shouldGoLeft(featureValue, featureSplits)) {
            agg.featureUpdate(leftNodeFeatureOffset, splitIndex, treePoint.label, instanceWeight)
          }
          splitIndex += 1
        }
      } else {
        // Ordered feature
        val binIndex = treePoint.binnedFeatures(featureIndex)
        agg.update(featureIndexIdx, binIndex, treePoint.label, instanceWeight)
      }
      featureIndexIdx += 1
    }
  }

  /**
    * Helper for binSeqOp, for regression and for classification with only ordered features.
    *
    * For each feature, the sufficient statistics of one bin are updated.
    *
    * @param agg            Array storing aggregate calculation, with a set of sufficient statistics for
    *                       each (feature, bin).
    * @param treePoint      Data point being aggregated.
    * @param instanceWeight Weight (importance) of instance in dataset.
    */
  private def orderedBinSeqOp(
                               agg: DTStatsAggregator,
                               treePoint: TreePoint,
                               instanceWeight: Double,
                               featuresForNode: Option[Array[Int]]): Unit = {
    val label = treePoint.label

    // Iterate over features.
    if (featuresForNode.nonEmpty) {
      // Use subsampled features
      var featureIndexIdx = 0
      while (featureIndexIdx < featuresForNode.get.length) {
        val binIndex = treePoint.binnedFeatures(featuresForNode.get.apply(featureIndexIdx))
        agg.update(featureIndexIdx, binIndex, label, instanceWeight)
        featureIndexIdx += 1
      }
    } else {
      // Use all features
      val numFeatures = agg.metadata.numFeatures
      var featureIndex = 0
      while (featureIndex < numFeatures) {
        val binIndex = treePoint.binnedFeatures(featureIndex)
        agg.update(featureIndex, binIndex, label, instanceWeight)
        featureIndex += 1
      }
    }
  }

}
