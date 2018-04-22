package org.apache.spark.ml.tree.impl

import java.util.Collections

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.RandomForest.NodeIndexInfo
import org.apache.spark.ml.tree.impl.TransferRandomForest._
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable
import scala.util.Random

object STRUTTransfer extends ModelTransfer {
  override def transferModels(
    trainedModels: Array[RichDecisionTreeClassificationModel],
    target: RDD[LabeledPoint],
    strategy: Strategy,
    numTrees: Int,
    featureSubsetStrategy: String,
    seed: Long,
    instr: Option[Instrumentation[_]],
    parentUID: Option[String]
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

    // First extract all nodes for threshold tuning
    val topNodes = trainedModels.map(_.rootLearningNode)
    // 记录在source数据上所有节点的stats信息
    // (treeIndex, nodeIndex) -> node stats
    val oldNodeStatsInfoMap = mutable.Map[(Int, Int), ImpurityStats]()
    val newNodeStatsInfoMap = mutable.Map[(Int, Int), ImpurityStats]()
    val nodeMap = mutable.Map[(Int, Int), LearningNode]()
    val isNodeFeatureContinuousMap = mutable.Map[(Int, Int), Boolean]()

    // Extract old stats info before
    topNodes.zipWithIndex.foreach(t => {
      val topNode = t._1
      val topNodeIndex = t._2
      val statsInfoMap = mutable.Map[TransferLearningNode, ImpurityStats]()
      extractNodes(topNode, statsInfoMap, metadata).foreach(leaf => {
        nodeStack.push((topNodeIndex, leaf))
      })

      statsInfoMap.foreach((tuple: (TransferLearningNode, ImpurityStats)) => {
        val node = tuple._1
        val nodeStats = tuple._2
        val nodeSplit = node.split
        oldNodeStatsInfoMap((topNodeIndex, node.id)) = nodeStats
        nodeMap((topNodeIndex, node.id)) = node
        // Note that leaf nodes do not have node split info
        if (nodeSplit.nonEmpty) {
          isNodeFeatureContinuousMap((topNodeIndex, node.id)) = metadata.isContinuous(nodeSplit.get.featureIndex)
        }
      })
    })

    val nodeByLevel = nodeStack.toList.groupBy { (tuple: (Int, LearningNode)) =>
      {
        LearningNode.indexToLevel(tuple._2.id)
      }
    }.toList

    val nodesToTrain = nodeByLevel
      .sortBy { _._1 }
      .map { _._2 }

    timer.stop("init")
    for (nodes <- nodesToTrain) {
      nodeStack.clear()
      logInfo(s"Nodes to transfer:${nodes.map(_._2.id).mkString(",")}")
      nodes.foreach(node => {
        nodeStack.push(node)
      })

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

        timer.start("calculateNewSplitsStats")
        calculateNewSplitsStats(
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
        timer.stop("calculateNewSplitsStats")
      }
    }
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

  private def pruneNoDataNode(node: LearningNode): Unit = {
    val left = node.leftChild
    val right = node.rightChild
    if (left.nonEmpty && right.nonEmpty &&
        node.stats.impurityCalculator.count == 0) {
      // prune to leaf
      logInfo(s"Pruning node ${node.id} to leaf")
      node.isLeaf = true
      node.leftChild = None
      node.rightChild = None
    }
    if (left.nonEmpty) {
      pruneNoDataNode(left.get)
    }
    if (right.nonEmpty) {
      pruneNoDataNode(right.get)
    }
  }

  private def normalize(input: Array[Double]): Unit = {
    val sum = input.sum
    for (elem <- input.zipWithIndex) {
      input(elem._2) = elem._1 / sum
    }
  }

  private def calculateNewSplitsStats(input: RDD[BaggedPoint[TreePoint]],
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

    def nodeBinSeqOp(treeIndex: Int,
                     nodeInfo: NodeIndexInfo,
                     agg: Array[DTStatsAggregator],
                     baggedPoint: BaggedPoint[TreePoint]): Unit = {
      if (nodeInfo != null) {
        val aggNodeIndex = nodeInfo.nodeIndexInGroup
        val featuresForNode = nodeInfo.featureSubset
        val instanceWeight = baggedPoint.subsampleWeights(treeIndex)
        if (metadata.unorderedFeatures.isEmpty) {
          orderedBinSeqOp(agg(aggNodeIndex), baggedPoint.datum, instanceWeight, featuresForNode)
        } else {
          mixedBinSeqOp(
            agg(aggNodeIndex),
            baggedPoint.datum,
            splits,
            metadata.unorderedFeatures,
            instanceWeight,
            featuresForNode
          )
        }
        agg(aggNodeIndex).updateParent(baggedPoint.datum.label, instanceWeight)
      }
    }

    def binSeqOp(agg: Array[DTStatsAggregator], baggedPoint: BaggedPoint[TreePoint]): Array[DTStatsAggregator] = {
      // Iterate over all nodes in this data pass
      treeToNodeToIndexInfo.foreach {
        case (treeIndex, nodeIndexToInfo) =>
          val path = topNodesForGroup(treeIndex)
            .asInstanceOf[TransferLearningNode]
            .predictPath(baggedPoint.datum.binnedFeatures, splits)
//          logWarning(s"path for node:${path.mkString(",")}")
          // We need to add current point info into all nodes in it's prediction path.
          path.foreach(nodeIndex => {
            val nodeIndexInfo = nodeIndexToInfo.getOrElse(nodeIndex, null)
            nodeBinSeqOp(treeIndex, nodeIndexInfo, agg, baggedPoint)
          })
      }
      agg
    }

    def binSeqOpWithNodeIdCache(
      agg: Array[DTStatsAggregator],
      dataPoint: (BaggedPoint[TreePoint], Array[Int])
    ): Array[DTStatsAggregator] = {
      treeToNodeToIndexInfo.foreach {
        case (treeIndex, nodeIndexToInfo) =>
          val baggedPoint = dataPoint._1
          val nodeIdCache = dataPoint._2
          val nodeIndex = nodeIdCache(treeIndex)
          nodeBinSeqOp(treeIndex, nodeIndexToInfo.getOrElse(nodeIndex, null), agg, baggedPoint)
      }

      agg
    }

    def getNodeToFeatures(
      treeToNodeToIndexInfo: Map[Int, Map[Int, NodeIndexInfo]]
    ): Option[Map[Int, Array[Int]]] = {
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
    nodesForGroup.foreach {
      case (treeIndex, nodesForTree) =>
        nodesForTree.foreach { node =>
          nodes(treeToNodeToIndexInfo(treeIndex)(node.id).nodeIndexInGroup) = node
        }
    }

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
        points.foreach(point => {
          binSeqOp(nodeStatsAggregators, point)
        })

//        println(s"NST:${nodeStatsAggregators.mkString(",")}")
        nodeStatsAggregators.zipWithIndex.map(_.swap).iterator
      }
    }

    val nodeNewStatsMap = partitionAggregates
      .reduceByKey((a, b) => a.merge(b))
      .filter {
        case (nodeIndex, _) =>
          val node = nodes(nodeIndex)
          node.rightChild.nonEmpty // filter leaf nodes
      }
      .map {
        case (nodeIndex, aggStats) =>
          val featuresForNode = nodeToFeaturesBc.value.flatMap { nodeToFeatures =>
            Some(nodeToFeatures(nodeIndex))
          }

          // find best split for each node
          val result =
            updateStats(aggStats, splits, featuresForNode, nodes(nodeIndex))
          (nodeIndex, result)
      }
      .filter { _._2.length > 0 }
      .collectAsMap()

//    nodesForGroup.foreach {
//      case (treeIndex, nodesForTree) =>
//        nodesForTree.foreach { node =>
//          val nodeIndex = node.id
//          val nodeInfo = treeToNodeToIndexInfo(treeIndex)(nodeIndex)
//          val aggNodeIndex = nodeInfo.nodeIndexInGroup
//          val newStatsInfoArray = nodeNewStatsMap(aggNodeIndex)
//          nodeN
//        }
//    }

//    println(s"nodeNewStatsMap.size:${nodeNewStatsMap.size}")

    // Update node split info
    nodeNewStatsMap.foreach((tuple: (Int, Array[(Split, ImpurityStats, Double, Double)])) => {
      val node = nodes(tuple._1).asInstanceOf[TransferLearningNode]
      val newSplits = tuple._2.map(_._1)
      val newStats = tuple._2.map(_._2)
      val jsds = tuple._2.map(_._3)
      val invertedJsds = tuple._2.map(_._4)
      var splitIndex = 0
      var best = Math.max(jsds(splitIndex), invertedJsds(splitIndex))
      Range(1, jsds.length - 1).foreach { i =>
        {
          if (newStats(i - 1).gain <= newStats(i).gain
              && newStats(i + 1).gain <= newStats(i).gain) {
            if (jsds(i) > best) {
              best = jsds(i)
              splitIndex = i
            }
            if (invertedJsds(i) > best) {
              best = invertedJsds(i)
              splitIndex = i
            }
          }
        }
      }

      if (Utils.gr(invertedJsds(splitIndex), jsds(splitIndex))) {
        val tmp = node.leftChild
        node.leftChild = node.rightChild
        node.rightChild = tmp
      }
      // Was there any useful split?
//      if (newStats.isEmpty) {
////        logWarning(s"node ${node.id} back to leaf")
//        node.isLeaf = true
//        node.leftChild = None
//        node.rightChild = None
//      } else {
      node.split = Some(newSplits(splitIndex))
      node.stats = newStats(splitIndex)
//      }
    })
  }

  /**
    *
    * @param binAggregates
    * @param splits
    * @param featuresForNode
    * @param node
    * @return Array of split, stats, divergenceGain, invertedDivergenceGain
    */
  private def updateStats(binAggregates: DTStatsAggregator,
                          splits: Array[Array[Split]],
                          featuresForNode: Option[Array[Int]],
                          node: LearningNode): Array[(Split, ImpurityStats, Double, Double)] = {

    // Calculate InformationGain and ImpurityStats if current node is top node
    val level = LearningNode.indexToLevel(node.id)
    var gainAndImpurityStats: ImpurityStats = if (level == 0 &&
      binAggregates.metadata.isContinuous(node.split.get.featureIndex)) {
      null
    } else {
      node.stats
    }

    val oldStats = node.stats

    val validFeatureSplits =
      Range(0, binAggregates.metadata.numFeaturesPerNode)
        .map { featureIndexIdx =>
          featuresForNode
            .map(features => (featureIndexIdx, features(featureIndexIdx)))
            .getOrElse((featureIndexIdx, featureIndexIdx))
        }
        .withFilter {
          case (_, featureIndex) =>
            binAggregates.metadata.numSplits(featureIndex) != 0 &&
              binAggregates.metadata.isContinuous(featureIndex) &&
              node.split.nonEmpty &&
              node.split.get.featureIndex == featureIndex
        }

    // For each (feature, split), calculate the gain, and select the best (feature, split).
    val splitsAndImpurityInfo =
      validFeatureSplits.flatMap {
        case (featureIndexIdx, featureIndex) =>
          val numSplits = binAggregates.metadata.numSplits(featureIndex)
          // Cumulative sum (scanLeft) of bin statistics.
          // Afterwards, binAggregates for a bin is the sum of aggregates for
          // that bin + all preceding bins.
          val nodeFeatureOffset = binAggregates.getFeatureOffset(featureIndexIdx)
          var splitIndex = 0
          while (splitIndex < numSplits) {
            binAggregates.mergeForFeature(nodeFeatureOffset, splitIndex + 1, splitIndex)
            splitIndex += 1
          }

          Range(0, numSplits)
            .map { splitIdx =>
              val leftChildStats =
                binAggregates.getImpurityCalculator(nodeFeatureOffset, splitIdx)
              val rightChildStats =
                binAggregates.getImpurityCalculator(nodeFeatureOffset, numSplits)
              rightChildStats.subtract(leftChildStats)
              gainAndImpurityStats = calculateImpurityStats(
                gainAndImpurityStats,
                leftChildStats,
                rightChildStats,
                binAggregates.metadata
              )
              (splits(featureIndex)(splitIdx), gainAndImpurityStats)
            }
      }

    def calcJSD(calcUnderTest: ImpurityCalculator,
                origCalc: ImpurityCalculator,
                numClasses: Int): Double = {
      var jsp = 0.0d
      Range(0, numClasses).foreach { cls =>
        {
          val o = calcUnderTest.prob(cls)
          val e = origCalc.prob(cls)
          val m = (o + e) / 2
          if (!Utils.eq(m, 0)) {
            var logO = Utils.log2(o / m) * o
            if (java.lang.Double.isNaN(logO)) logO = 0
            var logE = Utils.log2(e / m) * e
            if (java.lang.Double.isNaN(logE)) logE = 0
            jsp += (logO + logE)
          }
        }
      }
      jsp / 2
    }

    def calcPartJSD(partLeft: ImpurityCalculator,
                partRight: ImpurityCalculator,
                origLeft: ImpurityCalculator,
                origRight: ImpurityCalculator,
                numClasses: Int): Double = {
      val total = 1.0d * partLeft.count + partRight.count
      calcJSD(partLeft, origLeft, numClasses) * (partLeft.count / total) +
      calcJSD(partRight, origRight, numClasses) * (partRight.count / total)
    }

    val res = splitsAndImpurityInfo
      .filter(t => {
        t._2.leftImpurityCalculator != null &&
        t._2.rightImpurityCalculator != null
      })
      .map(t => {
        val newStats = t._2
        val meta = binAggregates.metadata
        val numClasses = meta.numClasses
        val oldLeft = oldStats.leftImpurityCalculator
        val oldRight = oldStats.rightImpurityCalculator
        val newLeft = newStats.leftImpurityCalculator
        val newRight = newStats.rightImpurityCalculator
        val numBags = 2
        val jsds = new Array[Double](numBags)
        jsds(0) = calcPartJSD(newLeft, newRight, oldLeft, oldRight, numClasses)
        jsds(1) = calcPartJSD(newRight, newLeft, oldLeft, oldRight, numClasses)
        val divergenceGain = 1 - jsds(0)
        val invertedDivergenceGain = 1 - jsds(1)
//        logWarning(
//          s"DG1:$divergenceGain, DG2:$invertedDivergenceGain"
//        )

        (t._1, t._2, divergenceGain, invertedDivergenceGain)
      })
      .toArray

    // if there's no data point reaching this node, prune the node.
    if (res.isEmpty) {
      logInfo(s"pruning node: ${node.id}")
      node.isLeaf = true
      node.leftChild = None
      node.rightChild = None
    }
    res
  }

  private def extractNodes(node: TransferLearningNode,
                           nodeStatsInfoMap: mutable.Map[TransferLearningNode, ImpurityStats],
                           metadata: DecisionTreeMetadata): Array[TransferLearningNode] = {
    nodeStatsInfoMap(node) = node.stats

    if (node.leftChild.isEmpty && node.rightChild.isEmpty) {
      Array(node)
    } else {
      Array(node) ++
        extractNodes(
          node.leftChild.get.asInstanceOf[TransferLearningNode],
          nodeStatsInfoMap,
          metadata
        ) ++
        extractNodes(
          node.rightChild.get.asInstanceOf[TransferLearningNode],
          nodeStatsInfoMap,
          metadata
        )
    }
  }
}
