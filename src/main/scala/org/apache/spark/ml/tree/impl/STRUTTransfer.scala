package org.apache.spark.ml.tree.impl

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.RandomForest.NodeIndexInfo
import org.apache.spark.ml.tree.impl.TransferRandomForest._
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.tree.configuration.Strategy
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

//    println("After transferring")
    // Extract new stats info
    topNodes.zipWithIndex.foreach(t => {
      val statsInfoMap = mutable.Map[TransferLearningNode, ImpurityStats]()
      extractNodes(t._1, statsInfoMap,  metadata, invalidateStats = false)

      statsInfoMap.foreach((tuple: (TransferLearningNode, ImpurityStats)) => {
        newNodeStatsInfoMap((t._2, tuple._1.id)) = tuple._2
      })
    })

    // After target data statistics info calculation
    oldNodeStatsInfoMap.keys
      .filter {
        isNodeFeatureContinuousMap.contains // filter all leaf nodes
      }
      .filter {
        isNodeFeatureContinuousMap(_)       // is this node uses continuous feature
      }
      .map { (key: (Int, Int)) => // acquire a tuple3 of all information needed
        (
          oldNodeStatsInfoMap(key),         // node's old stats info on Src data set
          newNodeStatsInfoMap(key),         // node's new stats info on Tgt data set
          key
        )
      }
      .filter { stats => // filter those invalid stats
        stats._1.leftImpurityCalculator != null &&
        stats._2.leftImpurityCalculator != null
      }
      .filter { stats => // filter those nodes need to switch children
        {
          val oldStats = stats._1
          val newStats = stats._2

          val totalCount = newStats.impurityCalculator.count.toDouble
          val leftCount = newStats.leftImpurityCalculator.count.toDouble
          val rightCount = newStats.rightImpurityCalculator.count.toDouble
          logInfo(s"Total:$totalCount, left:$leftCount, right:$rightCount")
          val oldLeft = oldStats.leftImpurityCalculator.stats.clone()
          val oldRight = oldStats.rightImpurityCalculator.stats.clone()
          val newLeft = newStats.leftImpurityCalculator.stats.clone()
          val newRight = newStats.rightImpurityCalculator.stats.clone()
          normalize(oldLeft)
          normalize(oldRight)
          normalize(newLeft)
          normalize(newRight)

          val calculator: (Array[Double], Array[Double]) => Double =
            smile.math.Math.JensenShannonDivergence

          val jsdLL = calculator(oldLeft, newLeft)
          val jsdLR = calculator(oldLeft, newRight)
          val jsdRR = calculator(oldRight, newRight)
          val jsdRL = calculator(oldRight, newLeft)
          val divergenceGain =
            1 - (leftCount / totalCount) * jsdLL - (rightCount / totalCount) * jsdRR
          val invertedDivergenceGain =
            1 - (rightCount / totalCount) * jsdLR - (leftCount / totalCount) * jsdRL
          logInfo(
            s"jsdLL:$jsdLL, jsdLR:$jsdLR, jsdRR:$jsdRR, jsdRL:$jsdRL; DG1:$divergenceGain, DG2:$invertedDivergenceGain"
          )
          // node point to split
          invertedDivergenceGain > divergenceGain
        }
      }
      .map { keys =>  // get nodes to switch
        nodeMap(keys._3)
      }
      .foreach { node => // switch node
        logInfo(s"Switching child node of node${node.id}")
        val tmp = node.rightChild
        node.rightChild = node.leftChild
        node.leftChild = tmp
      }

    // Prune unreachable nodes
//    topNodes foreach { pruneNoDataNode }

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
          //        println(s"path for node:${path.mkString(",")}")
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
        nodeStatsAggregators.zipWithIndex.map(_.swap).iterator
      }
    }

    val nodeNewStatsMap = partitionAggregates
      .reduceByKey((a, b) => a.merge(b))
      .map {
        case (nodeIndex, aggStats) =>
          val featuresForNode = nodeToFeaturesBc.value.flatMap { nodeToFeatures =>
            Some(nodeToFeatures(nodeIndex))
          }

          // find best split for each node
          val (split: Split, stats: ImpurityStats) =
            updateStats(aggStats, splits, featuresForNode, nodes(nodeIndex))
          (nodeIndex, (split, stats))
      }
      .collectAsMap()

    // Update node split info
    nodeNewStatsMap.foreach((tuple: (Int, (Split, ImpurityStats))) => {
      val node = nodes(tuple._1)
      val sp = tuple._2._1
      val stats = tuple._2._2
      if (tuple._2._1 != null &&
        metadata.isContinuous(sp.featureIndex) &&
        node.split.nonEmpty &&
        sp.featureIndex == node.split.get.featureIndex &&
        stats.valid) {
        node.split = Some(sp)
        node.stats = stats
      }
    })
  }

  private def updateStats(binAggregates: DTStatsAggregator,
                          splits: Array[Array[Split]],
                          featuresForNode: Option[Array[Int]],
                          node: LearningNode): (Split, ImpurityStats) = {

    // Calculate InformationGain and ImpurityStats if current node is top node
    val level = LearningNode.indexToLevel(node.id)
    var gainAndImpurityStats: ImpurityStats = if (level == 0) {
      null
    } else {
      node.stats
    }

    val validFeatureSplits =
      Range(0, binAggregates.metadata.numFeaturesPerNode).view
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
      validFeatureSplits.map {
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
          .maxBy { _._2.gain }
      }

    val (bestSplit, bestSplitStats) =
      if (splitsAndImpurityInfo.isEmpty) {
        // If no valid splits for features, then this split is invalid,
        // return invalid information gain stats.  Take any split and continue.
        // Splits is empty, so arbitrarily choose to split on any threshold
        val dummyFeatureIndex = featuresForNode.map(_.head).getOrElse(0)
        val parentImpurityCalculator = binAggregates.getParentImpurityCalculator()
          (
            new ContinuousSplit(dummyFeatureIndex, 0),
            ImpurityStats.getInvalidImpurityStats(parentImpurityCalculator)
          )
      } else {
        splitsAndImpurityInfo.maxBy(_._2.gain)
      }

    (bestSplit, bestSplitStats)
  }

  private def extractNodes(node: TransferLearningNode,
                           nodeStatsInfoMap: mutable.Map[TransferLearningNode, ImpurityStats],
                           metadata: DecisionTreeMetadata,
                           invalidateStats: Boolean = true): Array[TransferLearningNode] = {
    nodeStatsInfoMap(node) = node.stats
    //    println(s"Node:${node.id}, data count:${node.stats.impurityCalculator.count}")
    if (invalidateStats) {
      if (node.split.nonEmpty &&
        metadata.isContinuous(node.split.get.featureIndex)) {
//        node.stats = null
      }
    }
    if (node.leftChild.isEmpty && node.rightChild.isEmpty) {
      Array(node)
    } else {
      Array(node) ++
        extractNodes(
          node.leftChild.get.asInstanceOf[TransferLearningNode],
          nodeStatsInfoMap,
          metadata,
          invalidateStats
        ) ++
        extractNodes(
          node.rightChild.get.asInstanceOf[TransferLearningNode],
          nodeStatsInfoMap,
          metadata,
          invalidateStats
        )
    }
  }
}
