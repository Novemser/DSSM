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
    topNodes.zipWithIndex.foreach(t => {
      val topNode = t._1
      val topNodeIndex = t._2
      nodeStack.push((topNodeIndex, topNode))
    })
    timer.stop("init")

    while (nodeStack.nonEmpty) {
      // Collect some nodes to split, and choose features for each node (if subsampling).
      // Each group of nodes may come from one or multiple trees, and at multiple levels.
      val (nodesForGroup, treeToNodeToIndexInfo) =
        RandomForest.selectNodesToSplit(nodeStack, maxMemoryUsage, metadata, rng)
      // Sanity check (should never occur):
      assert(
        nodesForGroup.nonEmpty,
        s"RandomForest selected empty nodesForGroup.  Error for unknown reason."
      )

      // Only send trees to worker if they contain nodes being split this iteration.
      val topNodesForGroup: Map[Int, LearningNode] =
        nodesForGroup.keys.map(treeIdx => treeIdx -> topNodes(treeIdx)).toMap

      timer.start("calculateNewSplitsStats")
      STRUTTransfer
        .calculateNewSplitsStats(
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
      nodesForGroup.foreach(nodeMeta => {
        val treeIndex = nodeMeta._1
        val nodes = nodeMeta._2
        nodes.foreach(node => {
          if (node.leftChild.nonEmpty)
            nodeStack.push((treeIndex, node.leftChild.get))
          if (node.rightChild.nonEmpty)
            nodeStack.push((treeIndex, node.rightChild.get))
        })
      })
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
          // We need to add current point info into all nodes in it's prediction path.
          path.foreach(nodeIndex => {
            val nodeIndexInfo = nodeIndexToInfo.getOrElse(nodeIndex, null)
            nodeBinSeqOp(treeIndex, nodeIndexInfo, agg, baggedPoint)
          })
      }
      agg
    }

    // TODO using node id cache
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

    val partitionAggregates: RDD[(Int, DTStatsAggregator)] =
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

    val nodeNewStatsMap = partitionAggregates
      .reduceByKey((a, b) => a.merge(b))
      .map {
        case (nodeIndex, aggStats) =>
          val featuresForNode = nodeToFeaturesBc.value.flatMap { nodeToFeatures =>
            Some(nodeToFeatures(nodeIndex))
          }

          // update stats for each node
          val result =
            STRUTTransfer
              .updateStats(aggStats, splits, featuresForNode, nodes(nodeIndex))
          (nodeIndex, result)
      }
      .filter { _._2.nonEmpty }
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
//        logWarning(s"invert tree ${node.id}")
        val tmp = node.leftChild
        node.leftChild = node.rightChild
        node.rightChild = tmp
      }
//      if (newStats.isEmpty) {
//        logWarning(s"no useful tree ${node.id}")
//        node.isLeaf = true
//        node.leftChild = None
//        node.rightChild = None
//      }
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
    def calcJSD(calcUnderTest: ImpurityCalculator, origCalc: ImpurityCalculator, numClasses: Int): Double = {
      var jsp = 0.0d
      Range(0, numClasses).foreach { cls =>
        {
          val o = calcUnderTest.prob(cls)
          val e = if (cls < origCalc.stats.length) origCalc.prob(cls) else 0
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

    def shouldRecalculateStats(featureIndex: Int): Boolean =
      binAggregates.metadata.isContinuous(featureIndex) &&
        node.split.nonEmpty &&
        node.split.get.featureIndex == featureIndex

    // Calculate InformationGain and ImpurityStats
    // If this node's feature is not numeric, just update prediction metadata
    // otherwise, update threshold.

    // old stats need to be kept for further computation
    val oldStats = node.stats
    var gainAndImpurityStats: ImpurityStats = null

    val validFeatureSplits =
      Range(0, binAggregates.metadata.numFeaturesPerNode)
        .map { featureIndexIdx =>
          featuresForNode
            .map(features => (featureIndexIdx, features(featureIndexIdx)))
            .getOrElse((featureIndexIdx, featureIndexIdx))
        }
        .withFilter {
          case (_, featureIndex) =>
            binAggregates.metadata.numSplits(featureIndex) != 0
        }

//    println(s"validFeatureSplits.map(_ => 1).sum:${validFeatureSplits.map(_ => 1).sum}")

    // update those non continuous features
    val updateThresholdStats = validFeatureSplits.withFilter {
      case (_, featureIndex) => shouldRecalculateStats(featureIndex)
    }

    // For each (feature, split), calculate the gain, and select the best (feature, split).
    val thresholdUpdateInfo =
      updateThresholdStats.flatMap {
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

    if (thresholdUpdateInfo.nonEmpty) {
      val res = thresholdUpdateInfo
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
        //      println(s"res.isEmpty->continuousSplitsAndImpurityInfo:${continuousSplitsAndImpurityInfo.size}")
        if (thresholdUpdateInfo.nonEmpty) {
          val maxGainSplit = thresholdUpdateInfo.maxBy(_._2.gain)
          node.split = Some(maxGainSplit._1)
          node.stats = maxGainSplit._2
        }
        logInfo(s"STRUT Pruning node: ${node.id}, no data reach continuous node")
        node.isLeaf = true
        node.leftChild = None
        node.rightChild = None
      }
      return res
    }

    val updateStatsSplits = validFeatureSplits.withFilter {
      case (_, featureIndex) => !shouldRecalculateStats(featureIndex)
    }

    val nonContinuousSplitAndImpurityInfo =
      updateStatsSplits.map {
        case (featureIndexIdx, featureIndex) =>
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

                  (splitIdx, gainAndImpurityStats)
                }
                .maxBy(_._2.gain)
            (splits(featureIndex)(bestFeatureSplitIndex), bestFeatureGainStats)
          } else if (binAggregates.metadata.isUnordered(featureIndex)) {
            // Unordered categorical feature
            val leftChildOffset = binAggregates.getFeatureOffset(featureIndexIdx)
            val (bestFeatureSplitIndex, bestFeatureGainStats) =
              Range(0, numSplits)
                .map { splitIndex =>
                  val leftChildStats =
                    binAggregates.getImpurityCalculator(leftChildOffset, splitIndex)
                  val rightChildStats = binAggregates
                    .getParentImpurityCalculator()
                    .subtract(leftChildStats)
                  gainAndImpurityStats = calculateImpurityStats(
                    gainAndImpurityStats,
                    leftChildStats,
                    rightChildStats,
                    binAggregates.metadata
                  )
                  (splitIndex, gainAndImpurityStats)
                }
                .maxBy(_._2.gain)
            (splits(featureIndex)(bestFeatureSplitIndex), bestFeatureGainStats)
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

            logDebug(
              "Sorted centroids for categorical variable = " +
                categoriesSortedByCentroid.mkString(",")
            )

            // Cumulative sum (scanLeft) of bin statistics.
            // Afterwards, binAggregates for a bin is the sum of aggregates for
            // that bin + all preceding bins.
            var splitIndex = 0
            while (splitIndex < numSplits) {
              val currentCategory = categoriesSortedByCentroid(splitIndex)._1
              val nextCategory = categoriesSortedByCentroid(splitIndex + 1)._1
              binAggregates.mergeForFeature(nodeFeatureOffset, nextCategory, currentCategory)
              splitIndex += 1
            }
            // lastCategory = index of bin with total aggregates for this (node, feature)
            val lastCategory = categoriesSortedByCentroid.last._1
            // Find best split.
            val (bestFeatureSplitIndex, bestFeatureGainStats) =
              Range(0, numSplits)
                .map { splitIndex =>
                  val featureValue = categoriesSortedByCentroid(splitIndex)._1
                  val leftChildStats =
                    binAggregates.getImpurityCalculator(nodeFeatureOffset, featureValue)
                  val rightChildStats =
                    binAggregates.getImpurityCalculator(nodeFeatureOffset, lastCategory)
                  rightChildStats.subtract(leftChildStats)
                  gainAndImpurityStats = calculateImpurityStats(
                    gainAndImpurityStats,
                    leftChildStats,
                    rightChildStats,
                    binAggregates.metadata
                  )
                  (splitIndex, gainAndImpurityStats)
                }
                .maxBy(_._2.gain)
            val categoriesForSplit =
              categoriesSortedByCentroid.map(_._1.toDouble).slice(0, bestFeatureSplitIndex + 1)
            // 放在categoriesForSplit是走左边 true
            val bestFeatureSplit =
              new CategoricalSplit(featureIndex, categoriesForSplit.toArray, numCategories)
            (bestFeatureSplit, bestFeatureGainStats)
          }
      }

    if (nonContinuousSplitAndImpurityInfo.nonEmpty) {
      // if node's feature is not continuous, only update it's statistics info
      val impurityInfo = nonContinuousSplitAndImpurityInfo.maxBy(_._2.gain)
      // update stats info
      node.split = Some(impurityInfo._1)
      node.stats = impurityInfo._2
      // if no data reach this point
      if (!node.stats.valid || node.stats.gain < 0) {
        // Prune this node
        logInfo(s"STRUT Pruning node:${node.id}")
        node.isLeaf = true
        node.leftChild = None
        node.rightChild = None
      }
    }
    Array()
//    println(s"numFeaturesPerNode:${binAggregates.metadata.numFeaturesPerNode},nonContinuousSplitAndImpurityInfo:${nonContinuousSplitAndImpurityInfo.size},continuousSplits:${continuousSplitsAndImpurityInfo.size}")

//    println(s"binAggregates.metadata.numFeaturesPerNode:${binAggregates.metadata.numFeaturesPerNode}" +
//      s"continuousSplitsAndImpurityInfo.size:${continuousSplitsAndImpurityInfo.size}," +
//      s"nonContinuousSplitAndImpurityInfo.size:${nonContinuousSplitAndImpurityInfo.size}")
  }

}
